'''
MCMC driver for global-signal black-holes model.
To run:
mpirun -np 2 python global_signal_black_holes_mcmc.py -i <config_file>
'''
import copy
import sys
import os
import yaml
import argparse

import scipy.signal as signal
import scipy.interpolate as interp
import scipy.optimize as op
import numpy as np

try:
    from emcee import emcee
except ImportError:
    import emcee

import ptemcee

F21 = 1420405751.7667  # 21 cm frequency


def delta_Tb_analytic(freq, **kwargs):
    '''
    Analytic function describing delta T_b
    '''

    B = 4.*((freq-kwargs['NU0'])/kwargs['W'])**2.\
    *np.log(-1./kwargs['TAU'] *
    np.log((1.+np.exp(-kwargs['TAU']))/2.))
    return -kwargs['A']*(1-np.exp(-kwargs['TAU']*np.exp(B)))\
    /(1.-np.exp(-kwargs['TAU']))


def var_resid(resid_array, window_length=20):
    '''
    estimate rms noise in residuals (resid_array) by taking the running average
    of abs(resid-<resid>)**2.
    Args:
        resid_array: array of residuals (preferably mean zero)
        window_length: number of points to estimate rms from
    Returns:
        array of standard deviations at each position in array, estimated by
        taking rms of surrounding window_lenth points.
    '''
    window = np.zeros_like(resid_array)
    nd = len(resid_array)
    if np.mod(nd, 2) == 1:
        nd = nd-1
    iupper = int(nd/2+window_length/2)
    ilower = int(nd/2-window_length/2)
    window[ilower:iupper] = 1./window_length
    return signal.fftconvolve(
        window,
        np.abs(resid_array-np.mean(resid_array))**2., mode='same')


def Tbfg(x, params_dict):
    y_model = 0
    for param in params_dict.keys():
        if 'AFG' in param:
            polynum = int(param[3:])
            y_model += params_dict[param]*(x/x[int(len(x)/2)])**(polynum-2.5)
    return y_model


def Tb21(x, params_dict):
    y_model = delta_Tb_analytic(x, **params_dict)
    return y_model


def Tbfg_physical(x,params):
    '''
    physical foreground model
    x is frequency in units of MHz.
    params: dictionary of parameters
    '''
    x150=x/150.
    fgs=params['AFG0']*(x150)**(params['AFG1']+params['AFG2']*np.log(x150)**2.)
    fgs=fgs*(params['AFG3'])**(x150**-2.)+params['AFG4']*x150**-2.
    return fgs

def TbSky(
        params, x, params_dict, param_list,
        fg_model='POLYNOMIAL', sig_model='BOXGAUSS'):
    param_instance = copy.deepcopy(params_dict)
    for param, param_key in zip(params, param_list):
        param_instance[param_key] = param

    # Calculate the 21cm signal component
    if sig_model.lower() == 'boxgauss':
        y_model = Tb21(x, param_instance)
    elif sig_model.lower() == 'none':
        y_model = np.zeros_like(x, dtype=np.double)
    else:
        raise NameError('sig_model must be in ["BOXGAUSS", "NONE"]')

    # Calculate the foreground component
    if fg_model.lower() == 'polynomial':
        y_model = y_model + Tbfg(x, param_instance)
    elif fg_model.lower() == 'physical':
        y_model = y_model + Tbfg_physical(x, param_instance)
    else:
        raise NameError('fg_model must be in ["POLYNOMIAL", "PHYSICAL"]')
    return y_model


def lnlike(
        params, x, y, yvar, param_template, param_list,
        fg_model, sig_model):
    '''
    log-likelihood of parameters
    Args:
        params, instance of parameters defined in params_vary
        x, measured frequencies
        y, measured dTb
        yvar, measured error bars
    '''
    # Run heating
    y_model = TbSky(
        params, x, param_template, param_list,
        fg_model=fg_model,
        sig_model=sig_model)

    return -np.sum(0.5*(y_model-y)**2./yvar)


# Construct a prior for each parameter.
# Priors can be Gaussian, Log-Normal, or Uniform
def lnprior(params, param_list, param_priors):
    '''
    Compute the lnprior distribution for params whose prior-distributions
    are specified in the paramsv_priors dictionary (read from input yaml file)
    Priors supported are Uniform, Gaussian, or Log-Normal. No prior specified
    will result in no prior distribution placed on a given parameter.
    '''
    output = 0.
    for param, param_key in zip(params, param_list):
        if param_priors[param_key]['PRIOR'] == 'UNIFORM':
            if (
                    param <= param_priors[param_key]['MIN'] or
                    param >= param_priors[param_key]['MAX']):
                output -= np.inf
        elif param_priors[param_key]['PRIOR'] == 'GAUSSIAN':
            var = param_priors[param_key]['VAR']
            mu = param_priors[param_key]['MEAN']
            output += -.5*((param-mu)**2./var-np.log(2.*PI*var))
        elif param_priors[param_key]['PRIOR'] == 'LOGNORMAL':
            var = param_priors[param_key]['VAR']
            mu = param_priors[param_key]['MEAN']
            output += (
                -.5*((np.log(param)-mu)**2./var -
                np.log(2.*PI*var)) - np.log(param))

    return output


def lnprob(
        params, x, y, yvar, param_template, param_list, param_priors,
        fg_model, sig_model):
    lp = lnprior(params, param_list, param_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(
        params, x, y, yvar, param_template, param_list,
        fg_model, sig_model)


class Sampler():
    '''
    Class for running MCMC and storing output.
    '''
    def __init__(self, config, verbose=False):
        '''
        Initialize the sampler.
        Args:
            config_file, string with name of the config file.
        '''
        self.verbose = verbose
        self.minimized = False
        self.ln_ml = -np.inf
        self.sampled = False

        # Read config information for this model
        if type(config) is dict:
            self.config = config
        elif type(config) is str:
            with open(config, 'r') as ymlfile:
                self.config = yaml.load(ymlfile)
        else:
            raise TypeError('config must be either dict or string (filename).')

        # Read in measurement file
        # Assume first column is frequency, second column is
        # measured brightness temp and third column is the residual from
        # fitting an empirical model (see Bowman+ 2018).
        if self.config['DATAFILE'].endswith('.csv'):
            self.data = np.loadtxt(
                self.config['DATAFILE'],
                skiprows=1, delimiter=',')
        elif self.config['DATAFILE'].endswith('.npy'):
            self.data = np.load(self.config['DATAFILE'])

        self.data = np.array(self.data)

        # Constrain frequency range that is used for the fit
        select = (
            (self.data[:, 0] >= self.config['FMIN']) &
            (self.data[:, 0] <= self.config['FMAX']))

        self.data = self.data[select, :]
        self.freqs, self.tb_meas, self.dtb = (
                                self.data[:, 0],
                                self.data[:, 1],
                                self.data[:, 2])

        # Calculate std of residuals read list of parameters to vary from
        # config file, and set all other parameters to default starting values
        self.var_tb = var_resid(
            self.dtb,
            window_length=self.config['NPTS_NOISE_EST'])

        # Extract the foreground and signal model type
        self.fg_model = self.config['FGMODL']
        self.sig_model = self.config['SIGMODL']

        self.params = self.config['PARAMS']
        self.params_all = {}
        self.params_vary = {}
        # Populate params_all
        for param in self.params:
            if self.params[param]['TYPE'].lower() == 'vary':
                self.params_all[param] = self.params[param]['P0']
                self.params_vary[param] = self.params[param]
        self.ndim = len(self.params_vary)
        self.resid = np.zeros_like(self.var_tb)
        self.model = np.zeros_like(self.resid)

    def gradient_descent(self, param_list=None):
        '''
        perform gradient descent on parameters specified in param_list
        Args:
            param_list, list of parameter names to perform gradient
            descent on while holding all other parameters fixed.
            update the parameters in config_all.
        '''
        if param_list is None:
            param_list = self.params_vary

        nll = lambda *args: -lnprob(*args)
        result = op.minimize(
            nll,
            [self.params_all[pname] for pname in param_list],
            args=(
                self.freqs, self.tb_meas,
                self.var_tb, self.params_all, param_list, self.params_vary,
                self.fg_model, self.sig_model))["x"]

        for pnum, pname in enumerate(param_list):
            self.params_all[pname] = result[pnum]

        self.model = TbSky(
            result, self.freqs, self.params_all, [],
            self.fg_model, self.sig_model)
        self.ml_params = result
        self.resid = self.tb_meas-self.model
        self.ln_ml = lnprob(
            result, self.freqs, self.tb_meas, self.var_tb,
            self.params_all, self.params_vary.keys(), self.params_vary,
            self.fg_model, self.sig_model)

    def approximate_ml(self):
        if 'AFG0' in self.params_all.keys():
            params_nofg = []
            params_fg = []
            for pname in self.params_vary:
                if 'AFG' not in pname:
                    params_nofg = params_nofg+[pname]
                else:
                    params_fg = params_fg+[pname]
            # Perform gradient descent on foregrounds first
            self.gradient_descent(params_fg)
            # Perform gradient descent on no-fg params. If that list does not
            # exists, simply skip this step.
            if params_nofg:
                self.gradient_descent(params_nofg)
        # Perform gradient descent on all params
        self.gradient_descent()
        self.minimized = True

    def sample(self):
        '''
        Run the MCMC.
        '''
        # First make sure that the maximum likelihood params are fitted
        if not self.minimized:
            self.approximate_ml()
        # print(self.params_all)

        ndim, nwalkers = len(self.params_vary), self.config['NWALKERS']
        p0 = np.zeros((nwalkers, len(self.params_vary)))
        pml = [self.params_all[pname] for pname in self.params_vary]

        for pnum, pname in enumerate(self.params_vary):
            p0[:, pnum] = (np.random.randn(nwalkers)\
            * self.config['SAMPLE_BALL']+1.)*pml[pnum]

        plist = []

        for key in self.params_vary.keys():
            plist.append(key)

        args = (
            self.freqs, self.tb_meas, self.var_tb,
            self.params_all, plist, self.params_vary,
            self.fg_model, self.sig_model)

        if self.config['MPI']:
            from emcee.utils import MPIPool
            pool = MPIPool()

            if not pool.is_master():
                pool.wait()
                sys.exit(0)
                self.sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, lnprob,
                    args=args, pool=pool)

            self.sampler.run_mcmc(p0, self.config['NBURN'])  # burn in

            p0 = self.sampler.chain[:, -1, :].squeeze()

            self.sampler.reset()
            self.sampler.run_mcmc(p0, self.config['NSTEPS'])
            pool.close()
        else:
            if self.config['SAMPLER'] == 'PARALLELTEMPERING':
                logl = lambda x: lnlike(
                    x, self.freqs, self.tb_meas,
                    self.var_tb, self.params_all, self.params_vary,
                    self.fg_model, self.sig_model)

                logp = lambda x: lnprior(
                    x, self.params_vary.keys(), self.params_vary)

                self.sampler = ptemcee.Sampler(
                    ntemps=self.config['NTEMPS'],
                    nwalkers=self.config['NWALKERS'],
                    dim=self.ndim,
                    logl=logl,
                    logp=logp)
            else:
                self.sampler = emcee.EnsembleSampler(
                        nwalkers=self.config['NWALKERS'],
                        ndim=ndim,
                        log_prob_fn=lnprob,
                        args=args,
                        threads=self.config['THREADS'])

            # If we use PT sampling, we need a further dimension of
            # start parameters for the different temperatures
            if self.config['SAMPLER'] == 'PARALLELTEMPERING':
                p0 = np.array([p0 for m in range(self.config['NTEMPS'])])

            # Run the MCMC for the burn-in
            self.sampler.run_mcmc(
                p0,
                self.config['NBURN'],
                thin=self.config['NTHIN'])

            # Reset after burn-in and run the full chain
            if self.config['SAMPLER'] == 'PARALLELTEMPERING':
                p0 = self.sampler.chain[:, :, -1, :]
            else:
                p0 = self.sampler.chain[:, -1, :].squeeze()
            self.sampler.reset()
            self.sampler.run_mcmc(
                p0, self.config['NSTEPS'], thin=self.config['NTHIN'])

        # Create output directory
        if not os.path.exists(self.config['PROJECT_NAME']):
            os.makedirs(self.config['PROJECT_NAME'])

        # Save output and configuration
        with open(os.path.join(
                self.config['PROJECT_NAME'], 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        with open(os.path.join(
                self.config['PROJECT_NAME'], 'ml_params.yaml'), 'w') as f:
            yaml.dump(self.params_all, f, default_flow_style=False)

        self.sampled = True

        # Collect result parameters
        resultdict = {}

        resultdict['chain'] = self.sampler.chain,

        if (
                self.config['COMPUTECOVARIANCE'] &
                (self.config['SAMPLER'] == 'ENSEMBLESAMPLER')):

            # Estimate autocorrelation
            self.acors = self.sampler.acor.astype(int)
            resultdict['autocorrs'] = self.acors

            # Estimate covariance
            self.cov_samples = np.zeros((
                len(self.params_vary),
                len(self.params_vary)))
            resultdict['cov_samples'] = self.cov_samples

            for i in range(len(self.params_vary)):
                for j in range(len(self.params_vary)):
                    stepsize = np.max([self.acors[i], self.acors[j]])
                    csample_i = self.sampler.chain[i, ::stepsize, :].flatten()
                    csample_j = self.sampler.chain[j, ::stepsize, :].flatten()
                    self.cov_samples[i, j] = np.mean(
                        (csample_i-csample_i.mean()) *
                        (csample_j-csample_j.mean()))

            # Compute conservative evidence without prior factor
            self.conservative_evidence = np.exp(self.ln_ml)/np.sqrt(
                np.linalg.det(self.cov_samples))
            resultdict['conservative_evidence'] = self.conservative_evidence

        # Save the evidence from thermodynamic integration from the
        # PT sampler
        if self.config['SAMPLER'].lower() == 'paralleltempering':
            self.logz, self.dlogz = self.sampler.log_evidence_estimate(
                fburnin=0.)

            resultdict['log_thd_evidence'] = self.logz
            resultdict['dlog_thd_evidence'] = self.dlogz

        # Save as .npz
        np.savez(
            os.path.join(self.config['PROJECT_NAME'], 'output.npz'),
            **resultdict)


'''
Allow execution as a script.
'''
if __name__ == "__main__":
    desc=('MCMC driver for fitting edges data.\n'
          'To run: mpirun -np <num_processes>'
          'python global_sampler.py -c <config_file>')

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c', '--config', help='configuration file')

    parser.add_argument(
        '-v', '--verbose',
        help='print more output', action='store_true')
    # parser.add_argument('-p','--progress',
    # help='show progress bar',action='store_true')
    args = parser.parse_args()
    my_sampler = Sampler(args.config)
    my_sampler.sample()
