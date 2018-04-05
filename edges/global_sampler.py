'''
MCMC driver for global-signal black-holes model.
To run:
mpirun -np 2 python global_signal_black_holes_mcmc.py -i <config_file>
'''
import scipy.signal as signal
import scipy.interpolate as interp
import numpy as np
import yaml, argparse, yaml
import emcee
emcee_v=emcee.__version__.split('.')[0]
if int(emcee_v[0])>=3:
    from emcee import emcee

F21=1420405751.7667#21 cm frequency.
import copy,sys,os
import scipy.optimize as op


def delta_Tb_analytic(freq,**kwargs):
    '''
    Analytic function describing delta T_b
    '''

    B=4.*((freq-kwargs['NU0'])/kwargs['W'])**2.\
    *np.log(-1./kwargs['TAU']*\
    np.log((1.+np.exp(-kwargs['TAU']))/2.))
    return -kwargs['A']*(1-np.exp(-kwargs['TAU']*np.exp(B)))\
    /(1.-np.exp(-kwargs['TAU']))


def var_resid(resid_array,window_length=20):
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
    window=np.zeros_like(resid_array)
    nd=len(resid_array)
    if np.mod(nd,2)==1:
        nd=nd-1
    iupper=int(nd/2+window_length/2)
    ilower=int(nd/2-window_length/2)
    window[ilower:iupper]=1./window_length
    return signal.fftconvolve(window,
    np.abs(resid_array-np.mean(resid_array))**2.,mode='same')

def Tbfg(x,params_dict):
    y_model=params_dict['APOLY0']*(x/x[int(len(x)/2)])**(-2.5)
    for n in range(1,int(params_dict['NPOLY'])):
        y_model+=params_dict['APOLY'+str(n)]*(x/x[int(len(x)/2)])**(n-2.5)
    return y_model

def Tb21(x,params_dict):
    y_model=delta_Tb_analytic(x,**params_dict)
    return y_model

def TbSky(params,x,params_dict,param_list):
    param_instance=copy.deepcopy(params_dict)
    for param,param_key in zip(params,param_list):
        param_instance[param_key]=param
    y_model=Tb21(x,param_instance)\
    +Tbfg(x,param_instance)
    return y_model

def lnlike(params,x,y,yvar,param_template,param_list):
    '''
    log-likelihood of parameters
    Args:
        params, instance of parameters defined in params_vary
        x, measured frequencies
        y, measured dTb
        yvar, measured error bars
    '''
    #run heating
    y_model=TbSky(params,x,param_template,param_list)
    return -np.sum(0.5*(y_model-y)**2./yvar)

#Construct a prior for each parameter.
#Priors can be Gaussian, Log-Normal, or Uniform
def lnprior(params,param_list,param_priors):
    '''
    Compute the lnprior distribution for params whose prior-distributions
    are specified in the paramsv_priors dictionary (read from input yaml file)
    Priors supported are Uniform, Gaussian, or Log-Normal. No prior specified
    will result in no prior distribution placed on a given parameter.
    '''
    output=0.
    for param,param_key in zip(params,param_list):
        if param_priors[param_key]['TYPE']=='UNIFORM':
            if param <= param_priors[param_key]['MIN'] or \
            param >= param_priors[param_key]['MAX']:
                 output-=np.inf
        elif param_priors[param_key]['TYPE']=='GAUSSIAN':
            var=param_priors[param_key]['VAR']
            mu=param_priors[param_key]['MEAN']
            output+=-.5*((param-mu)**2./var-np.log(2.*PI*var))
        elif param_priors[param_key]['TYPE']=='LOGNORMAL':
            var=param_priors[param_key]['VAR']
            mu=param_priors[param_key]['MEAN']
            output+=-.5*((np.log(param)-mu)**2./var-np.log(2.*PI*var))\
            -np.log(param)
    return output

def lnprob(params,x,y,yvar,param_template,param_list,param_priors):
    lp=lnprior(params,param_list,param_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(params,x,y,yvar,param_template,param_list)


class Sampler():
    '''
    Class for running MCMC and storing output.
    '''
    def __init__(self,config_file,verbose=False):
        '''
        Initialize the sampler.
        Args:
            config_file, string with name of the config file.
        '''
        self.verbose=verbose
        self.minimized=False
        self.ln_ml=-np.inf
        self.sampled=False
        with open(config_file, 'r') as ymlfile:
            self.config= yaml.load(ymlfile)
        ymlfile.close()
        #read in measurement file
        #Assume first column is frequency, second column is measured brightness temp
        #and third column is the residual from fitting an empirical model
        #(see Bowman 2018)
        if self.config['DATAFILE'][-3:]=='csv':
            self.data=np.loadtxt(self.config['DATAFILE'],
            skiprows=1,delimiter=',')
        elif self.config['DATAFILE'][-3:]=='npy':
            self.data=np.load(self.config['DATAFILE'])
        select=self.data[:,0]>=self.config['FMIN']
        self.data=self.data[select,:]
        select=self.data[:,0]<=self.config['FMAX']
        self.data=self.data[select,:]
        self.freqs,self.tb_meas,self.dtb\
        =self.data[:,0],self.data[:,1],self.data[:,2]
        self.var_tb=var_resid(self.dtb,
        window_length=self.config['NPTS_NOISE_EST'])#Calculate std of residuals
        #read list of parameters to vary from config file,
        #and set all other parameters to default starting values
        self.params_all=self.config['PARAMS']
        self.params_vary=self.config['PARAMS2VARY']
        self.params_vary_priors=self.config['PRIORS']
        self.resid=np.zeros_like(self.var_tb)
        self.model=np.zeros_like(self.resid)

    def gradient_descent(self,param_list=None):
        '''
        perform gradient descent on parameters specified in param_list
        Args:
            param_list, list of parameter names to perform gradient
            descent on while holding all other parameters fixed.
            update the parameters in config_all.
        '''
        if param_list is None:
            param_list=self.params_vary
        nll = lambda *args: -lnlike(*args)
        result = op.minimize(nll,
        [self.params_all[pname] for pname in param_list],
        args=(self.freqs,self.tb_meas,self.var_tb,
        self.params_all,param_list))["x"]
        for pnum,pname in enumerate(param_list):
            self.params_all[pname]=result[pnum]
        self.model=TbSky(result,self.freqs,self.params_all,[])
        self.ml_params=result
        self.resid=self.tb_meas-self.model
        self.ln_ml=lnprob(result,self.freqs,self.tb_meas,
        self.var_tb,self.params_all,param_list,self.params_vary_priors)


    def approximate_ml(self):
        if self.params_all['NPOLY']>0:
            params_nofg=[]
            params_fg=[]
            for pname in self.params_vary:
                if 'APOLY' not in pname:
                    params_nofg=params_nofg+[pname]
                else:
                    params_fg=params_fg+[pname]
            #perform gradient descent on foregrounds first
            self.gradient_descent(params_fg)
            #perform gradient descent on no-fg params
            self.gradient_descent(params_nofg)
            #perform gradient descent on all params
        self.gradient_descent()
        self.minimized=True




    def sample(self):
        '''
        Run the MCMC.
        '''
        #first make sure that the maximum likelihood params are fitted
        if not self.minimized:
            self.approximate_ml()
        print(self.params_all)
        ndim,nwalkers=len(self.params_vary),self.config['NWALKERS']
        p0=np.zeros((nwalkers,len(self.params_vary)))

        pml=[self.params_all[pname] for pname in self.params_vary]
        for pnum,pname in enumerate(self.params_vary):
            p0[:,pnum]=(np.random.randn(nwalkers)\
            *self.config['SAMPLE_BALL']+1.)*pml[pnum]
        args=(self.freqs,self.tb_meas,self.var_tb,
        self.params_all,self.params_vary,
        self.params_vary_priors)
        if self.config['MPI']:
            from emcee.utils import MPIPool
            pool=MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            self.sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,
            args=args,pool=pool)
            self.sampler.run_mcmc(p0,self.config['NBURN'])#burn in
            p0=self.sampler.chain[:,-1,:].squeeze()
            self.sampler.reset()
            self.sampler.run_mcmc(p0,self.config['NSTEPS'])
            pool.close()
        else:
            self.sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,
            args=args,threads=self.config['THREADS'])
            self.sampler.run_mcmc(p0,self.config['NBURN'])#burn in
            p0=self.sampler.chain[:,-1,:].squeeze()
            self.sampler.reset()
            self.sampler.run_mcmc(p0,self.config['NSTEPS'])
        if not os.path.exists(self.config['PROJECT_NAME']):
            os.makedirs(self.config['PROJECT_NAME'])
        #save output and configuration
        with open(self.config['PROJECT_NAME']+'/config.yaml','w')\
         as yaml_file:
            yaml.dump(self.config,yaml_file,default_flow_style=False)
        yaml_file.close()
        with open(self.config['PROJECT_NAME']+'/ml_params.yaml','w')\
         as yaml_file:
            yaml.dump(self.params_all,yaml_file,default_flow_style=False)
        yaml_file.close()
        self.sampled=True
        self.acors=self.sampler.acor.astype(int)
        #estimate covariance
        self.cov_samples=np.zeros((len(self.params_vary),
        len(self.params_vary)))
        for i in range(len(self.params_vary)):
            for j in range(len(self.params_vary)):
                stepsize=np.max([self.acors[i],self.acors[j]])
                csample_i=self.sampler.chain[i,::stepsize,:].flatten()
                csample_j=self.sampler.chain[j,::stepsize,:].flatten()
                self.cov_samples[i,j]=np.mean((csample_i-csample_i.mean())\
                *(csample_j-csample_j.mean()))
        self.evidence=np.exp(self.ln_ml)/np.sqrt(np.linalg.det(self.cov_samples))#compute conservative evidence without prior factor
        np.save(self.config['PROJECT_NAME']+'/output.npz',
        chain=self.sampler.chain,evidence=self.evidence,cov_samples=self.cov_samples,autocorrs=self.acors)

'''
Allow execution as a script.
'''
if __name__ == "__main__":

    desc=('MCMC driver for fitting edges data.\n'
          'To run: mpirun -np <num_processes>'
          'python global_sampler.py -c <config_file>')
    parser=argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--config',
    help='configuration file')
    parser.add_argument('-v','--verbose',
    help='print more output',action='store_true')
    #parser.add_argument('-p','--progress',
    #help='show progress bar',action='store_true')
    args=parser.parse_args()
    my_sampler=Sampler(args.config)
    my_sampler.sample()
