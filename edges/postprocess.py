"""Postprocesses and makes plots for the EDGES analysis once the sampling
has been comleted (this is handled in global_sampler.py).
"""
import argparse
import warnings
import yaml
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('Set1')

import numpy as np
import pandas as pd

import site
site.addsitedir('/users/dlenz/software/pymodules/edges/')

from . import utils as ut
from . import plots, models
from . import global_sampler as gs


class Postprocessor(object):
    def __init__(self, config):
        self.config = ut.parse_config(config)

    @property
    def input_df(self):
        """input dataframe"""
        self._input_df = pd.read_csv(
            self.config['DATAFILE'], comment='#')

        return self._input_df

    @property
    def resultdict(self):
        """Results from the MCMC analysis. Include logL, logZ"""

        # Results from the MCMC analysis
        _resultdict = np.load(
            os.path.join(self.config['PROJECT_NAME'], 'output.npz'))

        return _resultdict

    @property
    def rms(self):
        return np.std(self.residual_val)

    @property
    def freq_in(self):
        return self.input_df['freq'].values

    @property
    def freq_min(self):
        return self.config['FMIN']

    @property
    def freq_max(self):
        return self.config['FMAX']

    @property
    def freq_mask(self):
        _mask = (
            (self.freq_in > self.freq_min) &
            (self.freq_in < self.freq_max))

        return _mask

    @property
    def freq_eff(self):
        _freq_eff = self.freq_in[self.freq_mask]

        return _freq_eff

    @property
    def tsky_in(self):
        return self.input_df['tsky'].values

    @property
    def dt21(self):
        return self.input_df['dt21'].values

    @property
    def injection_in(self):
        try:
            return self.input_df['injected'].values
        except KeyError:
            warnings.warn('Input DataFrame has no column with name injected')
            return None

    @property
    def outdir(self):
        self._outdir = self.config['PROJECT_NAME']
        return self._outdir

    @property
    def output(self):
        self._output = dict(np.load(
            os.path.join(self.outdir, 'output.npz')))
        return self._output

    @property
    def chain(self):
        self._chain = self.output['chain'].squeeze()
        return self._chain

    @property
    def flatchain(self):
        if self.is_pt:
            self._flatchain = self.chain[0].reshape((-1, self.ndim))
        else:
            self._flatchain = self.chain.reshape((-1, self.ndim))
        return self._flatchain

    @property
    def p_ml(self):
        with open(os.path.join(self.outdir, 'ml_params.yaml'), 'r') as ymlfile:
            self._p_ml = yaml.load(ymlfile)

        return self._p_ml

    @property
    def posterior_mean(self):
        return self.output['post_mean_vals']

    @property
    def fg_model(self):
        if self.fg_type == 'polynomial':
            return models.model_fg_poly

        elif self.fg_type == 'physical':
            raise NotImplementedError('Physical model not implemented yet.')

    @property
    def fg_model_val(self):
        return self.fg_model(
            self.freq_eff,
            *self.posterior_mean[-self.polyorder:])

    @property
    def model_val(self):
        return self.fg_model_val + self.signal_model_val

    @property
    def residual_val(self):
        return self.tsky_in[self.freq_mask] - self.model_val

    @property
    def signal_model(self):
        if self.sig_type == 'boxgauss':
            return lambda f, x: gs.delta_Tb_analytic(f, *x[:4])

        elif self.sig_type == 'none':
            return lambda x: np.zeros_like(x)

        else:
            raise KeyError('sig_type not in ["boxgauss", "none"]')

    @property
    def signal_model_val(self):
        return self.signal_model(self.freq_eff, self.posterior_mean)

    @property
    def ndim(self):
        return len(self.parameters_vary)

    @property
    def logL(self):
        val = self.output.get('logL')
        if val is not None:
            val = np.asscalar(val)
        return val

    @property
    def log_thd_evidence(self):
        self._log_thd_evidence = self.output.get('log_thd_evidence')
        return self._log_thd_evidence

    @property
    def dlog_thd_evidence(self):
        self._dlog_thd_evidence = self.output.get('dlog_thd_evidence')
        return self._dlog_thd_evidence

    @property
    def samplertype(self):
        if self.config['SAMPLER'].lower() == 'paralleltempering':
            self._samplertype = 'pt'
        elif self.config['SAMPLER'].lower() == 'ensemblesampler':
            self._samplertype = 'ensemble'

        return self._samplertype

    @property
    def is_pt(self):
        if self.samplertype == 'pt':
            return True
        else:
            return False

    @property
    def parameters_vary(self):
        p = [
            k for k, v in self.config['PARAMS'].items() if v['TYPE'] == 'VARY']
        return p

    @property
    def parameters_const(self):
        p = [
            k for k, v in self.config['PARAMS'].items() if v['TYPE'] != 'VARY']
        return p

    @property
    def parameter_names(self):
        return self.parameters_const + self.parameters_vary

    @property
    def polyorder(self):
        orders = [int(s[-1:]) for s in self.parameter_names if s[:3] == 'AFG']
        return np.amax(orders)+1

    @property
    def fg_type(self):
        if self.config['FGMODL'].lower() == 'polynomial':
            return 'polynomial'
        elif self.config['FGMODL'].lower() == 'physical':
            return 'physical'
        raise NameError('fg_type must be in ["polynomial", "physical"].')

    @property
    def sig_type(self):
        if self.config['SIGMODL'].lower() == 'boxgauss':
            return 'boxgauss'
        elif self.config['SIGMODL'].lower() == 'none':
            return 'none'
        raise NameError('sig_type must be in ["boxgauss", "none"].')

    @property
    def parameter_labels(self):
        """These are in the same order as the chain"""

        # Labels for the signal
        if self.sig_type == 'boxgauss':
            sig_labels = ['nu0', 'tau', 'A', 'w']
        elif self.sig_type == 'none':
            sig_labels = []

        # Labels for the foregrounds
        if self.fg_type == 'polynomial':
            fg_labels = [f'a{i}' for i in range(self.polyorder)]
        elif self.fg_type == 'physical':
            fg_labels = []

        labels = sig_labels + fg_labels
        return labels

    @property
    def input_fg(self):
        if 'fg' in self.input_df.keys():
            return self.input_df['fg'].values
        else:
            return None

    @property
    def input_t21(self):
        if 't21' in self.input_df.keys():
            return self.input_df['t21'].values
        else:
            return None

    @property
    def is_sim(self):
        if 'fg' in self.input_df.keys():
            return True
        else:
            return False


# Allow execution as a script.
def main():
    desc = """Postprocessing for the EDGES data.
            To run:
            python postprocess.py -c <config_file>
            """

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c', '--config', help='configuration file')

    parser.add_argument(
        '-v', '--verbose',
        help='print more output', action='store_true')
    # parser.add_argument('-p','--progress',
    # help='show progress bar',action='store_true')

    # Initiate postprocessing
    args = parser.parse_args()
    pp = Postprocessor(args.config)
    outdir = pp.outdir

    # Plot sample chain
    plots.plot_example_chain(
        pp.chain, pt=pp.is_pt,
        outname=os.path.join(outdir, 'example_chain.pdf'))

    # Plot triangle
    # plots.plot_triangle(
    #     pp.samples,
    #     labels=pproc.parameter_labels)



    return 0


if __name__ == "__main__":
    main()
