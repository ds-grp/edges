import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import corner

from . import global_sampler as gs

sns.set_palette('pastel')
sns.set_style('whitegrid')


def plot_triangle(samples, labels, outname=None):
    """Generates a triangle plot from the MCMC chain

    Parameters
    ----------

    samples : 2D-array, shape (nsamples, ndim)
        MCMC samples
    labels : list of labels, length ndim
        Parameter labels for the corner plot
    outname : str, optional
        If not None, save plot at this location

    """

    corner.corner(
        samples,
        labels=labels)

    if outname is not None:
        plt.savefig(outname, dpi=300)


def plot_example_chain(chain, pt=True, outname=None):
    if pt:
        for temp in range(chain.shape[0]):
            # One panel per temperature (increasing in temperature)
            # Walker index 0 only
            # All samples, shown on x-axis
            # parameter with index 2
            plt.figure(figsize=(15, 4))
            plt.plot(chain[temp, 0, :, 2], alpha=0.5)
    else:
            # Every 5th walker
            # All samples, shown on x-axis
            # parameter with index 2
            plt.figure(figsize=(15, 4))
            [plt.plot(c[:, 2], alpha=0.3, c='k') for c in chain[::5]]

    if outname is not None:
        plt.savefig(outname, dpi=300)


class SpectrumPlot(object):
    """Takes the postprocessing object from postprocess.py to plot the
    model and residual"""

    _fig_residual = None
    _fig_tsky = None

    _ax_residual = None
    _ax_tsky = None

    def __init__(self, pproc, figsize=(8, 6)):
        """

        Parameters
        ----------
        pproc : postprocessing object from postprocess.py

        """
        self.pproc = pproc
        self.figsize = figsize

        # self.plot_signal()
        # self.plot_residual()
        # plt.show()

    @property
    def ax_residual(self):
        if self._ax_residual is None:
            self.make_residual_figure()
        return self._ax_residual

    @property
    def fig_residual(self):
        if self._fig_residual is None:
            self.make_residual_figure()
        return self._fig_residual

    @property
    def fig_tsky(self):
        if self._fig_tsky is None:
            self.make_tsky_figure()
        return self._fig_tsky

    @property
    def ax_tsky(self):
        if self._ax_tsky is None:
            self.make_tsky_figure()
        return self._ax_tsky

    def make_residual_figure(self):
        self._fig_residual, self._ax_residual = plt.subplots(
            figsize=self.figsize)

        self._ax_residual.set_xlabel(r'$\nu\ [\rm MHz]$')
        self._ax_residual.set_ylabel(r'$T_B\ [\rm K]$')

    def make_tsky_figure(self):
        self._fig_tsky, self._ax_tsky = plt.subplots(figsize=self.figsize)
        self._ax_tsky.set_xlabel(r'$\nu\ [\rm MHz]$')
        self._ax_tsky.set_ylabel(r'$T_B\ [\rm K]$')

    @property
    def freq_in(self):
        return self.pproc.freq_in

    @property
    def freq_eff(self):
        return self.pproc.freq_eff

    @property
    def freq_mask(self):
        return self.pproc.freq_mask

    @property
    def tsky_in(self):
        return self.pproc.tsky_in

    @property
    def chain(self):
        return self.pproc.chain

    @property
    def flatchain(self):
        return self.pproc.flatchain

    def plot_signal(self, n_samples=10, bestfit=True):

        # Extract the random samples
        random_samples = self.flatchain[
            np.random.choice(
                self.flatchain.shape[0], n_samples, replace=False)]

        for i, random_sample in enumerate(random_samples):
            if i == 0:
                label = 'Trough samples'
            else:
                label = None
            self.ax_residual.plot(
                self.freq_eff,
                self.pproc.signal_model(self.freq_eff, random_sample),
                alpha=0.2,
                c='k',
                label=label)

        if bestfit:
            self.ax_residual.plot(
                self.freq_eff, self.pproc.signal_model_val,
                c='k', label='Best-fit trough')

        self.ax_residual.legend()

    def plot_residual(self):
        self.ax_residual.plot(
            self.freq_eff, self.pproc.residual_val, c='blue', label='Residual')
        self.ax_residual.legend()

    def plot_injection(self):
        if self.pproc.injection_in is not None:
            self.ax_residual.plot(
                self.freq_in, self.pproc.injection_in,
                label='Injection', c='orange')
            self.ax_residual.legend()

    def plot_tsky(self):
        self.ax_tsky.plot(
            self.freq_in,
            self.tsky_in,
            c='k',
            label='Tsky in'
        )
        self.ax_tsky.legend()

    def plot_model(self):
        self.ax_tsky.plot(
            self.freq_eff,
            self.pproc.model_val,
            c='blue',
            label='Total model'
        )
        self.ax_tsky.legend()

    def plot_fg_model(self):
        self.ax_tsky.plot(
            self.freq_eff,
            self.pproc.fg_model_val,
            c='red',
            label='Foreground model')
        self.ax_tsky.legend()

    def plot_signalplusresidual(self):
        self.ax_residual.plot(
            self.freq_eff,
            self.pproc.signal_model_val + self.pproc.residual_val,
            c='red',
            label='Trough + residual')

        self.ax_residual.legend()

    def save_plots(self, outdir):

        # T_sky
        plt.figure(self.fig_tsky.number)
        plt.savefig(os.path.join(
            outdir, 't_sky.pdf'), dpi=300)

        # dT
        plt.figure(self.fig_residual.number)
        plt.savefig(os.path.join(
            outdir, 'residual.pdf'), dpi=300)
