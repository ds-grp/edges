import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import corner

from . import global_sampler as gs

sns.set_palette('pastel')
sns.set_style('whitegrid')


def plot_sim(freqs, fg=None, signal=None, noise=None, outname=None):
    """Plot a simulated sky signal, consisting of a required frequency array
    and optionally foreground, signal, and noise."""

    xlabel = r'$\nu\ [\rm MHz]$'
    ylabel = r'$T_B\ [\rm K]$'

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), ncols=3)

    total = np.sum(
        [arr for arr in [fg, signal, noise] if arr is not None],
        axis=0)

    if fg is not None:
        ax1.set_title('Foreground and total')
        ax1.plot(freqs, fg, label='Foreground')
        ax1.plot(freqs, total*1.1, label='Total x 1.1')
        ax1.legend()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)

    if signal is not None:
        ax2.set_title('Signal')
        ax2.plot(freqs, signal, label='Signal')
        ax2.legend()
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)

    if signal is not None:
        ax3.set_title('Noise')
        ax3.plot(freqs, noise, label='Noise')
        ax3.legend()
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)

    if outname is not None:
        plt.savefig(outname, dpi=300)


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


def plot_all_chains(chains, pt=False, outname=None):
    n_chains,  = chains.shape
    for chain in chains:
        # Every 5th walker
        # All samples, shown on x-axis
        plt.figure(figsize=(15, 4))
        [plt.plot(c[:, 2], alpha=0.3, c='k') for c in chain[::5]]

    if outname is not None:
        plt.savefig(outname, dpi=300)


def plot_example_chain(chain, pt=False, outname=None):
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

    def __init__(self, pproc, figsize=(7, 5)):
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

    @property
    def result_str(self):
        s = (
            f"log L = {self.pproc.logL:.2f}\n"
            f"RMS = {self.pproc.rms*1000:.2f} mK")

        return s

    def full_residual_plot(self):
        """Makes all the relevant plots for the residual plot"""

        self.plot_residual()
        self.plot_signal(n_samples=10, bestfit=True)
        self.plot_injection()
        # self.ax_residual.set_ylim(-1., 1.)
        self.plot_signalplusresidual()
        # self.plot_numin_numax()
        self.add_infotext()

    def full_tsky_plot(self):
        """Makes all the relevant plots for the t_sky plot"""
        self.plot_tsky()
        self.plot_fg_model()
        self.plot_model()

    def residual_plot(self, freq, spec, label=None):
        self.ax_residual.plot(
            freq, spec, label=label, ls='--')

    def tsky_plot(self, freq, spec, label=None):
        self.ax_tsky.plot(
            freq, spec, label=label, ls='--')

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

    def plot_numin_numax(self):
        """Add vertical lines to indicate the fit range in frequency"""
        self.ax_residual.vlines(
            [self.pproc.freq_eff[0], self.pproc.freq_eff[-1]],
            *self.ax_residual.get_ylim(),
            linestyles='dotted',
            label='Fit limits')
        self.ax_residual.legend()

    def plot_injection(self):
        if self.pproc.injection_in is not None:
            self.ax_residual.plot(
                self.freq_in, self.pproc.injection_in,
                label='Injection', c='orange')
            self.ax_residual.legend()

    def plot_t21_in(self):
        if self.pproc.input_t21 is not None:
            self.ax_residual.plot(
                self.freq_in, self.pproc.input_t21,
                c='k', ls='--', label='Input trough')

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

    def add_infotext(self):
        """Adds textbox with fit parameters to the residual plot."""

        # bbox = dict(boxstyle='round', facecolor='grey', alpha=0.25)
        self.ax_residual.plot(
            [], [], ' ',
            label=self.result_str,)
        # self.ax_residual.legend(frameon=True, facecolor='grey')
        # self.ax_residual.text(
        #     0.8, 0.1,
        #     self.result_str,
        #     transform=self.ax_residual.transAxes,
        #     fontsize=10,
        #     verticalalignment='top',
        #     bbox=bbox)

    def save_plots(self, outdir):

        # T_sky
        plt.figure(self.fig_tsky.number)
        plt.savefig(os.path.join(
            outdir, 't_sky.pdf'), dpi=300)

        # dT
        plt.figure(self.fig_residual.number)
        plt.savefig(os.path.join(
            outdir, 'residual.pdf'), dpi=300)
