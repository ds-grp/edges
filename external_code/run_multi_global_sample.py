 #!/usr/bin/env python
"""
Sample code for sampling a multivariate Gaussian using emcee.
Allow multiple global model inputs
"""
#from __future__ import print_function
import numpy as np
import numpy.ma as ma
import pylab as pl
import matplotlib.pyplot as plt
import emcee
import global_util as gl
import corner
import matplotlib as mpl
import os
#import mpi4py.rc
#mpi4py.rc.initialize = False

# ==============
# File IO
# ==============
NB = False
if NB:
    fdir = '/Users/tcchang/Projects/global_entry/Plots/March18/NB/'
else:
    fdir = '/Users/tcchang/Projects/global_entry/Plots/March18/BB/'
    
#if not os.path.isdir(pwrout_base):
#    os.mkdir(pwrout_base)
    
# =========================================
# Plot theory model to test implementation.
# =========================================
Cp = gl.define_cosmology()
Gl = gl.define_global()
Gl3 = gl.define_global3()
Glt = gl.define_global_trough()
Gl3_dip = gl.define_global3_dip()
Fg = gl.define_foreground()


# ==== Set the global signal range 
# Vary 3 parameters, ampC, zC, dzC
# ================================
n_ampC   = 5
ampCl=-10.
ampCh=-500.
ampC_array = np.linspace(ampCl,ampCh,num=n_ampC)

# location of dip
n_zC = 5
zCl=10.
zCh=30. 
zC_array = np.linspace(zCl,zCh,num=n_zC)

# width of dip
n_dzC = 3
dzCl = 1.
dzCh = 10.
dzC_array = np.linspace(dzCl,dzCh,num=n_dzC)

# redshift/freq bins
nz    = 200
nul   = 1.
nuh   = 200.
nu_arr = np.linspace(nul,nuh,num=nz)

# parameters
nu0=1420.405751786 #MHz

# useful arrays
z_arr=nu0/nu_arr-1.
dT_mod = np.zeros([n_ampC,n_zC, n_dzC, nz])
dT_fg  = np.zeros(nz) # same foreground model
sigT   = np.zeros(nz) # same thermal noise model
#
# Define Narrow Bands
nsubnu=6
# Lowest freq band set by noise/fg level -- no need to go into Tsys noise given an intergration time
# Highest freq band set by 21cm signal level -- no need to go below z~6 or so.

# Do 10% fractional BW  (25,75,125,18,54, 90.) [18+25]
subnul=[24.,71.,119.,18, 52.,86.] # freq interval
subnuh=[26.,79.,131.,19.,56.,94.]

# define mask
nu_mask=np.arange(subnul[0],subnuh[0])
for f in np.arange(nsubnu-1):
    band1=np.arange(subnul[f+1],subnuh[f+1])
    nu_mask=np.append(nu_mask,band1)
#b1=np.arange(band1.size)

'''
# === For plotting full model ===
# Set up the global models, somewhat ad hoc 
for iamp in np.arange(n_ampC):
    for izc in np.arange(n_zC):
        for idzc in np.arange(n_dzC):
            # redefine parameters
            Gl3 = gl.define_global3()
            Fg = gl.define_foreground()
            Gl3['Amp_C'] = ampC_array[iamp]
            Gl3['z_C'] = zC_array[izc]
            Gl3['dz_C'] = dzC_array[idzc]
            #
            # modify models
            offset = gl.dT_global3(240., Cp, Gl3)
            #print 'offset',offset
            Amp_D_default = Gl3['Amp_D']
            z_D_default=Gl3['z_D']
            if (izc == 1 or izc == 2 or izc == 3):
                Gl3['z_D'] = (z_D_default+offset*0.03)
                Gl3['Amp_D']=Gl3['Amp_D']-offset*1.3
            if (izc == 0):
                Gl3['z_D'] = (z_D_default+offset*0.02)
                Gl3['Amp_D']=Gl3['Amp_D']-offset*1.15
            # compute model
            for iz in np.arange(nz):
                nu = nu_arr[iz]
                dT_mod[iamp,izc, idzc, iz] = gl.dT_global3(nu,Cp,Gl3)
            Gl3['Amp_D']=Amp_D_default
            Gl3['z_D'] = z_D_default
for iz in np.arange(nz):
    nu = nu_arr[iz]
    dT_fg[iz]  = gl.dT_fg(nu,Fg) 
    sigT[iz]   = gl.dT_noise(nu,Fg)


# Make signal plot
# --- Save nominal signal model
# -----------------------------
# Reproduce Fig. 3 of HPBB
# Add z axis on top
# --- Save nominal signal and fg model
# ------------------------------------
# Reproduce Fig. 2 of HPBB
#label_size = 15
#plt.rcParams['xtick.labelsize'] = label_size 

fig = plt.figure()
ax = fig.add_subplot(111)
for iamp in np.arange(n_ampC):
    for izc in np.arange(n_zC):
        for idzc in np.arange(n_dzC):
            ax.plot(nu_arr,dT_mod[iamp,izc,idzc,:])#,color='red',label=r'Nominal model 1')
            #ax.plot(nu_arr,dT_mod[1,:],color='red',linestyle='--',label=r'Nominal model 2')
ax.plot(nu_arr,sigT,color='green',label=r'Noise level (rms)')
## plot narrow bands
for f in np.arange(nsubnu):
    ax.axvspan(subnul[f], subnuh[f], color='y', alpha=0.5, lw=0)
ax2 = ax.twiny()
new_tick_locations = np.array([20.,40.,60.,80.,100.,120.,140.,160.,180.,210.,240.])
def tick_function(X):
    V = nu0/(X)-1.
    return ["%.0f" % z for z in V]

zticks=tick_function(new_tick_locations)
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(zticks)#,weight='bold')
#ax2.set_xticklabels(['{:g}'.format(z) for z in zticks]);
#ax2.set_xticklabels(['{:g}'.format(z) for z in z_arr])
zmin, zmax = 10,250
ax.set_xlim(zmin, zmax)
ax2.set_xlim(zmin, zmax)
ax.set_ylim(-250, 40)
ax.set_xlabel(r'Frequency [MHz]',weight='bold')
ax2.set_xlabel(r'Redshift',weight='bold')
ax.set_ylabel(r'$\mathbf{\delta_{T_b}}$ [mK]',weight='bold')
ax.minorticks_on()

ax.legend(loc='lower right')
ax.text(17,-52,r'Dark',color='black',weight='bold')
ax.text(17,-60,r'Ages',color='black',weight='bold')
ax.text(40,-82,r'First',color='black',weight='bold')
ax.text(40,-90,r'Stars',color='black',weight='bold')
ax.text(90,-74,r'First',color='black',weight='bold')
ax.text(85,-82,r'Accreting',color='black',weight='bold')
ax.text(83,-90,r'Black Holes',color='black',weight='bold')
ax.text(110,20,r'Epoch of Reionization',color='black',weight='bold')

fig.savefig(fdir+'/model_multi.pdf', dpi=200, bbox_inches='tight')
plt.close(fig)

#assert(0)
'''

def lnprior(theta):
    T_ref,z_A,dz_A,Amp_A,z_B,dz_B,Amp_B,z_C,dz_C,Amp_C,z_D,dz_D,Amp_D,T0,a1,a2,a3 = theta
    if  z_A < 0. or z_A > 100.:
        return -np.inf
    if  z_B < 0. or z_B > 100.:
        return -np.inf
    if  z_C < 0. or z_C > 100.:
        return -np.inf
    if  z_D < 0. or z_D > 100.:
        return -np.inf
    if  dz_A < 0.:
        return -np.inf
    if  dz_C < 0.:
        return -np.inf
    if  dz_D < 0.:
        return -np.inf
    '''
    # Define simple uniform prior
    mu,sig = theta
    if 0. < mu < 100. and 0.00001 < sig < 100.0 :
        return 0.0
    '''
    return 0.

def lnprior3(theta3):
    z_A,dz_A,Amp_A,z_C,dz_C,Amp_C,z_D,dz_D,Amp_D,T0,a1,a2,a3 = theta3
    if  z_A < 0. or z_A > 100.:
        return -np.inf
    if  z_C < 0. or z_C > 100.:
        return -np.inf
    if  z_D < 0. or z_D > 100.:
        return -np.inf
    if  dz_A < 0.:
        return -np.inf
    if  dz_C < 0.:
        return -np.inf
    if  dz_D < 0.:
        return -np.inf

    return 0.

def lnprior3_trough(theta3_trough):
    z_C,dz_C,Amp_C,z_D,dz_D,Amp_D,T0,a1,a2,a3 = theta3_trough

    if  z_C < 0. or z_C > 100.:
        return -np.inf
    if  z_D < 0. or z_D > 100.:
        return -np.inf
    if  dz_C < 0.:
        return -np.inf
    if  dz_D < 0.:
        return -np.inf

    return 0.

def lnprior3_dip(theta3_dip):
    z_C,dz_C,Amp_C,T0,a1,a2,a3 = theta3_dip

    if  z_C < 0. or z_C > 100.:
        return -np.inf
    if  dz_C < 0.:
        return -np.inf

    return 0.

def lnprior_trough(theta_trough):
    T_ref,z_C,dz_C,Amp_C,T0,a1,a2,a3 = theta_trough
    if  z_C < 0. or z_C > 100.:
        return -np.inf
    if  dz_C < 0.:
        return -np.inf

    return 0.

def lnprob(theta,x,y,yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x,y,yerr)

def lnprob3(theta3,x,y,yerr):
    lp = lnprior3(theta3)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike3(theta3,x,y,yerr)

def lnprob3_trough(theta3_trough,x,y,yerr):
    lp = lnprior3_trough(theta3_trough)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike3_trough(theta3_trough,x,y,yerr)

def lnprob_trough(theta_trough,x,y,yerr):
    lp = lnprior_trough(theta_trough)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_trough(theta_trough,x,y,yerr)

def lnprob3_dip(theta3_dip,x,y,yerr):
    lp = lnprior3_dip(theta3_dip)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike3_dip(theta3_dip,x,y,yerr)


def theta2params(theta):
    #
    global Cp,Gl,Fg
    T_ref,z_A,dz_A,Amp_A,z_B,dz_B,Amp_B,z_C,dz_C,Amp_C,z_D,dz_D,Amp_D,T0,a1,a2,a3 = theta

    # Define defaults
    Cp = gl.define_cosmology()
    Gl = gl.define_global()
    Fg = gl.define_foreground()
    # Update defaults accordingly
    Gl['T_ref'] = T_ref
    Gl['z_A']   = z_A
    Gl['dz_A']  = dz_A
    Gl['Amp_A'] = Amp_A
    Gl['z_B']   = z_B
    Gl['dz_B']  = dz_B
    Gl['Amp_B'] = Amp_B
    Gl['z_C']   = z_C
    Gl['dz_C']  = dz_C
    Gl['Amp_C'] = Amp_C
    Gl['z_D']   = z_D
    Gl['dz_D']  = dz_D
    Gl['Amp_D'] = Amp_D
    Fg['T0'] = T0
    Fg['a1'] = a1
    Fg['a2'] = a2
    Fg['a3'] = a3

    return

def theta2params3(theta3):
    #
    global Cp,Gl3,Fg
    z_A,dz_A,Amp_A,z_C,dz_C,Amp_C,z_D,dz_D,Amp_D,T0,a1,a2,a3 = theta3

    # Define defaults
    Cp = gl.define_cosmology()
    Gl3 = gl.define_global3()
    Fg = gl.define_foreground()
    # Update defaults accordingly
    Gl3['z_A']   = z_A
    Gl3['dz_A']  = dz_A
    Gl3['Amp_A'] = Amp_A
    Gl3['z_C']   = z_C
    Gl3['dz_C']  = dz_C
    Gl3['Amp_C'] = Amp_C
    Gl3['z_D']   = z_D
    Gl3['dz_D']  = dz_D
    Gl3['Amp_D'] = Amp_D

    Fg['T0'] = T0
    Fg['a1'] = a1
    Fg['a2'] = a2
    Fg['a3'] = a3

    return

def theta2params3_trough(theta3_trough):
    #
    global Cp,Gl3_trough,Fg
    z_C,dz_C,Amp_C,z_D,dz_D,Amp_D,T0,a1,a2,a3 = theta3_trough

    # Define defaults
    Cp = gl.define_cosmology()
    Gl3_trough = gl.define_global3_trough()
    Fg = gl.define_foreground()
    # Update defaults accordingly
    Gl3_trough['z_C']   = z_C
    Gl3_trough['dz_C']  = dz_C
    Gl3_trough['Amp_C'] = Amp_C
    Gl3_trough['z_D']   = z_D
    Gl3_trough['dz_D']  = dz_D
    Gl3_trough['Amp_D'] = Amp_D

    Fg['T0'] = T0
    Fg['a1'] = a1
    Fg['a2'] = a2
    Fg['a3'] = a3

    return

def theta2params3_dip(theta3_dip):
    #
    global Cp,Gl3_dip,Fg
    z_C,dz_C,Amp_C,T0,a1,a2,a3 = theta3_dip

    # Define defaults
    Cp = gl.define_cosmology()
    Gl3_dip = gl.define_global3_dip()
    Fg = gl.define_foreground()
    # Update defaults accordingly
    Gl3_dip['z_C']   = z_C
    Gl3_dip['dz_C']  = dz_C
    Gl3_dip['Amp_C'] = Amp_C

    Fg['T0'] = T0
    Fg['a1'] = a1
    Fg['a2'] = a2
    Fg['a3'] = a3

    return

def theta2params_trough(theta_trough):
    #
    global Cp,Glt,Fg
    T_ref,z_C,dz_C,Amp_C,T0,a1,a2,a3 = theta_trough

    # Define defaults
    Cp = gl.define_cosmology()
    Glt = gl.define_global_trough()
    Fg = gl.define_foreground()
    # Update defaults accordingly
    Glt['T_ref'] = T_ref
    Glt['z_C']   = z_C
    Glt['dz_C']  = dz_C
    Glt['Amp_C'] = Amp_C

    Fg['T0'] = T0
    Fg['a1'] = a1
    Fg['a2'] = a2
    Fg['a3'] = a3
    #
    return

def lnlike(theta,x,y,yerr):
    theta2params(theta)
    # Compute model and chi2 on the fly
    nx   = x.size
    chi2 = 0.
    for ix in np.arange(x.size):
        mod   = gl.dT_model(x[ix],Cp,Gl,Fg)
        sig   = gl.dT_noise(x[ix],Fg)
        chi2 += ((y[ix]-mod)/sig)**2
    return -chi2/2.

def lnlike3(theta3,x,y,yerr):
    theta2params3(theta3)
    # Compute model and chi2 on the fly
    nx   = x.size
    chi2 = 0.
    for ix in np.arange(x.size):
        mod   = gl.dT_model3(x[ix],Cp,Gl3,Fg)
        sig   = gl.dT_noise(x[ix],Fg)
        chi2 += ((y[ix]-mod)/sig)**2
    return -chi2/2.

def lnlike3_trough(theta3_trough,x,y,yerr):
    theta2params3_trough(theta3_trough)
    # Compute model and chi2 on the fly
    nx   = x.size
    chi2 = 0.
    for ix in np.arange(x.size):
        mod   = gl.dT_model3_trough(x[ix],Cp,Gl3_trough,Fg)
        sig   = gl.dT_noise(x[ix],Fg)
        chi2 += ((y[ix]-mod)/sig)**2
    return -chi2/2.

def lnlike3_dip(theta3_dip,x,y,yerr):
    theta2params3_dip(theta3_dip)
    # Compute model and chi2 on the fly
    nx   = x.size
    chi2 = 0.
    for ix in np.arange(x.size):
        mod   = gl.dT_model3_dip(x[ix],Cp,Gl3_dip,Fg)
        sig   = gl.dT_noise(x[ix],Fg)
        chi2 += ((y[ix]-mod)/sig)**2
    return -chi2/2.


def lnlike_trough(theta,x,y,yerr):
    theta2params_trough(theta)
    # Compute model and chi2 on the fly
    nx   = x.size
    chi2 = 0.
    for ix in np.arange(x.size):
        mod   = gl.dT_model_trough(x[ix],Cp,Glt,Fg)
        sig   = gl.dT_noise(x[ix],Fg)
        chi2 += ((y[ix]-mod)/sig)**2
    return -chi2/2.


# Run a bunch of models =========

# === Define sampler parameters
#ndim     = 4+6 # Number of parameters to solve (4 for fg, 6 for signal). 2-gaussian model. (global3_trough) Noise perfectly known.
ndim     = 4+9 # Number of parameters to solve (4 for fg, 9 for signal). 3-gaussian model. Noise perfectly known.
nwalkers = 100 # Number of walkers

# === Define fake data
Cp = gl.define_cosmology()
Glt = gl.define_global_trough()
Gl3 = gl.define_global3()
Gl3_trough = gl.define_global3_trough()
Gl3_dip = gl.define_global3_dip()
Gl = gl.define_global()
Fg = gl.define_foreground()


# === Define run parameters

### Change NarroBand or BraodBand here
if NB:
    x=nu_mask #NB
else:
    x = nu_arr #BB
nx = x.size


# ================
# run each model  
##
#for iamp in np.arange(n_ampC):
for iamp in [1]:
    #Gl3['Amp_C'] = ampC_array[iamp]
    #for izc in np.arange(n_zC):
    for izc in[1]:
        #Gl3['z_C'] = zC_array[izc]
        #for idzc in np.arange(n_dzC):
        for idzc in [1]:
        #redefine paramters
            Gl3 = gl.define_global3()
            Gl3_trough = gl.define_global3_trough()
            Fg = gl.define_foreground()
            Gl3['Amp_C'] = ampC_array[iamp]
            Gl3['z_C'] = zC_array[izc]
            Gl3['dz_C'] = dzC_array[idzc]
            # redshift bins
            #
            # modify models
            offset = gl.dT_global3(240., Cp, Gl3)
            Amp_D_default = Gl3['Amp_D']
            z_D_default=Gl3['z_D']
            '''
            if (izc == 1 or izc == 2 or izc == 3):
                Gl3['z_D'] = (z_D_default+offset*0.03)
                Gl3['Amp_D']=Gl3['Amp_D']-offset*2.5
            if (izc == 0):
                Gl3['z_D'] = (z_D_default+offset*0.02)
                Gl3['Amp_D']=Gl3['Amp_D']-offset*1.15
            '''
            #
            # set up the data arrays
            y    = np.zeros(nx,dtype='float64') # Measurement
            yerr = np.zeros(nx,dtype='float64')
            sig_exp = np.zeros(nx,dtype='float64')
            noise_exp = np.zeros(nx,dtype='float64')
            noise_ran = np.zeros(nx,dtype='float64')
            fg_exp = np.zeros(nx,dtype='float64')
            # for full spectrum input
            sigT = np.zeros(nz,dtype='float64')
            dT_mod = np.zeros(nz,dtype='float64')
            #
            # set up directory
            odir = 'AmpC_'+str(ampC_array[iamp])+'_zC_'+str(zC_array[izc])+'_dzC_'+str(dzC_array[idzc])
            outputdir = fdir+odir
            if not os.path.isdir(outputdir):
                os.mkdir(outputdir)
            #
            for iz in np.arange(nx):
                nu = x[iz]
                noise_ran[iz] =  gl.dT_noise(nu,Fg)*np.random.randn(1)
                y[iz] = gl.dT_model3(nu,Cp,Gl3,Fg) + noise_ran[iz]
                yerr[iz] = gl.dT_noise(nu,Fg)
                sig_exp[iz] = gl.dT_global3(nu,Cp,Gl3)
                noise_exp[iz] = gl.dT_noise(nu,Fg)
                fg_exp[iz] = gl.dT_fg(nu,Fg)
            for iz in np.arange(nz):
                nu = nu_arr[iz]
                sigT[iz]   = gl.dT_noise(nu,Fg)
                dT_mod[iz] = gl.dT_global3(nu,Cp,Gl3)
            Gl3['Amp_D']=Amp_D_default
            Gl3['z_D'] = z_D_default

            # preparing for MCMC
            # Initial input model parameters
            # === First choice for each walker (10% scatter from the truth)
            # === Supoosedly one truth for all models for now
            #theta0_3T = np.array([20.,4.,-180.,10.,2.5,22.,2250.6,-2.521,-0.227,0.0935]) # global3_trough
            #p0 = [theta0_3T+np.random.rand(ndim)*theta0_3T*0.1 for i in range(nwalkers)]
            theta0_3 = np.array([80.,20.,-40.,20.,4.,-180.,10.,2.5,22.,2250.6,-2.521,-0.227,0.0935]) # For global3
            p0 = [theta0_3+np.random.rand(ndim)*theta0_3*0.1 for i in range(nwalkers)]
            
            #print 'Like :',lnlike3_trough(theta0_3T,x,y,yerr)
            print 'Like :',lnlike3(theta0_3,x,y,yerr)
            
            # === Initialize the sampler with the chosen specs
            #sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob3_trough,args=(x,y,yerr))#,threads=1)
            sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob3,args=(x,y,yerr))#,threads=1)
            
            # === Run steps as a burn-in
            nsteps  = 2000 #2000 #1000 #500
            nburnin = 200 #500 #200 #100
            sampler.run_mcmc(p0,nsteps)

            # === Flatten the chain
            samples = sampler.chain[:,nburnin:,:].reshape((-1, ndim))
            
            # === Make corner plot
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # === Plot triangle
            #labels = ['$z_C$','$dz_C$','$A_C$',\
            #'$z_D$','$dz_D$','$A_D$','$T_0$','$a_1$','$a_2$','$a_3$']
            labels = ['$z_A$','$dz_A$','A_A','$z_C$','$dz_C$','$A_C$',\
                '$z_D$','$dz_D$','$A_D$','$T_0$','$a_1$','$a_2$','$a_3$']
            label_size = 15

            #fig = corner.corner(samples, labels=labels,fontsize=20,truths=theta0_3T,show_titles=True)#, title_kwargs={"fontsize": 15})
            fig = corner.corner(samples, labels=labels,fontsize=20,truths=theta0_3,show_titles=True)#, title_kwargs={"fontsize": 15})
            fig.savefig(outputdir+"/triangle.pdf")
            plt.close(fig)

            #assert(0)

            # == Save best fit values
            # this doesn't seem necessasy. from emcee documentation but not sure why.
            #samples[:, 2] = np.exp(samples[:, 2])

            zA_mcmc, dzA_mcmc, AA_mcmc, zC_mcmc, dzC_mcmc, AC_mcmc, zD_mcmc, dzD_mcmc, AD_mcmc, T0_mcmc, a1_mcmc, a2_mcmc, a3_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
            #
            print 'MCMC results', zA_mcmc, dzA_mcmc, AA_mcmc, zC_mcmc, dzC_mcmc, AC_mcmc, zD_mcmc, dzD_mcmc, AD_mcmc, T0_mcmc, a1_mcmc, a2_mcmc, a3_mcmc
            Gl_mcmc = {'z_A':zA_mcmc[0],'dz_A':dzA_mcmc[0],'Amp_A':AA_mcmc[0],\
                       'z_C':zC_mcmc[0],'dz_C':dzC_mcmc[0],'Amp_C':AC_mcmc[0],\
                       'z_D':zD_mcmc[0],'dz_D':dzD_mcmc[0],'Amp_D':AD_mcmc[0]} 
            np.savetxt(outputdir+'/Gl_mcmc_err',(zA_mcmc, dzA_mcmc, AA_mcmc, zC_mcmc, dzC_mcmc, AC_mcmc, zD_mcmc, dzD_mcmc, AD_mcmc, T0_mcmc, a1_mcmc, a2_mcmc, a3_mcmc))
            '''
            zC_mcmc, dzC_mcmc, AC_mcmc, zD_mcmc, dzD_mcmc, AD_mcmc, T0_mcmc, a1_mcmc, a2_mcmc, a3_mcmc = \
              map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
              
            print 'MCMC results',zC_mcmc, dzC_mcmc, AC_mcmc, zD_mcmc, dzD_mcmc, AD_mcmc, T0_mcmc, a1_mcmc, \
              a2_mcmc, a3_mcmc
            
            Gl_mcmc = {'z_C':zC_mcmc[0],'dz_C':dzC_mcmc[0],'Amp_C':AC_mcmc[0],\
                       'z_D':zD_mcmc[0],'dz_D':dzD_mcmc[0],'Amp_D':AD_mcmc[0]}
                       
            np.savetxt(outputdir+'/Gl_mcmc_err',(zC_mcmc, dzC_mcmc, AC_mcmc, T0_mcmc, a1_mcmc, a2_mcmc, a3_mcmc))
            #np.savetxt(outputdir+'/MCMC_samples',samples)
            '''
            
            # === Make model plot
            #
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(nu_arr,dT_mod,color='red',label=r'Input Model')
            ax.plot(nu_arr,sigT,color='green',label=r'Noise level (rms)')
            ax.plot(nu_arr,gl.dT_global3(nu_arr,Cp,Gl_mcmc),color='red',linestyle='--',label=r'MCMC Model')
            #ax.plot(nu_arr,gl.dT_global3_trough(nu_arr,Cp,Gl_mcmc),color='red',linestyle='--',label=r'MCMC Model')
            for f in np.arange(nsubnu):
                ax.axvspan(subnul[f], subnuh[f], color='y', alpha=0.5, lw=0)
            ax2 = ax.twiny()
            new_tick_locations = np.array([20.,40.,60.,80.,100.,120.,140.,160.,180.,210.,240.])
            def tick_function(X):
                V = nu0/(X)-1.
                return ["%.0f" % z for z in V]

            zticks=tick_function(new_tick_locations)
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(zticks)#,weight='bold')
            zmin, zmax = 10,300
            ax.set_xlim(zmin, zmax)
            ax2.set_xlim(zmin, zmax)
            ax.set_ylim(-250, 40)
            ax.set_xlabel(r'Frequency [MHz]',weight='bold')
            ax2.set_xlabel(r'Redshift',weight='bold')
            ax.set_ylabel(r'$\mathbf{\delta_{T_b}}$ [mK]',weight='bold')
            ax.minorticks_on()

            ax.legend(loc='lower right')
            ax.text(17,-52,r'Dark',color='black',weight='bold')
            ax.text(17,-60,r'Ages',color='black',weight='bold')
            ax.text(40,-82,r'First',color='black',weight='bold')
            ax.text(40,-90,r'Stars',color='black',weight='bold')
            ax.text(90,-74,r'First',color='black',weight='bold')
            ax.text(85,-82,r'Accreting',color='black',weight='bold')
            ax.text(83,-90,r'Black Holes',color='black',weight='bold')
            ax.text(110,20,r'Epoch of Reionization',color='black',weight='bold')

            fig.savefig(outputdir+'/global.pdf', dpi=200, bbox_inches='tight')
            plt.close(fig)

            
#assert(0)

