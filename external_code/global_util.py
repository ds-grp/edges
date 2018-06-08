import numpy as np
from scipy import constants as cst

def define_cosmology():

    # All at z=0
    # Table 1 of http://arxiv.org/pdf/1502.01589v2.pdf
    Cosmo = {'Omega_b':0.049,'Omega_m':0.313,'Omega_K':0.0,'Omega_L':0.687,'h':0.6748,'Omega_R':0.,\
             'coverH0':2999.7,'sig8':0.829,'T0':2.72548}
    # coverH0 in Mpc/h

    '''
    # Load default camb parameters
    # For now, the above parameters are chose consistent with the file parameters by hand...
    camb4py.load(defaults='params_default_simap.ini')
    camb = camb4py.load() # use built-in CAMB

    # Read matching Pk (do it better with call to camb later) at z=0
    #k0,Pk0 = np.loadtxt('test_matterpower.dat',unpack=True)

    # Compute matching Pk at z=0.
    result = camb(get_scalar_cls=True,get_transfer=True)
    k0     = result['transfer_matterpower'][:,0]
    Pk0    = result['transfer_matterpower'][:,1]
    Cosmo['Pk']=(k0,Pk0)

    # Store matching total transfer function at z=0, and normalized to equal 1 at low k at z=0.
    kt     = result['transfer'][:,0]
    Tkt    = result['transfer'][:,6]
    Tkt   /= Tkt[0]
    Cosmo['Tk_tot']=(kt,Tkt)
    '''
    return Cosmo

def define_global():
    # Define global signal parameter models
    #Global = {'T_ref':1.e3,'x_alpha':1.e1,'z_ref':25.,'z_rei':8.,'delta_z_rei':2.,'z_t':13.,\
    #          'delta_z_t':2.,'z_alpha':25.,'delta_z_alpha':10.,'nu21':1420.}
    #Fiducial paramters 5/11/16
    #Global = {'T_ref':-5.,'z_A':80.,'dz_A':20.,'Amp_A':-40.,\
    #          'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
    #          'z_C':20.,'dz_C':4.,'Amp_C':-110.,\
    #           'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    # array = np.array([-2.,-4.,0.])
    # Global_copy = copy.deepcopy(Global)
    # Global['T_ref'] = array[index]
    Global = {'T_ref':-5.,'z_A':80.,'dz_A':20.,'Amp_A':-40.,\
              'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
              'z_C':20.,'dz_C':4.,'Amp_C':-180.,\
              'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    return Global

def define_global3():
    # Define global signal parameter models
    #Global = {'T_ref':1.e3,'x_alpha':1.e1,'z_ref':25.,'z_rei':8.,'delta_z_rei':2.,'z_t':13.,\
    #          'delta_z_t':2.,'z_alpha':25.,'delta_z_alpha':10.,'nu21':1420.}
    #Fiducial paramters 5/11/16
    #Global = {'T_ref':-5.,'z_A':80.,'dz_A':20.,'Amp_A':-40.,\
    #          'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
    #          'z_C':20.,'dz_C':4.,'Amp_C':-110.,\
    #           'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    # array = np.array([-2.,-4.,0.])
    # Global_copy = copy.deepcopy(Global)
    # Global['T_ref'] = array[index]
    Global3 = {'z_A':80.,'dz_A':20.,'Amp_A':-40.,\
              'z_C':20.,'dz_C':4.,'Amp_C':-180.,\
              'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    return Global3

def define_global3_trough():
    # Define global signal parameter models
    #Global = {'T_ref':1.e3,'x_alpha':1.e1,'z_ref':25.,'z_rei':8.,'delta_z_rei':2.,'z_t':13.,\
    #          'delta_z_t':2.,'z_alpha':25.,'delta_z_alpha':10.,'nu21':1420.}
    #Fiducial paramters 5/11/16
    #Global = {'T_ref':-5.,'z_A':80.,'dz_A':20.,'Amp_A':-40.,\
    #          'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
    #          'z_C':20.,'dz_C':4.,'Amp_C':-110.,\
    #           'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    # array = np.array([-2.,-4.,0.])
    # Global_copy = copy.deepcopy(Global)
    # Global['T_ref'] = array[index]
    Global3_trough = {'z_C':20.,'dz_C':4.,'Amp_C':-180.,\
              'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    return Global3_trough


def define_global_trough():
    # Define global signal parameter models
    #Global = {'T_ref':1.e3,'x_alpha':1.e1,'z_ref':25.,'z_rei':8.,'delta_z_rei':2.,'z_t':13.,\
    #          'delta_z_t':2.,'z_alpha':25.,'delta_z_alpha':10.,'nu21':1420.}
    #Fiducial paramters 5/11/16
    #Global = {'T_ref':-5.,'z_A':80.,'dz_A':20.,'AmpA':-40.,\
    #          'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
    #          'z_C':20.,'dz_C':4.,'Amp_C':-110.,\
    #           'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    # array = np.array([-2.,-4.,0.])
    # Global_copy = copy.deepcopy(Global)
    # Global['T_ref'] = array[index]
    Global_trough = {'T_ref':-5.,\
               #     'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
               #      'z_C':20.,'dz_C':4.,'Amp_C':-180.} #absorption trough only
                     'z_C':20.,'dz_C':4.,'Amp_C':-180.} #absorption trough only
    return Global_trough

def define_global3_dip():
    # Define global signal parameter models
    #Global = {'T_ref':1.e3,'x_alpha':1.e1,'z_ref':25.,'z_rei':8.,'delta_z_rei':2.,'z_t':13.,\
    #          'delta_z_t':2.,'z_alpha':25.,'delta_z_alpha':10.,'nu21':1420.}
    #Fiducial paramters 5/11/16
    #Global = {'T_ref':-5.,'z_A':80.,'dz_A':20.,'AmpA':-40.,\
    #          'z_B':30.,'dz_B':5.,'Amp_B': 1.,\
    #          'z_C':20.,'dz_C':4.,'Amp_C':-110.,\
    #           'z_D':10.,'dz_D':2.5,'Amp_D':22.} # Inspired by Fig 3 of HPBB
    # array = np.array([-2.,-4.,0.])
    # Global_copy = copy.deepcopy(Global)
    # Global['T_ref'] = array[index]
    Global3_dip = {'z_C':20.,'dz_C':4.,'Amp_C':-180.} #absorption trough only
    return Global3_dip


def define_foreground():
    # Define fg model parameters
    #    Fg = {'nu0':80,'T0':2250.6,'a1':0.8-2.521,'a2':-0.227,'a3':0.0935}
    Fg = {'nu0':80,'T0':2250.6,'a1':-2.521,'a2':-0.227,'a3':0.0935}
    # nu0 in MHz, T0 in K, as are unitless
    return Fg

def define_instrument():
    # Define instrument parameters
    fg = {'nu0':80,'T0':2250.6,'a1':0.8-2.521,'a2':-0.227,'a3':0.0935}
    # nu0 in MHz, T0 in K, as are unitless
    return fg

def T_cmb(z,Cp):
    # CMB temperature in K
    T_cmb = Cp['T0']*(1+z)
    return T_cmb

def T_gas(z,Gl,Cp):
    # Gas temperature in K
    T_gas = T_cmb(1+Gl['z_ref'])*((1+z)/(1+Gl['z_ref']))**2
    return T_gas

def x_HI(z,Gl,Cp):
    # Neutral hydrogen fraction
    x_HI = np.tanh((z-Gl['z_rei'])/Gl['delta_z_rei'])+1.
    x_HI/= 2.
    return x_HI

def T_k(z,Gl,Cp):
    # Kinetic temperature parametrization in K
    T_k = Gl['T_ref']*np.tanh((z-Gl['z_t'])/Gl['delta_z_t'])+1.
    T_k/= 2.
    return T_k

def x_alpha(z,Gl):
    x_alpha = Gl['x_alpha']*np.tanh((z-Gl['z_alpha'])/Gl['delta_z_alpha'])+1.
    x_alpha/= 2.
    return x_alpha

def T_alpha(z,Gl):
    T_alpha = Gl['x_alpha']*np.tanh((z-Gl['z_alpha'])/Gl['delta_z_alpha'])+1.
    T_alpha/= 2.
    return T_alpha

def x_c(z,Gl):
    x_c = Gl['x_alpha']*np.tanh((z-Gl['z_alpha'])/Gl['delta_z_alpha'])+1.
    x_c/= 2.
    return x_c

def T_spin(z,Gl,Cp):
    T_spin = 1.#1./T_cmb(z,Cp)+x_alpha(z,Gl)/T_alpha
    return T_spin

'''
def dT_global(z,Gl,Cp):
    # Global temperature in mK.
    T_spin = T_spin(z,Gl,Cp)
    T_cmb  = T_cmb(z,Cp)
    omb    = Cp['Omega_b']*Cp['h']**2
    omm    = Cp['Omega_m']*Cp['h']**2
    T_global = 27.*x_HI(z,Gl,Cp)*(omb/0.023)*((0.15/omm)*(1+z)/10.)**0.5*(T_spin-T_cmb)/T_cmb
    return T_global
'''

def dT_global(nu,Cp,Gl):
    # Global temperature in mK.
    # Toy model inspired by Fig. 3 of HPBB
    nu21 = 1420. # MHz
    z    = nu21/nu-1.
    dT_global = Gl['T_ref'] \
              + Gl['Amp_A']*np.exp(-0.5*((z-Gl['z_A'])/Gl['dz_A'])**2.)\
              + Gl['Amp_B']*np.exp(-0.5*((z-Gl['z_B'])/Gl['dz_B'])**2.)\
              + Gl['Amp_C']*np.exp(-0.5*((z-Gl['z_C'])/Gl['dz_C'])**2.)\
              + Gl['Amp_D']*np.exp(-0.5*((z-Gl['z_D'])/Gl['dz_D'])**2.)
    return dT_global

def dT_global3(nu,Cp,Gl3):
    # Global temperature in mK.
    # Toy model inspired by Fig. 3 of HPBB
    nu21 = 1420. # MHz
    z    = nu21/nu-1.
    dT_global3 = Gl3['Amp_A']*np.exp(-0.5*((z-Gl3['z_A'])/Gl3['dz_A'])**2.)\
               + Gl3['Amp_C']*np.exp(-0.5*((z-Gl3['z_C'])/Gl3['dz_C'])**2.)\
               + Gl3['Amp_D']*np.exp(-0.5*((z-Gl3['z_D'])/Gl3['dz_D'])**2.)
    return dT_global3

def dT_global3_trough(nu,Cp,Gl3_trough):
    # Global temperature in mK.
    # Toy model inspired by Fig. 3 of HPBB
    nu21 = 1420. # MHz
    z    = nu21/nu-1.
    dT_global3_trough = Gl3_trough['Amp_C']*np.exp(-0.5*((z-Gl3_trough['z_C'])/Gl3_trough['dz_C'])**2.)\
               + Gl3_trough['Amp_D']*np.exp(-0.5*((z-Gl3_trough['z_D'])/Gl3_trough['dz_D'])**2.)
    return dT_global3_trough

def dT_global_trough(nu,Cp,Glt):
    # Global temperature in mK.
    # Toy model inspired by Fig. 3 of HPBB
    nu21 = 1420. # MHz
    z    = nu21/nu-1.
    dT_global_trough = Glt['T_ref'] \
        + Glt['Amp_C']*np.exp(-0.5*((z-Glt['z_C'])/Glt['dz_C'])**2.)
             # + Glt['Amp_D']*np.exp(-0.5*((z-Gl['z_D'])/Gl['dz_D'])**2.)
    return dT_global_trough

def dT_global3_dip(nu,Cp,Gl3_dip):
    # Global temperature in mK.
    # Toy model inspired by Fig. 3 of HPBB
    nu21 = 1420. # MHz
    z    = nu21/nu-1.
    dT_global3_dip = Gl3_dip['Amp_C']*np.exp(-0.5*((z-Gl3_dip['z_C'])/Gl3_dip['dz_C'])**2.)
    return dT_global3_dip

'''
def dT_global(z,Gl,Cp):
    # Global temperature in mK.
    # Toy model
    T_s = T_spin(z,Gl,Cp)
    T_r = T_cmb(z,Cp)
    omb = Cp['Omega_b']*Cp['h']**2
    omm = Cp['Omega_m']*Cp['h']**2
    #print Gl['z_alpha'],Gl['delta_z_alpha']
    #T_global = 27.*x_HI(z,Gl,Cp)*(omb/0.023)*((0.15/omm)*(1+z)/10.)**0.5*(T_spin-T_cmb)/T_cmb
    dT_global = -0.20*np.exp(-0.5*((z-Gl['z_alpha'])/Gl['delta_z_alpha'])**2.)
    return dT_global
'''


#Why Foreground this way?  Feb 2017
def dT_fg(nu,Fg):
    # Fg temperature variation in K
    # nu in MHz
    nu0  = Fg['nu0']
    x    = np.log(nu/nu0)
    logT = np.log(Fg['T0'])+Fg['a1']*x+Fg['a2']*x**2+Fg['a3']*x**3
    return np.exp(logT)*1.e3

'''
def dT_fg(nu,Fg):
    # Fg temperature variation in K
    # nu in MHz
    nu0  = Fg['nu0']
    x    = (nu/nu0)
    T = Fg['T0']+Fg['a1']*x+Fg['a2']*x**2+Fg['a3']*x**3
    return T*1.e3
    # return in mK
'''

def dT_noise(nu,Fg):
    # sigT in mK
    sigT = dT_fg(nu,Fg)
    # Cooked up noise levels based on Fig.2 of HPBB
    #return sigT*3.e-7
    return sigT*3.e-7

def dT_model(nu,Cp,Gl,Fg):
    # Data model in mK
    # nu in MHz
    signal     = dT_global(nu,Cp,Gl)
    foreground = dT_fg(nu,Fg)
    #print 'dT ',Fg['a1']
    return signal+foreground

def dT_model3(nu,Cp,Gl3,Fg):
    # Data model in mK
    # nu in MHz
    signal     = dT_global3(nu,Cp,Gl3)
    foreground = dT_fg(nu,Fg)*0.001
    #print 'dT ',Fg['a1']
    return signal+foreground

def dT_model3_trough(nu,Cp,Gl3_trough,Fg):
    # Data model in mK
    # nu in MHz
    signal     = dT_global3_trough(nu,Cp,Gl3_trough)
    foreground = dT_fg(nu,Fg)
    #print 'dT ',Fg['a1']
    return signal+foreground

def dT_model_trough(nu,Cp,Glt,Fg):
    # Data model in mK
    # nu in MHz
    signal     = dT_global_trough(nu,Cp,Glt)
    foreground = dT_fg(nu,Fg)
    #print 'dT ',Fg['a1']
    return signal+foreground

def dT_model3_dip(nu,Cp,Gl3_dip,Fg):
    # Data model in mK
    # nu in MHz
    signal     = dT_global3_dip(nu,Cp,Gl3_dip)
    foreground = dT_fg(nu,Fg)
    #print 'dT ',Fg['a1']
    return signal+foreground
