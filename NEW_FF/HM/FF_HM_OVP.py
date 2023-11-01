#!/home/krish.shah/anaconda3/envs/Teobresums/bin/python3

import sys
sys.path.append('/home/krish.shah/teobresums/Python')
import pycbc
import EOBRun_module
import numpy as np
import matplotlib.pyplot as plt
import time
from pycbc.waveform import utils
import scipy
from copy import deepcopy
from pycbc.waveform import get_fd_waveform, get_td_waveform

"""
Contains functions relevant for domaining variables.
"""
def wrap_reflective( x, x1, x2):
    """
    Function to wrap and reflect a real number around the points x1 and x2. 
    Example - For spins, we will have 'wrap_reflective(1.1, -1, 1) = 0.9'; 'wrap_reflective(-1.1, -1, 1) = -0.9'; and so on.
    Parameters
    ----------
    x : float
        Value to be reflected.
    x1 : float
        The LHS reflective point.
    x2 : float
        The RHS reflective point.
    Returns
    -------
    float
        Wrapped and reflective value of x around x1 and x2.
    """        
    if x2 == None: #i.e., x2 can be unbounded
        x2 = 1e4   #assign a big number (but not np.inf as it will bias the algebra)
    period = 2*(x2 - x1)
    x_frac = (x - x1) % period
    if x_frac <= period/2:
        x_ref = x_frac + x1  
    elif x_frac > period/2:
        x_ref = (period - x_frac) + x1 
    return x_ref

    # function to wrap a real number periodically around the points x1 and x2. 
    # Ex- for a sine func, we will have 'periodic(2*np.pi + x, 0, 2*np.pi) = x'.
def wrap_periodic( x, x1, x2):
    """
    Function to wrap a real number periodically around the points x1 and x2. 
    Example - For spins, we will have 'wrap_periodic(2*np.pi + x, 0, 2*np.pi) = x'.
    Parameters
    ----------
    x : float
        Value to be reflected.
    x1 : float
        The LHS coordinate of boundary.
    x2 : float
        The RHS coordinate of boundary.
    Returns
    -------
    float
        Periodically wrapped value of x around x1 and x2.
    """    
    period = (x2 - x1)
    x_frac = (x - x1) % period
    x_p = x_frac + x1  
    return x_p

    # wraps x between (x1, x2) assuming boundaries to be either periodic or reflective.
def wrap( x, x1, x2, boundary='reflective'):
    """
    Function to wrap a real number around the points x1 and x2 either periodically or reflectively. 
    Example - (i) For spins, we will have 'wrap(2*np.pi + x, 0, 2*np.pi, 'periodic') = x';
    'wrap(1.1, -1, 1, 'reflective') = 0.9'; 'wrap(-1.1, -1, 1, 'reflective') = -0.9'.
    Parameters
    ----------
    x : float
        Value to be reflected.
    x1 : float
        The LHS coordinate of boundary.
    x2 : float
        The RHS coordinate of boundary.
    boundary : {'reflective', 'periodic'}, optional.
        Boundary type to conisder while wrapping. Default = 'reflective'.
    Returns
    -------
    float
        Periodically wrapped value of x around x1 and x2.
    Raises
    ------
    KeyError
        Allowed keywords for boundary are: {'reflective', 'periodic'}.
    """        
    if (boundary == 'reflective' or boundary == 'Reflective'):
        return wrap_reflective(x, x1, x2)
    elif (boundary == 'periodic' or boundary == 'Periodic'):
        return wrap_periodic(x, x1, x2)
    else:
        raise KeyError("Incorrect keyword provided for the argument 'boundary'. Allowed keywords are: {'reflective', 'periodic'}.")

def ext_sp( a):
    """
    Checks if the dimensionless spin magnitude `a` satisfies 0.998 < s < 1, otherwise assign a = 0.9.
    """
    if (abs(round(a, 3)) > 0.998):
        return 0.9*a/np.abs(a)
    else:
        return a
def dom_indv_sp( x):
    """
    Domain of an individual spin component: wrapping and reflection of real line around (-1, 1).
    
    """
    sp_ref = wrap(x, -1., 1., boundary='reflective')
    sp_ref = ext_sp(sp_ref)
    return sp_ref  
def dom_mag_sp( sp):
    """
    Domain of three spin components: ensures that spin magnitude is less than one for a given set of three spin components.
    """        
    try:
        assert len(sp) == 3, 'Spin should have three components [s_x, s_y, s_z], but entered spin has length = {} instead'.format(len(sp))
        sp = np.array(sp)
        a = np.linalg.norm([sp[0], sp[1], sp[2]])
        if a != 0:
            a_new = ext_sp(a)
            sp = a_new*sp/a
        return sp 
    except TypeError:
        return dom_indv_sp(sp)     
def dom_sp( sp):
    """
    Final combined function for wrapping of spin values - can handle both 3-component and 1-component spin values.
    Parameters
    ----------
    sp : {float, list}
        Spin value(s).
    Returns
    -------
    {float, list}
        Wrapped spin value(s).
    """
    try:  
        sp = list(map(lambda s: dom_indv_sp(s), sp))
        sp = dom_mag_sp(sp)
    except TypeError:
        sp = dom_indv_sp(sp)
    return sp 
def dom_m( x, m_min=3.5, m_max=None):
    """
    Returns wrapped mass value(s): wrapping and reflection of real line around (3.2, \inf), 
    where \inf is used so that `m > 3.2` is the only real restriction.
    Parameters
    ----------
    x : float
        Mass value to be wrapped within domain.
    m_min : float, optional
        Minimum mass to consider while wrapping. Default = 3.5.
    m_max : {None, float}, optional
        Maximum mass to consider while wrapping. Default = None.
    Returns
    -------
    float
        Wrapped Mass value within domain.
    """
    m_ref = wrap(x, m_min, m_max, boundary='reflective')
    return m_ref     
def dom_chirp( x, cm_min=3.05, cm_max=None):   # because chirp(3.5, 3.5) ~ 3.05
    """
    Returns wrapped Chirp Mass value(s): wrapping and reflection of real line around (3, 1e4), 
    where 1e4 is a large enough number so that `CM > 3` is the only real restriction.
    Parameters
    ----------
    x : float
        Chirp mass value to be wrapped within domain.
    cm_min : float, optional
        Minimum Chirp mass to consider while wrapping. Default = 3.5.
    cm_max : {None, float}, optional
        Maximum Chirp mass to consider while wrapping. Default = None.
    Returns
    -------
    float
        Wrapped Chirp Mass value within domain.
    """       
    cm_ref = wrap(x, cm_min, cm_max, boundary='reflective')
    return cm_ref   

    # domain of Mass Ratio values: wrapping and reflection of real line around (~0, 1).
def dom_q( x, q_min=1/18., q_max=1):
    """
    Returns wrapped mass ratio value(s): wrapping and reflection of real line around (~0, 1),
    assuming q = min(m1/m2, m2/m1) \in (0, 1).
    Parameters
    ----------
    x : float
        Mass ratio value to be wrapped within domain.
    q_min : float, optional
        Minimum mass ratio to consider while wrapping. Default = 3.5.
    q_max : {None, float}, optional
        Maximum mass ratio to consider while wrapping. Default = None.
    Returns
    -------
    float
        Wrapped mass ratio value within domain.
    """  
    x_wrap = wrap(x, q_min, q_max, boundary='reflective')
    return x_wrap

    # domain of Symmetric Mass Ratio values: wrapping and reflection of real line around (~0, 1/4).
def dom_eta( x, eta_min=0.05, eta_max=1/4.):
    """
    Returns wrapped symmetric mass ratio value(s): wrapping and reflection of real line around (~0, 1/4.).
    Parameters
    ----------
    x : float
        Mass ratio value to be wrapped within domain.
    eta_min : float, optional
        Minimum symmetric mass ratio to consider while wrapping. Default = 3.5.
    eta_max : {None, float}, optional
        Maximum symmetric mass ratio to consider while wrapping. Default = None.
    Returns
    -------
    float
        Wrapped symmetric mass ratio value within domain.
    """  
    x_wrap = wrap(x, eta_min, eta_max, boundary='reflective')
    return x_wrap 

def sort_desc(x, col_ind=-2): 
    """
    Sorts an array using nth column in decreasing order.

    """

    return x[x[:,col_ind].argsort()][::-1]  
"""
Here we have functions used for calculating FF using overlap function.
"""
###########################
def optimse_func_2(x, *args):
    """
    This function calculates the overlap by shifting the waveform in time.
    """
    wf, rec = deepcopy(args[0]), deepcopy(args[1]) # wf is the waveform and rec is the recovery waveform which will be shifted.
    t_c = x
    f = rec.sample_frequencies
    rec *= np.exp(1j*(2*np.pi*f*t_c)) #Shifts in time for maximum overlap
    
    overlap = pycbc.filter.matchedfilter.overlap(wf, rec, low_frequency_cutoff=20, high_frequency_cutoff=None, normalized=True)    
    #We return 1- overlap so that it can be used further to minimize.
    return 1-overlap

def overlap_wfs(rec,wf):
    """
    This function optimises the match/overlap by shifting the waveforms in time.
    """
    rec = rec
    wf = wf    
    x0 = 0
    #Minimizing (1 - overlap) using scipy optimize and nelder mead
    vals = scipy.optimize.minimize(optimse_func_2, x0, args=(wf,rec),method='Nelder-Mead',options = {'xatol': 1e-4})   
    # Returning the best match which will be taken as another agument to maximise over.
    return 1 - vals.fun

def objective_wf(x,*args):
    """
    Here we have the objective function where we optimise over intrinsic parameters.
    In this case over Mtotal, q, coa_phase
    """
    x[0], x[1] = dom_m(x[0]), dom_q(x[1])
    Mt, q = x[0],x[1]
    phi_c = x[2]
    #Variables are the parameters on which we want to maximize match over.
    #This fucntion is used in compute_FF as function whose return value we want to maximize

    prms = Mt , q, phi_c

    wf , kwargs = args
    rec = gen_wf(prms,**kwargs) # Genertaing waveforms using gen_wf function.

    #Resizing so there is no length error between data and template.
    flen = max(len(wf), len(rec))
    wf.resize(flen)
    rec.resize(flen)
    #We return 1- overlap so that it can be used further to minimize.
    return 1 - overlap_wfs(rec,wf)

def gen_wf(prms,**kwargs ):
    """
    Generating waveforms based on the parameters of every iteration.
    """
    Mt , q, phi_c = prms
    inj_params = dict(approximant="IMRPhenomXPHM",
                         mass1=Mt * q/ (1 + q), 
                         mass2=Mt  / (1 + q), 
                         delta_t=1.0/4096, 
                         f_lower=20.0,
                         spin1z = 0,
                         spin2z = 0,
                         coa_phase = phi_c, 
                         inclination=kwargs['inclination'],
                         #mode_array = [[2,2]],
                         distance = 100,
                         delta_f =0.01)
    #Wf generation
    rec, _ = get_fd_waveform(**inj_params)   

    return rec

def gen_seed(Mtot,q,sigma_Mt=0.1,sigma_q = 0.02):
    """
    generating seed parameters : The initial parameters that we start our scipy run with.
    """
    Mtot_inj = np.random.normal(Mtot, sigma_Mt, 1)[0]
    q_inj = np.random.normal(q, sigma_q, 1)[0]
    return[dom_m(Mtot_inj), dom_q(q_inj)]

def compute_FF(signal,**kwargs):

    """
    The main function where we calculate max FF value using scipy and iterating over the intrinsic parameters.
    """
    Mtot = kwargs['mt_inj']
    q = kwargs['q_inj']
    seed_params = gen_seed(Mtot,q) #Generates seed Parameters

    x0 = [seed_params[0],seed_params[1],-0.5] #Coa_Phase param self given

    tmp_sig = deepcopy(signal)
    FF =  scipy.optimize.minimize(objective_wf, x0, args=(tmp_sig,kwargs), method='Nelder-Mead',options = {'xatol': 1e-4})   
    return [(1 - FF.fun),list(FF.x)]

#####################
# Injection 

#Loading q and Inclination value on whose parameter space I want to find the trend.
ID = int(sys.argv[1]) #All runs on different process on sarathi
q_l_load = np.loadtxt('/home/krish.shah/GW_Lensing/FF_Computation/NEW_FF/HM/Q_L_Pair.txt')
q_inj, l_inj = q_l_load[ID]

mt_inj = 100.0 #Total mass injected
inj_params = dict(approximant="IMRPhenomXPHM",
                         mass1=mt_inj / (1 + q_inj), 
                         mass2=mt_inj *q_inj  / (1 + q_inj), 
                         delta_t=1.0/4096, 
                         f_lower=20.0,
                         spin1z = 0,
                         spin2z = 0, 
                         inclination=l_inj,
                         distance = 100,
                         #mode_array = [[2,2]],
                         delta_f =0.01)

#Wf generation
hp,hc = get_fd_waveform(**inj_params)

#Lensing Shift
function = np.sqrt(1.0)*np.exp( -1j*np.pi/2)
hpl = hp*function

########################
#Recovery FF

kwargs = dict(mt_inj = mt_inj, q_inj = q_inj, inclination = l_inj)

"""
This is for how many times we want to run our process to obtain the best value of match
"""

def nruns(niters):
    #FF_value = np.zeros(niters)
    FV = []
    
    for i in range(niters):
        A  =  compute_FF(hpl,**kwargs)
        
        XY = [[A[0],A[1]]]
        FV += XY

    return FV

FF_V  = nruns(5) #5 point maximization

FF_V_O = np.array(FF_V,dtype='object')
FF_V_O = sort_desc(FF_V_O)
#print(FF_V_O)
# np.savetxt('FF_Values_HM_New.txt',FF_V_O)
#FF_V_O[0,0]

#Saves file for every run which has multiple outputs ordered according to max FF and one file for each Sarathi run

np.savetxt('/home/krish.shah//GW_Lensing/FF_Computation/NEW_FF/HM/FF_Values_HM_'+str(q_inj)+'_'+str(l_inj)+'.txt', FF_V_O,fmt='%s')