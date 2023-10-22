#!/home/krish.shah/anaconda3/envs/Teobresums/bin/python3

######################################################################################################
### PROGRAM TO COMPUTE FITTING FACTOR (MAX MATCH); Supports MPI Parallelisation ###
# To run as mpi, use bash and execute as follows:
# mpiexec -n numprocs python -m mpi4py pyfile FILENAME.py
######################################################################################################


######################################################################################
### Sec. 1: Importing Packages ###
######################################################################################

import numpy as np

import pycbc
from pycbc.waveform import get_fd_waveform

#from mpmath import *
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from time import time
import func_timeout
import random
from copy import deepcopy

import sys
# sys.path.append('/home/anuj.mishra/my_packages/')
# sys.path.append('/home/anuj/git_repos/GWMAT/src/')
sys.path.append('/home/krish.shah/GW_Lensing/FF_Computation/')
import py_lgw
lgw = py_lgw.lensed_wf_gen()

## MPI Initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# constants
G = 6.67430*1e-11 
c = 299792458. 
M_sun = 1.989*1e30



##########################################################################################
### Sec 2: Class for wrapping parameters within domain while sampling
##########################################################################################
class CBC_parms_domain:
    """
    Contains functions relevant for fitting factor computation.

    """
    def wrap_reflective(self, x, x1, x2):
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
    def wrap_periodic(self, x, x1, x2):
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
    def wrap(self, x, x1, x2, boundary='reflective'):
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
            return self.wrap_reflective(x, x1, x2)
        elif (boundary == 'periodic' or boundary == 'Periodic'):
            return self.wrap_periodic(x, x1, x2)
        else:
            raise KeyError("Incorrect keyword provided for the argument 'boundary'. Allowed keywords are: {'reflective', 'periodic'}.")

    def ext_sp(self, a):
        """
        Checks if the dimensionless spin magnitude `a` satisfies 0.998 < s < 1, otherwise assign a = 0.9.
        """
        if (abs(round(a, 3)) > 0.998):
            return 0.9*a/np.abs(a)
        else:
            return a

    def dom_indv_sp(self, x):
        """
        Domain of an individual spin component: wrapping and reflection of real line around (-1, 1).
        
        """

        sp_ref = self.wrap(x, -1., 1., boundary='reflective')
        sp_ref = self.ext_sp(sp_ref)
        return sp_ref  

    def dom_mag_sp(self, sp):
        """
        Domain of three spin components: ensures that spin magnitude is less than one for a given set of three spin components.

        """        

        try:
            assert len(sp) == 3, 'Spin should have three components [s_x, s_y, s_z], but entered spin has length = {} instead'.format(len(sp))
            sp = np.array(sp)
            a = np.linalg.norm([sp[0], sp[1], sp[2]])
            if a != 0:
                a_new = self.ext_sp(a)
                sp = a_new*sp/a
            return sp 
        except TypeError:
            return self.dom_indv_sp(sp) 
    
    def dom_sp(self, sp):
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
            sp = list(map(lambda s: self.dom_indv_sp(s), sp))
            sp = self.dom_mag_sp(sp)
        except TypeError:
            sp = self.dom_indv_sp(sp)
        return sp 
   

    def dom_m(self, x, m_min=3.5, m_max=None):
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

        m_ref = self.wrap(x, m_min, m_max, boundary='reflective')
        return m_ref     

    def dom_chirp(self, x, cm_min=3.05, cm_max=None):   # because chirp(3.5, 3.5) ~ 3.05
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

        cm_ref = self.wrap(x, cm_min, cm_max, boundary='reflective')
        return cm_ref   

    # domain of Mass Ratio values: wrapping and reflection of real line around (~0, 1).
    def dom_q(self, x, q_min=1/18., q_max=1):
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

        x_wrap = self.wrap(x, q_min, q_max, boundary='reflective')
        return x_wrap

    # domain of Symmetric Mass Ratio values: wrapping and reflection of real line around (~0, 1/4).
    def dom_eta(self, x, eta_min=0.05, eta_max=1/4.):
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

        x_wrap = self.wrap(x, eta_min, eta_max, boundary='reflective')
        return x_wrap 


##########################################################################################
### Sec 3: Functions for Computing Unlensed FF (i.e., usual maximisation used for signals)
##########################################################################################

class FF_UL_4D_aligned_spin(CBC_parms_domain):
    """
    Functions for `wf_model` == 'UL_4D'.

    """ 
    
    def gen_seed_prm_UL_4D(self, chirp_mass=25, q=1, X2_para=0, X1_perp=0, sigma_mchirp=1, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 4D [m1, m2, s_1z, s_2z] for match maximisation; uses reasonable initial bounds.  
        
        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        x2para = np.random.normal(X2_para, sigma_chi, 1)[0]
        x1perp = np.random.normal(X1_perp, sigma_chi, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(x2para), self.dom_sp(x1perp)]    

    def gen_seed_near_best_fit_UL_4D(self, x, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
        chirp_mass, q, x2par, x1per = x
        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        x2para = np.random.normal(x2par, sigma_chi, 1)[0]
        x1perp = np.random.normal(x1per, sigma_chi, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(x2para), self.dom_sp(x1perp)]
        return x_near
   
    def ul_wf_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXP", **kwargs): 
        """
        Generates unlensed Wf.

        """
        
        m1, m2, x2_para, x1_perp = prms
        
        #print(kwargs['spin1x'], kwargs['spin1y'], sz1)
        #spin1x,spin1y,spin1z = kwargs['spin1x'], kwargs['spin1y'], sz1
        #spin2x,spin2y,spin2z = kwargs['spin2x'], kwargs['spin2y'], sz2

        fd_hp, fd_hc = get_fd_waveform(
            approximant = apx,
            mass1 = m1,
            mass2 = m2,
            spin1z =0,
            spin2z= x2_para,
            spin1x =  x1_perp/np.sqrt(2),
            spin1y = x1_perp/np.sqrt(2),
            spin2x = 0,
            spin2y = 0,
            distance = 100.,
            coa_phase = kwargs['coa_phase'],
            inclination = kwargs['inclination'],
            f_ref = kwargs['f_ref'],
            f_lower = f_low,
            delta_f = df) 
        return fd_hp, fd_hc
    
    def objective_func_UL_4D(self, x, *args):    
        """
        Objective function for the maximisation/minimsation.

        """

    #   print(x)
        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = lgw.mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5):  # if mchirp and q doesn't lead to reasonable binary masses, they will be avoided
            return 1e4  # a large number

        x[2], x[3] = self.dom_sp(x[2]), self.dom_sp(x[3])

        gen_prms = m1, m2, x[2], x[3]

        signal, f_low, f_high, apx, psd, kwargs = args
        df_lw = signal.delta_f

        try:
            ul_template = self.ul_wf_gen_fd(gen_prms, df = df_lw, f_low=f_low, apx=apx, **kwargs)[0]
            return -1*lgw.match_wfs_fd(signal, ul_template, f_low=f_low, f_high=f_high, psd=psd)[0]
        except RuntimeError:
            print('warning: returning a large number')
            return 1e4  # a large number



def random_gen_from_powerlaw(alpha, xmin, xmax, N=1):
    r = np.random.power(alpha, N)
    return xmin + (xmax - xmin)*r

def sort_desc(x, col_ind=-2): 
    """
    Sorts an array using nth column in decreasing order.

    """

    return x[x[:,col_ind].argsort()][::-1]  


##########################################################################################
### Sec 6: Main Function for computing FF
##########################################################################################


def compute_fitting_factor(signal, wf_model = 'UL_4D', apx="IMRPhenomXP", psd=None, f_low=20, f_high=None, rank=rank, ncpus=size, 
                           method='serial', n_iters=['default'], xatols=['default'], max_iters=['default'], 
                           branch_num = 3, branch_depth = 3, **kwargs):
    """
    Function to compute the fitting factor (max. match) over unlensed template bank using Nelder-Mead minimation algorithm. 
    The PSD used is aLIGOZeroDetHighPower (almost equivalent to O4). Option to change PSD will come in future versions of the code.
    It supports MPI parallelisation. 
    Caution - Make sure: len(xatols)==len(max_iters)==len(n_iters)

    Parameters
    ----------
    signal : pycbc.types.TimeSeries
        WF for which FF will be evaluated.
    wf_model : {'UL_4D', 'ML_5D', 'ML_6D'}, optional
        WF model to use for recovery. Default = 'UL_4D'.
        * 'UL_4D' represents 4D aligned spin recovery in {chirp_mass, mass_ratio, spin1z, spin2z}.
        * 'ML_5D' represents 5D microlensed aligned spin recovery in {redshifted_lens_mass, impact_parameter, chirp_mass, mass_ratio, spin1z, spin2z}.
        * 'ML_6D' represents 6D microlensed aligned spin recovery in {redshifted_lens_mass, impact_parameter, chirp_mass, mass_ratio, spin1z, spin2z}.

    apx : str, optional
        Name of LAL WF approximant to use for recovery. Default="IMRPhenomXP".
    psd : str, optional
        Path to PSD file. Default = None.
    f_low : ({20., float}), optional 
        Starting frequency for waveform generation (in Hz). 
    f_high : ({None., float}), optional 
        Maximum frequency for matched filter computations (in Hz).  
    rank : int
        Rank of the process. An integer between [0, ncpus-1].
    ncpus : int
        Total number of processors.
    method : {'serial', 'hierarchical'}, optional
        Method to use for finding FF., Default = 'serial'.
    n_iters : {'default', list of floats}, optional
        Number of iterations with different seed points for FF computation. It can be an int or a list of values. Default = ['default'] which is equivalent to 1].
    xatols : {'default', list of floats}, optional
        List of tolerance values of the coordinate while sampling, i.e., absolute error in xopt between iterations that is 
        acceptable for convergence. 
        Default = ['default'] corresponds to xatol = 1e-4.
    max_iters : {'default', list of floats}, optional
        Max number of iterations to use per minimisation. 
        If n_iters is given as list then max_iters has to be a list of similar length. 
        Default = ['default'] corresponds to maxiter = None.
    branch_num : int, optional
        Number of branches to use in heirarchical method. Default = 3.
    branch_depth : int, optional
        Depth of each branch in the heirarchical method. Default 3.

    
    Returns
    -------
    numpy.array, 
        An array containing the combined result from all the cores in the form [FF, best_matched_WF_parameters] for each iteration.
        The array is sorted in descending order, so the best match value will be the 0th element in the array.

    """    
    
    keyword_args = dict(Mtot=np.random.uniform(7, 200), q=random_gen_from_powerlaw(alpha=2, xmin=0.05, xmax=1), 
                        chi_eff = 0, chi_p = 0,
                        coa_phase = 0, inclination = 0, f_ref = f_low,
                        sigma_mchirp=1, sigma_q=0.2, sigma_chi=0.1,
                        max_wait_per_iter=None, default_value=0)
    keyword_args.update(chirp_mass=lgw.mtot_q_to_mchirp(keyword_args['Mtot'], keyword_args['q']))
    keyword_args.update(kwargs)
    
    ### 1. Taking care of inputs
    # (i) if: n_iters is a number then run as many simulations with default settings
    #   else: take input settings from the input lists provided 
    if (isinstance(n_iters, int) == True or isinstance(n_iters, float) == True):
        n_sims = n_iters
        n_iters = ['default']*n_sims
        xatols = ['default']*n_sims
        max_iters = ['default']*n_sims
    else:
        assert len(xatols)==len(max_iters)==len(n_iters), "length of lists containing {xatols, max_iters, n_iters}\
        should be same, but got {%s, %s, %s} instead!/"%(len(xatols), len(max_iters), len(n_iters))

        n_sims = len(xatols)
    
    # (ii) assigning default values to the 'defaults': {itols=1, xatol=1e-4, max_iter=None}
    for i in range(len(n_iters)):
        if n_iters[i] == 'default':
            n_iters[i] = 1 
            
    for i in range(len(xatols)):
        if xatols[i] == 'default':
            xatols[i] = 1e-4

    for i in range(len(max_iters)):
        if max_iters[i] == 'default':
            max_iters[i] = None
     
    # (iii) combining all input sets into one
    inputs = []
    for i in range(n_sims):
        for j in range(n_iters[i]):
            inputs.append([xatols[i], 1, max_iters[i]])
    random.shuffle(inputs)   # shuffling the input set so that all processors get statisticaly the same amount of computational work 
    inputs = np.array(inputs)

    xatols = inputs[:,0]
    n_iters = inputs[:,1]
    max_iters = inputs[:,2]
    
    n_sims = len(xatols)
    assert n_sims == np.sum(n_iters), "something's fishy!"
    
    if wf_model == 'UL_4D':
        FF_UL = FF_UL_4D_aligned_spin()
        gen_prms = FF_UL.gen_seed_prm_UL_4D
        gen_seed_near_best_fit = FF_UL.gen_seed_near_best_fit_UL_4D
        objective_func = FF_UL.objective_func_UL_4D
      
    else:
        raise Exception("Allowed values for the keyword argument 'wf_model' = ['UL_4D', ML_5D, 'ML_6D']. But 'wf_model = %s' provided instead."%(wf_model) )
    
    
    if method not in ['serial', 'hierarchical']:
        raise Exception("Allowed values for the keyword argument 'method' = ['serial', 'hierarchical']. But 'method = %s' provided instead."%(method) )
    
    
    def run_function(f, args, max_wait_per_iter=keyword_args['max_wait_per_iter'], default_value=keyword_args['default_value']):
        try:
            return func_timeout.func_timeout(max_wait_per_iter,
                                             f,
                                             args=args)
        except func_timeout.FunctionTimedOut:
            pass
        return default_value

    def tailored_minimize_func(objective_func, loc_x0, tmp_sig, f_low, f_high, apx, psd, keyword_args, xatol, max_iters):
        return minimize(objective_func, loc_x0, args=(tmp_sig, f_low, f_high, apx, psd, keyword_args), method='Nelder-Mead', options={'disp':False, 'adaptive':True, 'xatol':xatol, 'maxiter':max_iters})

    
    loc_res_prms = []
    d_nsims = len(lgw.distribute(array = range(n_sims), rank = rank, ncpus = ncpus)) # distributing the job among processors

    Mtot, q, chieff, chip = keyword_args['Mtot'], keyword_args['q'], keyword_args['chi_eff'], keyword_args['chi_p']

    X2_para = (chieff)*(1+q)/q
    X1_perp = chip

    sigma_mchirp, sigma_q, sigma_chi = keyword_args['sigma_mchirp'], keyword_args['sigma_q'], keyword_args['sigma_chi']
    #print(Mtot, q, chi_1, chi_2, Mlz, y_lens)
        
    if method == 'serial':
        
        ### 2. Computing max. matches multiple times using different tolerance values and initial points (as provided).
        for i in range(d_nsims):   
            comp_masses = lgw.mtot_q_to_m1m2(Mtot, q)
            
            if wf_model == 'UL_4D':
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, X2_para=X2_para, X1_perp=X1_perp,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi)

            loc_x0 = tmp_best_val
            tmp_sig = deepcopy(signal)
            
            loc_res = run_function(tailored_minimize_func, 
                                   [objective_func, loc_x0, tmp_sig, f_low, f_high, apx, psd, keyword_args, xatols[i], max_iters[i]], 
                                  )
            #print(loc_res.fun, loc_res.x)                   
            #loc_res = minimize(objective_func, loc_x0, args=(tmp_sig, f_low, f_high, apx, keyword_args), method='Nelder-Mead', 
            #                   options={'disp':False, 'adaptive':True, 'xatol':xatols[i], 'maxiter':max_iters[i]})
            
            if loc_res != keyword_args['default_value']:
                tmp_FF_val = -1*objective_func(loc_res.x, tmp_sig, f_low, f_high, apx, psd, keyword_args)
                tmp_FF_prms = list(loc_res.x)
            
            else:
                tmp_FF_val = keyword_args['default_value']
                tmp_FF_prms = [0]*len(tmp_best_val)
            
            tmp_loc_res_prms = [[tmp_FF_val, tmp_FF_prms]]
            loc_res_prms += tmp_loc_res_prms        
    
    if method == 'hierarchical':
        
        for i in range(d_nsims): 
            print('\n(rank: %s, iter #%s):'%(rank, i+1))
            comp_masses = lgw.mtot_q_to_m1m2(Mtot, q)
            
            if wf_model == 'UL_4D':
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi)
            
            for j in range(branch_depth):
                print('\n(rank: %s, branch_depth #%s):'%(rank, j+1))

                loc_branch_res_prms = []
                for k in range(branch_num):
                    print('\n(rank: %s, branch_num #%s):'%(rank, k+1))
                    print(xatols[i], max_iters[i])
                    loc_x0 = gen_seed_near_best_fit(tmp_best_val) #gen_seed_near_best_fit(tmp_best_val)
                    tmp_sig = deepcopy(signal)
                    loc_res = minimize(objective_func, loc_x0, args=(tmp_sig, f_low, f_high, apx, psd, keyword_args), method='Nelder-Mead',\
                                       options={'adaptive':True, 'disp':False, 'xatol':xatols[i], 'maxiter':max_iters[i]})
                    tmp_loc_res_prms = [[-1*objective_func(loc_res.x, tmp_sig, f_low, f_high, apx, psd, keyword_args), list(loc_res.x)]]
                    loc_branch_res_prms += tmp_loc_res_prms     
                    loc_res_prms += tmp_loc_res_prms
                    
                comb_branch_res = np.array(loc_branch_res_prms, dtype='object')
                comb_branch_res = sort_desc(comb_branch_res) 
                tmp_best_val = comb_branch_res[0][-1]
                
        
    comm.Barrier()
    results = comm.allreduce(loc_res_prms)
    comb_res = np.array(results, dtype='object')
    comb_res = sort_desc(comb_res)        
    return comb_res      

