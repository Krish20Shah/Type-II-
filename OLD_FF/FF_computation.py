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
    
    def gen_seed_prm_UL_4D(self, chirp_mass=25, q=1, chi_1=0, chi_2=0, sigma_mchirp=1, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 4D [m1, m2, s_1z, s_2z] for match maximisation; uses reasonable initial bounds.  
        
        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(chi_1, sigma_chi, 1)[0]
        sz2 = np.random.normal(chi_2, sigma_chi, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2)]    

    def gen_seed_near_best_fit_UL_4D(self, x, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
        chirp_mass, q, a1, a2 = x
        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(a1, sigma_chi, 1)[0]
        sz2 = np.random.normal(a2, sigma_chi, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2)]
        return x_near
   
    def ul_wf_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXP", **kwargs): 
        """
        Generates unlensed Wf.

        """
        
        m1, m2, sz1, sz2 = prms
        #print(kwargs['spin1x'], kwargs['spin1y'], sz1)
        #spin1x,spin1y,spin1z = kwargs['spin1x'], kwargs['spin1y'], sz1
        #spin2x,spin2y,spin2z = kwargs['spin2x'], kwargs['spin2y'], sz2
        spin1x,spin1y,spin1z = self.dom_sp( [kwargs['spin1x'], kwargs['spin1y'], sz1])
        spin2x,spin2y,spin2z= self.dom_sp( [kwargs['spin2x'], kwargs['spin2y'],sz2])
        fd_hp, fd_hc = get_fd_waveform(
            approximant = apx,
            mass1 = m1,
            mass2 = m2,
            spin1z =spin1z,
            spin2z= spin2z,
            spin1x =spin1x,
            spin1y = spin1y,
            spin2x = spin2x,
            spin2y = spin2y,
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


##########################################################################################
### Sec 4: Functions for Computing Microlensed FF
##########################################################################################

# ## Uncomment the following and give proper paths for computing Microlensed FFs
# #path to the dir containing cython module
# path_to_pnt_lens_class = '/home/anuj/git_repos/GWMAT/pnt_Ff_lookup_table/src/'
# #path to the lookup lookup table for point lens amplification factors
# path_to_lookup_table_data = '/home/anuj/git_repos/GWMAT/pnt_Ff_lookup_table/data/point_lens_Ff_lookup_table_Geo_relErr_1p0.pkl'
# # '/home/anuj.mishra/misc_data/pnt_Ff_grid_data/lookup_table_generation/data/point_lens_Ff_lookup_table_Geo_relErr_1p0.pkl'
# 
# # loading Cython version of the point lens class
# sys.path.append(path_to_pnt_lens_class)   
# import cythonized_pnt_lens_class as pnt_lens_cy
 
class point_lens_MiL:

    def __init__(self):        
        print('## Loading and setting up the lookup table for point-lens amplification factor ##')
        import pickle
        with open(path_to_lookup_table_data, 'rb') as f:
            self.Ff_grid = pickle.load(f)
            self.ys_grid, self.ws_grid = self.y_w_grid_data(self.Ff_grid) 
        print('## Done ##')       
        
    ## functions
    def y_w_grid_data(self, Ff_grid):
        ys_grid = np.array([Ff_grid[str(i)]['y'] for i in range(len(Ff_grid))])
        ws_grid = Ff_grid['0']['ws']
        return ys_grid, ws_grid

    def y_index(self, yl, ys_grid):
        return np.argmin(np.abs(ys_grid - yl))

    def w_index(self, w, ws_grid):
        return np.argmin(np.abs(ws_grid - w))

    def pnt_Ff_lookup_table(self, fs, Mlz, yl, ys_grid=None, ws_grid=None, extrapolate=True):
        ys_grid, ws_grid = self.ys_grid, self.ws_grid
        wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
        wc = pnt_lens_cy.wc_geo_re1p0(yl)

        wfs_1 = wfs[wfs <= np.min(ws_grid)]
        Ffs_1 = np.array([1]*len(wfs_1))

        wfs_2 = wfs[(wfs > np.min(ws_grid))&(wfs <= np.max(ws_grid))]
        wfs_2_wave = wfs_2[wfs_2 <= wc]
        wfs_2_geo = wfs_2[wfs_2 > wc]

        i_y  = self.y_index(yl, ys_grid)
        tmp_Ff_dict = self.Ff_grid[str(i_y)]
        ws = tmp_Ff_dict['ws']
        Ffs = tmp_Ff_dict['Ffs_real'] + 1j*tmp_Ff_dict['Ffs_imag']
        fill_val = ['interpolate', 'extrapolate'][extrapolate]
        i_Ff = interp1d(ws, Ffs, fill_value=fill_val)
        Ffs_2_wave = i_Ff(wfs_2_wave)

        Ffs_2_geo = np.array([pnt_lens_cy.point_Fw_geo(w, yl) for w in wfs_2_geo])

        wfs_3 = wfs[wfs > np.max(ws_grid) ]
        Ffs_3 = np.array([pnt_lens_cy.point_Fw_geo(w, Mlz) for w in wfs_3])

        Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))
        assert len(Ffs)==len(fs), 'len(Ffs) = {} does not match len(fs) = {}'.format(len(Ffs), len(fs))
        return Ffs     
    
    
class FF_ML_6D_aligned_spin(point_lens_MiL, FF_UL_4D_aligned_spin):   
    """
    Functions for `wf_model` == 'ML_6D'.

    """
    
    def gen_seed_prm_ML_6D(self, chirp_mass=25, q=1, chi_1=0, chi_2=0, Mlz=500, y_lens=1, 
                           sigma_Mlz=10, sigma_y=0.5, sigma_mchirp=0.5, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 4D [m1, m2, s_1z, s_2z] for match maximisation; uses reasonable initial bounds.

        """ 

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(chi_1, sigma_chi, 1)[0]
        sz2 = np.random.normal(chi_2, sigma_chi, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2),\
                self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]    

    def gen_seed_near_best_fit_ML_6D(self, x, sigma_Mlz=10, sigma_y=0.5, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
        chirp_mass, q, sz1, sz2, Mlz, y_lens = x

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz1 = np.random.normal(sz1, sigma_chi, 1)[0]
        sz2 = np.random.normal(sz2, sigma_chi, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz1), self.dom_sp(sz2),\
                 self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]
        return x_near
       
        
    def ml_wf_gen_fd(self, prms, df, f_low=20, apx="IMRPhenomXP", **kwargs): 

        m1, m2, sz1, sz2, Mlz, yl = prms
        tmp_prms = m1, m2, sz1, sz2
        fd_hp, fd_hc  = self.ul_wf_gen_fd(tmp_prms, df, apx=apx, f_low=f_low, **kwargs)

        # Adding Microlensing effects
        if round(Mlz) == 0:
            return fd_hp, fd_hc
        else:
            fs = fd_hp.sample_frequencies
            wfs = np.array([pnt_lens_cy.w_of_f(f, Mlz) for f in fs])
            
            Ff = self.pnt_Ff_lookup_table(fs=fs, Mlz=Mlz, yl=yl)
            lfd_hp = Ff*fd_hp
            lfd_hp = pycbc.types.FrequencySeries(lfd_hp, delta_f = df)

            lfd_hc = Ff*fd_hc
            lfd_hc = pycbc.types.FrequencySeries(lfd_hc, delta_f = df)
            return lfd_hp, lfd_hc     
        
        
    def objective_func_ML_6D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

    #     print(x)
        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = lgw.mchirp_q_to_m1m2(x[0], x[1])
        if (m1 < 3.5 or m2 < 3.5):  # if mchirp and q doesn't lead to reasonable binary masses, they will be avoided
            return 1e4  # a large number

        x[2], x[3] = self.dom_sp(x[2]), self.dom_sp(x[3])
        x[4], x[5] = self.wrap_reflective(x[4], 10, 1e4), self.wrap_reflective(x[5], 0.01, 3)

        gen_prms = m1, m2, x[2], x[3], x[4], x[5]

        signal, f_low, f_high, apx, psd, kwargs = args
        df_lw = signal.delta_f
        
        try:
            ml_template = self.ml_wf_gen_fd(gen_prms, df = df_lw, f_low=f_low, apx=apx, **kwargs)[0]
            return -1*lgw.match_wfs_fd(signal, ml_template, f_low=f_low, f_high=f_high, psd=psd)[0]
        except RuntimeError:
            print('warning: returning a large number')
            return 1e4  # a large number 
        
class FF_ML_5D_aligned_spin(FF_ML_6D_aligned_spin):
    """
    Functions for `wf_model` = 'ML_5D'.

    """ 


    def gen_seed_prm_ML_5D(self, chirp_mass=25, q=0.5, chi=0, Mlz=500, y_lens=1, sigma_Mlz=10, sigma_y=0.5, sigma_mchirp=0.5, sigma_q=0.2, sigma_chi=0.2):
        """
        Generates seed point in 4D [m1, m2, s_1z, s_2z] for match maximisation; uses reasonable initial bounds.  

        """

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz = np.random.normal(chi, sigma_chi, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        return [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz),\
                self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]

    def gen_seed_near_best_fit_ML_5D(self, x, sigma_Mlz=10, sigma_y=0.5, sigma_mchirp = 0.5, sigma_q = 0.1, sigma_chi = 0.1):
    #     m1, m2, a1, a2 = x
    #     chirp_mass, q = lgw.m1m2_to_mchirp(m1, m2), lgw.m1m2_to_q(m1, m2)
        chirp_mass, q, a, Mlz, y_lens = x

        mchirp = np.random.normal(chirp_mass, sigma_mchirp, 1)[0]
        q = np.random.normal(q, sigma_q, 1)[0]
        sz = np.random.normal(x[2], sigma_chi, 1)[0]
        Mlz = np.random.normal(Mlz, sigma_Mlz, 1)[0]
        y_lens = np.random.normal(y_lens, sigma_y, 1)[0]
        x_near = [self.dom_chirp(mchirp), self.dom_q(q), self.dom_sp(sz),\
                 self.wrap_reflective(Mlz, 10, 1e4), self.wrap_reflective(y_lens, 0.01, 3)]
        return x_near


    def objective_func_ML_5D(self, x, *args):
        """
        Objective function for the maximisation/minimsation.

        """

    #     print(x)
        x[0], x[1] = self.dom_chirp(x[0]), self.dom_q(x[1])
        m1, m2 = lgw.mchirp_q_to_m1m2(x[0], x[1])

        if (m1 < 3.5 or m2 < 3.5):  # if mchirp and q doesn't lead to reasonable binary masses, they will be avoided
            return 1e4  # a large number

        x[2] = self.dom_sp(x[2])
        x[3], x[4] = self.wrap_reflective(x[3], 10, 1e4), self.wrap_reflective(x[4], 0.01, 3)

        gen_prms = m1, m2, x[2], x[2], x[3], x[4]

        signal, f_low, f_high, apx, psd, kwargs = args
        df_lw = signal.delta_f
        #print(x[4])
        try:
            ml_template = self.ml_wf_gen_fd(gen_prms, df = df_lw, f_low=f_low, apx=apx, **kwargs)[0]
            return -1*lgw.match_wfs_fd(signal, ml_template, f_low=f_low, f_high=f_high, psd=psd)[0]
        except RuntimeError:
            print('warning: returning a large number')
            return 1e4  # a large number


##########################################################################################
### Sec. 5: Defining some useful functions 
##########################################################################################

def inject_microlensed_signal(Mlz, y_lens, Mtot, q, apx = "IMRPhenomXP", f_low = 20., f_high=None):
    """
    Initialises the setup for given microlensing and non-spinning binary parameters. 
    Returns: label, lensed and unlensed WFs in FD and TD, direct match between UL and ML WF, and frequency spacing (df) for lensed WF.  

    """
    
    #label = str(proc_ID) + '_' + wf_model + '_model_FF'+'_Mlz_' + lgw.str_m(Mlz) + '_y_' + lgw.str_y(y_lens) + '_Mtot_' + lgw.str_m(Mtot) + '_q_' + lgw.str_y(q)
    label = 'Mlz_' + lgw.str_m(Mlz) + '_y_' + lgw.str_y(y_lens) + '_Mtot_' + lgw.str_m(Mtot) + '_q_' + lgw.str_y(q)

    if Mtot > 20:    # because f_RD(Mtot=20) ~ 900 Hz
        sample_rate=2048
    else:
        sample_rate=4096

    dt = 1./sample_rate

    init_prms = dict(f_low=f_low, f_high=f_high, sample_rate=sample_rate, approximant=apx)
    lgw.sample_rate = sample_rate
    lgw.f_low = f_low
    lgw.apx = apx

    m1, m2 = lgw.mtot_q_to_m1m2(Mtot, q)
    cbc_prms = dict(mass_1=m1, mass_2=m2, a_1=0, a_2=0, tilt_1=0, tilt_2=0, phi_12=0, phi_jl=0, 
                    luminosity_distance=100, theta_jn=0, polarization=0, coa_phase=0, trig_time=1242529720)

    ## lensed waveform generation
    lens_prms = dict(m_lens=Mlz, y_lens=y_lens, z_lens=0)
    prms = {**lens_prms, **cbc_prms}
    lwfs = lgw.lensed_pure_polarized_wf_gen(**prms)

    # choosing plus polarized WF for the analysis (without loss of generality)
    lwf_fd = lwfs['lensed_FD_WF_hp']
    lwf_td = lwfs['lensed_TD_WF_hp']

    ## Unlensed waveform
    uwf_fd = lwfs['unlensed_FD_WF_hp']
    uwf_td = lwfs['unlensed_TD_WF_hp']

    df_lw = lwf_fd.delta_f    # frequency spacing for the lensed WF
    
    # match calculations
    m_td, _ = lgw.match_wfs_td(uwf_td, lwf_td, f_low=f_low, f_high=f_high)  # Direct Match: m(UL, ML)
    m_fd, _ = lgw.match_wfs_fd(uwf_fd, lwf_fd, f_low=f_low, f_high=f_high)
    d_match = max(m_fd, m_td)
   
    if rank == 0:  
        if round(m_td, 4) == round(m_fd, 4):
            print('\nmatch(uwf, lwf): {:.5f}'.format(d_match))
        else:
            print('\nCAUTION: Discripency in Match Calculation\nmatch(uwf_td, lwf_td): {:.5f}'.format(m_td))
            print('match(uwf_fd, lwf_fd): {:.5f}'.format(m_fd))
    
    res_dict = dict(label=label, lwf_fd=lwf_fd, lwf_td=lwf_td, uwf_fd=uwf_fd, uwf_td=uwf_td, df_lw=df_lw, d_match=d_match)
    #lgw.print_dict(res_dict)
    return res_dict  


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
                        spin1z=0, spin2z=0,
                        spin1x=0, spin1y=0, spin2x=0, spin2y=0, 
                        Mlz=10**np.random.uniform(1, 4), y_lens=random_gen_from_powerlaw(alpha=2, xmin=0.01, xmax=3), 
                        coa_phase = 0, inclination = 0, f_ref = f_low,
                        sigma_mchirp=1, sigma_q=0.2, sigma_chi=0.2,
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
    elif wf_model == 'ML_6D':
        FF_ML = FF_ML_6D_aligned_spin()
        gen_prms = FF_ML.gen_seed_prm_ML_6D
        gen_seed_near_best_fit = FF_ML.gen_seed_near_best_fit_ML_6D
        objective_func = FF_ML.objective_func_ML_6D       
    elif wf_model == 'ML_5D':
        FF_ML = FF_ML_5D_aligned_spin()
        gen_prms = FF_ML.gen_seed_prm_ML_5D
        gen_seed_near_best_fit = FF_ML.gen_seed_near_best_fit_ML_5D
        objective_func = FF_ML.objective_func_ML_5D         
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

    Mtot, q, sz_1, sz_2 = keyword_args['Mtot'], keyword_args['q'], keyword_args['spin1z'], keyword_args['spin2z']
    Mlz, y_lens = keyword_args['Mlz'], keyword_args['y_lens'] 
    sigma_mchirp, sigma_q, sigma_chi = keyword_args['sigma_mchirp'], keyword_args['sigma_q'], keyword_args['sigma_chi']
    #print(Mtot, q, chi_1, chi_2, Mlz, y_lens)
        
    if method == 'serial':
        
        ### 2. Computing max. matches multiple times using different tolerance values and initial points (as provided).
        for i in range(d_nsims):   
            comp_masses = lgw.mtot_q_to_m1m2(Mtot, q)
            
            if wf_model == 'UL_4D':
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi)
            elif wf_model == 'ML_6D':
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi, Mlz=Mlz, y_lens=y_lens)
            elif wf_model == 'ML_5D':
                chi = (comp_masses[0]*sz_1 + comp_masses[1]*sz_2)*1.0 / Mtot
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi=chi,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi, Mlz=Mlz, y_lens=y_lens)
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
            elif wf_model == 'ML_6D':
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi_1=sz_1, chi_2=sz_2,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi, Mlz=Mlz, y_lens=y_lens)
            elif wf_model == 'ML_5D':
                chi = (comp_masses[0]*sz_1 + comp_masses[1]*sz_2)*1.0 / Mtot
                tmp_best_val = gen_prms(chirp_mass=lgw.m1m2_to_mchirp(comp_masses[0], comp_masses[1]), q=q, chi=chi,
                                        sigma_mchirp=sigma_mchirp, sigma_q=sigma_q, sigma_chi=sigma_chi, Mlz=Mlz, y_lens=y_lens)
            
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


if __name__ == '__main__':
    # Inj parameters:
    Mtot, q = 60, 1
    Mlz, y = 100, 0.1
    inj_prms = Mtot, q, Mlz, y

    f_low, f_high = 20., None
    f_ref = f_low

    wf_model = "UL_4D"  # ["UL_4D", "ML_5D", "ML_6D"]
    apx_inj = "IMRPhenomXP" 
    apx_rec = "IMRPhenomXP" 

    inj_res = inject_microlensed_signal(Mlz, y, Mtot, q, apx=apx_inj)

    ## computing FF

    lwf_fd = inj_res['lwf_fd']

    f_low, f_high = 20, None
    n_iters = 15

    kwargs = dict(Mtot=Mtot, q=q, Mlz=Mlz, y_lens=y,
                 max_wait_per_iter = 1e3, default_value = 0.0 )

    t1 = time()
    FF_res = compute_fitting_factor(lwf_fd, wf_model=wf_model, apx=apx_rec, f_low=f_low, f_high=f_high, psd=None, \
                                      n_iters=[n_iters], xatols=['default'], max_iters=['default'], branch_num=None, branch_depth=None,\
                                                      method='serial', **kwargs)
    t2 = time()
    print('Best recovered match (FF) in {} runs: {}'.format(n_iters, FF_res[0][0]) )
    print('Parameters corresponding to the best matched WF (m_chirp, q, Sz1, Sz2): ', FF_res[0][1])
    print('Total Computation time: {:.4f} s'.format(t2-t1))
