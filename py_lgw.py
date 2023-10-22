#!/home/krish.shah/anaconda3/envs/Teobresums/bin/python3
### importing packages

__author__ = "Anuj Mishra"

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

import math
import collections
from copy import deepcopy

from mpmath import hyp1f1, gamma

import lal
import lalsimulation

import pycbc
from pycbc import types, waveform
from pycbc.waveform import get_td_waveform
from pycbc.filter import match
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.detector import Detector
from pycbc.frame import frame
import pycbc.noise
import pycbc.psd
import lalinference.imrtgr.nrutils as nr

import matplotlib.pyplot as plt

import os
cwd = os.path.dirname(__file__) + '/'


# Constants
G_SI = lal.G_SI  #6.67430*1e-11 
c_SI = lal.C_SI  #299792458. 
Msun_SI = lal.MSUN_SI  #1.989*1e30



# User Defined Classes
class point_lens:
    """
    A class containing functions for studying an isolated point lens case. 
    In such cases, two images are formed: one minima (type I) and one saddle (type II). 

    """
    
    # image positions
    def x_minima(self, y):
        """
        Returns the image position for the minima (type I) image.
        
        Parameters
        ----------
        y : float
           The impact parameter. 

        Returns
        -------
        float:
           Image position.

        """

        return (y+np.sqrt(y**2+4))/2
    
    def x_saddle(self, y):
        """
        Returns the image position for the saddle-point (type II) image.
        
        Parameters
        ----------
        y : float
           The impact parameter. 

        Returns
        -------
        float :
           Image position.

        """

        return (y-np.sqrt(y**2+4))/2
    
    #image magnifications
    def magification_minima(self, y):
        """
        Returns the image magnification for the minima (type I) image.
        
        Parameters
        ----------
        y : float
           The impact parameter. 

        Returns
        -------
        float :
           Image magnification.

        """

        return 1/2 + (y**2+2)/(2*y*np.sqrt(y**2+4))   
    
    def magification_saddle(self, y):
        """
        Returns the image magnification for the saddle-point (type II) image.
        
        Parameters
        ----------
        y : float
           The impact parameter. 

        Returns
        -------
        float :
           Image magnification.

        """

        return 1/2 - (y**2+2)/(2*y*np.sqrt(y**2+4))

    #time delay between the two micro-images
    def td_point_dimensionless(self, y):
        """
        Returns the dimensionless time-delay between the two micro-images.
        
        Parameters
        ----------
        y : float
           The impact parameter. 

        Returns
        -------
        float :
           dimensionless time-delay between micro-images.

        """

        return (y*np.sqrt(y**2+4))/2. + np.log((np.sqrt(y**2+4)+y)/(np.sqrt(y**2+4)-y))

    def td_point_sec(self, ml, y, zl=0):
        """
        Returns the time-delay between the two micro-images in seconds.
        
        Parameters
        ----------
        ml : float
            Microlens mass.
        y : float
           The impact parameter. 
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making it as the redshifted lens mass).

        Returns
        -------
        float:
           time-delay between micro-images in seconds.

        """

        return (4*G_SI*Msun_SI*ml*(1+zl)/c_SI**3)*((y*np.sqrt(y**2+4))/2. + np.log((np.sqrt(y**2+4)+y)/(np.sqrt(y**2+4)-y)))

    # Geometric and Quasi Geometric approximations
    def point_Fw_geo(self, w, y):
        """
        Returns the lensing amplification factor, F(w), assuming geometric optics approximation.
        
        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter. 
           
        Returns
        -------
        complex :
            Amplification factor F(w).

        """

        return np.sqrt(np.abs(self.magification_minima(y))) - 1j*np.sqrt(np.abs(self.magification_saddle(y)))*np.exp(1j*w*self.td_point_dimensionless(y))
    
    def point_Fw_Qgeo(self, w, y):
        """
        Returns the lensing amplification factor, F(w), assuming Quasi-geometric optics approximation.
        References - arXiv:0402165
        
        Parameters
        ----------
        w : float 
           Dimensionless frequency.
        y : float
           The impact parameter. 
           
        Returns
        -------
        complex :
           Amplification factor, F(w).

        """

        return self.point_Fw_geo(w,y) + (1j/(3*w))*((4*self.x_minima(y)**2-1)/(pow((self.x_minima(y)**2+1),3)*(self.x_minima(y)**2-1)))*np.sqrt(np.abs(self.magification_minima(y))) + \
               (1/(3*w))*((4*self.x_saddle(y)**2-1)/(pow((self.x_saddle(y)**2+1),3)*(self.x_saddle(y)**2-1)))*np.sqrt(np.abs(self.magification_saddle(y)))*np.exp(1j*w*self.td_point_dimensionless(y)) 
    
    # dimensionless frequency (w) in terms of dimensionful frequency (f), w(f)
    def w_of_f(self, f, ml, zl=0):
        """
        Converts a dimensionful frequency (f) to the dimensionless frequency (w).
        
        Parameters
        ----------
        f : float 
           The dimensionful frequency which is to be converted.
        ml : float
            Microlens mass.
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making it as the redshifted lens mass).

        Returns
        -------
        float :
           The dimensionless frequency, w(f).

        """

        wf = f*8.*np.pi*G_SI*ml*(1+zl)*Msun_SI/c_SI**3
        return wf
    
    # dimensionful frequency (f) in terms of dimensionless frequency (w), f(w)
    def f_of_w(self, w, ml, zl=0):
        """
        Converts a dimensionless frequency (w) to the dimensionful frequency (f).
        
        Parameters
        ----------
        w : float 
           The dimensionless frequency which is to be converted.
        ml : float
            Microlens mass.
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making it as the redshifted lens mass).

        Returns
        -------
        float :
           The dimensionful frequency, f(w).

        """

        fw = w * (c_SI**3) / (8.*np.pi*G_SI*ml*(1+zl)*Msun_SI)        
        return fw

    # cutoff frequencies for transition to geometric optics
    def wc_geo_re0p1(self, y):
        """
        Returns a cutoff dimensionless frequency (wc) for a given y such that w > wc gives relative error < 0.1 % 
        when geometric optics approximation is used. Valid for y in range (0.01, 5.00).
        
        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00). 
            
        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        if y <= 0.12:
            wc = 15112.5 - 52563.5*y
        elif 0.12 < y <= 1.5:
            wc = -34.08 - 12.84*pow(y,-1) + 114.33*pow(y,-2) + 0.89*pow(y,-3) 
        elif y > 1.5:
            wc = -15.02 + 18.25*y - 2.66*y**2
        if (y < 0.01 or round(y,3) > 5.00):
            print('Warning: y = {} is outside interpolation range (0.01, 5.00). Thus, Extrapolating! '.format(y))
        return wc

    def wc_geo_re1p0(self, y):
        """
        Returns a cutoff dimensionless frequency (wc) for a given y such that w > wc gives relative error < 1.0 % 
        when geometric optics approximation is used. Valid for y in range (0.01, 5.00).
        
        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00). 
            
        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        if y <= 0.071:
            wc = 16604 - 202686*y
        elif y > 0.071:
            wc = 0.64 + 0.97*pow(y, -1) + 6*pow(y, -2) + 0.38*pow(y, -3)  
        if (y < 0.01 or round(y,3) > 5.00):
            print('Warning: y = {} is outside interpolation range (0.01, 5.00). Thus, Extrapolating! '.format(y))
        return wc
    
    # cutoff frequencies for transition to Quasi-geometric optics
    def wc_Qgeo_re0p1(self, y):
        """
        Returns a cutoff dimensionless frequency (wc) for a given y such that w > wc gives relative error < 0.1 % 
        when Quasi-geometric optics approximation is used. Valid for y in range (0.01, 5.00).
        
        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00). 
            
        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        wc = 9*pow(y,-1) + 0.04*pow(y,-2)

        if (y < 0.01 or round(y,3) > 5.00):
            print('Warning: y = {} is outside interpolation range (0.01, 5.00). Thus, Extrapolating! '.format(y))
        return wc  

    def wc_Qgeo_re1p0(self, y):
        """
        Returns a cutoff dimensionless frequency (wc) for a given y such that w > wc gives relative error < 1.0 % 
        when Quasi-geometric optics approximation is used. Valid for y in range (0.01, 5.00).
        
        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00). 
            
        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        wc = 4*pow(y,-1) - np.log(y)/5 
        if (y < 0.01 or round(y, 3) > 5.00):
            print('Warning: y = {} is outside interpolation range (0.01, 5.00). Thus, Extrapolating! '.format(y))
        return wc  
       
    # Amplification factor related functions    
    def point_Fw(self, w, y):
        """
        Returns the amplification factor, F(w, y), for point lens using the analytical formula. 
        It breaks down, or is difficult to compute, when the system approaches geometrical optics regime.
        In cases where it is not converging, use _point_Fw_eff()_ which can handle any values.
        
        Parameters
        ----------
        w : float 
           The dimensionless frequency.
        y : float
           The impact parameter. 
            
        Returns
        -------
        complex:
           The Amplification Factor, F(w, y).

        """

        if w==0:
            return 1
        else:
            w = np.float128(w)
            xm = np.float128((y+np.sqrt(y*y + 4.))/2.)
            pm = np.float128(pow(xm-y,2)/2 - np.log(xm))
            hp = np.log(w/2.)-(2.*pm)
            h = np.exp((np.pi*w/4.)+1j*(hp*w/2.))
            gm = gamma(1.-(1j*w/2.))
            hf = hyp1f1((1j*w/2.),1.,(1j*y*y*w/2.))
            Ff= h*gm*hf
            return complex(Ff.real, Ff.imag)  
   
    def point_Fw_eff(self, w, y):
        """
        An efficient computation of the point-lens amplification factor, F(w, y), that uses analytical expression along with 
        the knowledge of the Quasi-geo. and the geo. optics limit. 
        It is recommended to be used in general as it can handle any values. 
        
        For dimensionful variant of this function, use: _point_Ff_eff()_.
        
        For mapping this function to an array of frequencies, use _point_Fw_eff_map()_
        Parameters
        ----------
        w : float 
           The dimensionless frequency which is to be converted.
        y : float
           The impact parameter. 
            
        Returns
        -------
        complex :
           The Amplification Factor, F(w, y).

        """

        wc_geo = self.wc_geo_re0p1(y)
        wc_Qgeo = self.wc_Qgeo_re0p1(y)
        if w < wc_Qgeo:
            return self.point_Fw(w,y)
        elif (w>=wc_Qgeo and w<wc_geo):
            return self.point_Fw_Qgeo(w,y)
        else:
            return self.point_Fw_geo(w,y)

    def point_Ff_eff(self, f, ml, y, zl=0):
        """     
        Returns an efficient computation of the point-lens amplification factor, F(f, ml, y, zl=0), that uses analytical expression along with 
        the knowledge of the Quasi-geo. and the geo. optics limit. 
        It is recommended to be used in general as it can handle any values. 
        
        For dimensionless variant of this function, use: _point_Fw_eff()_.
        
        For mapping this function to an array of frequencies, use _point_Ff_eff_map()_.
        
        Parameters
        ----------
        f : float 
           Frequency (in Hz).
        ml : float
           Microlens mass.
        y : float
           The impact parameter. 
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making it as the redshifted lens mass).
           
        Returns
        -------
        complex:
           The Amplification Factor, F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return  self.point_Fw_eff(w, y)         
        
    def point_Fw_eff_map(self, ws, y):
        """
        Mapping fucntion for point_Fw_eff(). This takes an array of (dimensionless) frequencies as input.
        
        Parameters
        ----------
        ws : float array.
            Array of dimensionless frequencies.
        y : float
           The impact parameter. 
            
        Returns
        -------
        complex array :
           Array containing the amplification factors, F(ws, y).

        """

        return np.array(list(map(lambda w: self.point_Fw_eff(w, y), ws)))
    
    def point_Ff_eff_map(self, fs, ml, y, zl=0):
        """
        Mapping fucntion for point_Ff_eff(). This takes an array of frequencies as input.

        Parameters
        ----------
        f : float array
           Array of Frequencies (in Hz).
        ml : float
            Microlens mass.
        y : float
           The impact parameter. 
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making ml as the redshifted lens mass).
           
        Returns
        -------
        complex array :
            Array containing the amplification factors, F(fs, ml, y, zl).

        """

        return np.array(list(map(lambda f: self.point_Ff_eff(f, ml, y, zl), fs)))
    
    def point_Fw_map(self, ws, y):
        """
        Mapping fucntion for point_Fw(). This takes an array of (dimensionless) frequencies as input.
        
        Parameters
        ----------
        ws : float array
            Array of dimensionless frequencies.
        y : float
           The impact parameter. 
            
        Returns
        -------
        complex array :
           Array containing the amplification factors, F(ws, y).

        """

        return np.array(list(map(lambda w: self.point_Fw(w, y), ws)))
    
    def point_Ff(self, f, ml, y, zl=0):
        """
        Returns amplification factor, F(f, ml, y, zl=0), for point lens using actual analytic formula. 
        It breaks down, or is difficult to compute, when the system approaches geometrical optics regime.
        
        This function is dimensionful variant of point_Fw().
        
        Parameters
        ----------
        fs : float
            Frequency (in Hz).
        ml : float
            Microlens mass.
        y : float
           The impact parameter. 
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making it as the redshifted lens mass).
            
        Returns
        -------
        complex :
           The Amplification Factor, F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.point_Fw(w, y)

    def point_Ff_map(self, fs, ml, y, zl=0):
        """
        Mapping fucntion for _point_Ff()_. This takes an array of frequencies as input.
        
        Parameters
        ----------
        fs : float array
            Array of Frequencies (in Hz).
        ml : float
            Microlens mass.
        y : float
           The impact parameter. 
        zl : float, optional
            Lens-redshift. Default = 0 (this is equivalent to absorbing the (1+zl) term into ml thereby making it as the redshifted lens mass).
            
        Returns
        -------
        complex array :
           Array containing the amplification factors, F(fs, ml, y, zl).

        """

        return np.array(list(map(lambda f: self.point_Ff(f, ml, y, zl), fs)))
    
    def point_Ff_geo(self, f, ml, y, zl=0):
        """
        Returns the lensing amplification factor, F(f, ml, y, zl=0), assuming geometric optics approximation.
        
        This function is dimensionful variant of point_Fw_geo().
        
        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter. 
           
        Returns
        -------
        complex :
            Amplification factor F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.point_Fw_geo(w, y)
    
    def point_Ff_Qgeo(self, f, ml, y, zl=0):
        """
        Returns the lensing amplification factor, F(f, ml, y, zl=0), assuming Quasi-geometric optics approximation.
        
        This function is dimensionful variant of point_Ff_Qgeo().
        
        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter. 
           
        Returns
        -------
        complex :
            Amplification factor F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.point_Fw_geo(w, y)
    


class cosmology:
    """
    Contains basic functions to compute cosmological distances as functions of redshift, and vice-versa.

    """
    
    def __init__(self, Omega_m=0.315, Omega_lambda=0.685, Omega_k=0., Ho=69.8e3):
        """
        Fixing cosmology.

        Parameters
        ----------
        Omega_m : float, optional
            matter_density, by default 0.315
        Omega_lambda : float, optional
            dark_energy_density, by default 0.685
        Omega_k : float, optional
            determines curvature, by default 0.
        Ho : float, optional
            Hubbles constant (ms^-1 Mpc^-1), by default 69.8e3

        """

        self.Omega_m = Omega_m             # matter_density     
        self.Omega_lambda = Omega_lambda   # dark_energy_density
        self.Omega_k = Omega_k             # determines curvature
        self.Omega_r = 1 - self.Omega_m - self.Omega_lambda - self.Omega_k   #radiation_energy_density
        self.Ho = Ho  # (ms^-1 Mpc^-1)
        self.DHo = c_SI/self.Ho     # Mpc   
    
    def inverse_dimless_Hubble_parameter(self, z):
        """ 
        Returns Ho/H(z) = 1/E(z), inverse of the dimensionless Hubble_parameter.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            Ho/H(z) = 1/E(z)

        """

        return self.DHo*pow(self.Omega_r*(1 + z)**4 + self.Omega_m*(1 + z)**3 + self.Omega_k*(1 + z)**2 + self.Omega_lambda, -1/2)

    def d_comoving(self, z):
        """
        Returns the comoving distance as a function of redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            The comoving distance, in Mpc.

        """   

        return quad(self.inverse_dimless_Hubble_parameter, 0, z)[0]

    def d_ang_dia(self, z):
        """
        Returns the angular diameter distance as a function of redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            The angular diameter distance, in Mpc.

        """     

        return pow(1+z,-1)*self.d_comoving(z)

    def d_ang_dia_zs(self, z1, z2):
        """
        Returns the angular diameter distance between two redshifts z1 and z2.

        Parameters
        ----------
        z1 : float
            First Redshift
        z2 : float
            Second Redshift

        Returns
        -------
        float
            The angular diameter distance between the two redshift, in Mpc.

        """     

        return pow(1+z2,-1)*(self.d_comoving(z2)*np.sqrt(1+self.Omega_r*pow(self.d_comoving(z1)/self.DHo, 2)) -     
                    self.d_comoving(z1)*np.sqrt(1+self.Omega_r*pow(self.d_comoving(z2)/self.DHo, 2)))
    
    def d_luminosity(self, z):
        """
        Returns the luminosity distance as a function of redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            The luminosity distance, in Mpc.

        """     

        return (1+z)*self.d_comoving(z)
    
    def z_at_d_comoving(self, dc):
        """
        Finds the redshift corresponding a given comoving distance using binary search.

        Parameters
        ----------
        dc : float
            The comoving distance, in Mpc.

        Returns
        -------
        float
            Redshift
            
        """   
              
        z_min, z_max = 0, 10  
        z = (z_min+z_max)/2.
        dist = self.d_comoving(z=z)
        while abs(1 - dc/dist) > 0.001:
            z = (z_min+z_max)/2.
            dist = self.d_comoving(z=z)
            if dist < dc:
                z_min = z
            else:
                z_max = z
        return z  

    def z_at_d_luminosity(self, dL):
        """
        Finds the redshift corresponding a given luminosity distance using binary search. 

        Parameters
        ----------
        dL : float
            The luminosity distance, in Mpc.

        Returns
        -------
        float
            Redshift

        """  

        z_min, z_max = 0, 10 
        z = (z_min+z_max)/2.
        dist = self.d_luminosity(z=z)
        while abs(1 - dL/dist) > 0.001:
            z = (z_min+z_max)/2.
            dist = self.d_luminosity(z=z)
            if dist < dL:
                z_min = z
            else:
                z_max = z
        return z 
      
    
class GW_analysis:
    """
    Contains basic functions relevant for analysing GWs from CBCs.

    """

    def r_ISCO(self, m_tot, a_f):
        """
        Returns the equatorial Innermost Stable Circular Orbit (ISCO), also known as radius of the marginally stable orbit. For Kerr metric, it depends on 
        whether the orbit is prograde (negative sign) or retrograde (positive sign).
        References: 
        Eq.2.21 of _Bardeen et al. <https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B/abstract>_,
        Eq.1 of _Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>_
        
        Parameters
        ----------
        m_tot = m1+m2 : float
            Binary mass (in solar masses).
        a_f : float
            Dimensionless spin parameter of the remnant compact object.

        Returns
        -------
        dict: 
            Dictionary of:

            * R_ISCO_retrograde: float
                ISCO radius for a particle in retrogade motion (in solar masses).

        """

        fac = m_tot
        z1 = 1 + np.cbrt(1 - a_f**2)*(np.cbrt(1 + a_f) + np.cbrt(1 - a_f))
        z2 = np.sqrt(3*a_f**2 + z1**2)
        risco_n = fac*(3 + z2 - np.sqrt((3 - z1)*(3 + z1 + 2*z2)))
        risco_p = fac*(3 + z2 + np.sqrt((3 - z1)*(3 + z1 + 2*z2)))
        r_dict = dict(R_ISCO_retrograde=risco_p, R_ISCO_prograde=risco_n)
        return r_dict

    def f_GW_Kerr_ISCO(self, m_tot, a_f):         
        """
        Returns GW frequency at ISCO for a spinning BH Binary.
        References: Eq.4 of `Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>`

        Parameters
        ----------
        m_tot=m1+m2 : float
            Binary Mass (in solar masses).
        a_f : float
            Final dimensionless spin magnitude of the remnant.

        Returns
        -------
        dict: 
            Dictionary of:

            * f_ISCO_retrograde: float
                GW frequency at ISCO for binaries in retrogade motion (in solar masses).
            * f_ISCO_prograde: float
                ISCO radius for a particle in prograde motion (in solar masses).  
                 
        """     

        fac = c_SI**3/(2*np.pi*G_SI*m_tot*Msun_SI)
        r_res = self.r_ISCO(m_tot, a_f)
        r_n = r_res['R_ISCO_prograde']
        r_p = r_res['R_ISCO_retrograde']
        f_orb_isco_n = fac*(a_f + pow(r_n/m_tot, 3/2))**(-1)
        f_orb_isco_p = fac*(a_f + pow(r_p/m_tot, 3/2))**(-1)
        f_n, f_p = 2*f_orb_isco_n, 2*f_orb_isco_p    # because of quadrupolar contributions, f_gw = 2 * f_orb
        f_dict = dict(f_ISCO_retrograde=f_p, f_ISCO_prograde=f_n)
        return f_dict

    def f_GW_Schwarz_ISCO(self, m_tot): 
        """
        Returns GW frequency at ISCO for a non-spinning BH Binary.
        References: Eq.4 of `Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>`

        Parameters
        ----------
        m_tot = m1+m2: float
           Binary Mass (in solar masses).

        Returns
        -------
        float
          GW Frequency at ISCO (in Hz).

        """        
        return self.f_GW_Kerr_ISCO(m_tot, a_f=0)['f_ISCO_prograde']

    def f_GW_BKL_ISCO(self, m1, m2):
        """
        Mass ratio dependent GW frequency at ISCO derived from estimates of the final spin
        of a merged black hole in a paper by Buonanno, Kidder, Lehner
        (arXiv:0709.3839).  See also arxiv:0801.4297v2 eq.(5)

        Parameters
        ----------
        m1 : float or numpy.array
            The mass of the first component object in the binary (in solar masses).
        m2 : float or numpy.array
            The mass of the second component object in the binary (in solar masses).

        Returns
        -------
        f : float or numpy.array
            GW Frequency at ISCO (in Hz).

        """

        # q is defined to be in [0,1] for this formula
        q = np.minimum(m1/m2, m2/m1)
        return 0.5 * self.f_GW_Schwarz_ISCO(m1+m2) * ( 1 + 2.8*q - 2.6*q**2 + 0.8*q**3 )


    def remnant_prms_estimate(self, m1, m2, a1=0., a2=0., tilt1=0., tilt2=0., phi12=0.):
        """
        Returns the mass and spin of the final remnant based on initial binary configuration.

        Parameters
        ----------
        m1 : float or numpy.array
            The mass of the first component object in the binary (in solar masses).
        m2 : float or numpy.array
            The mass of the second component object in the binary (in solar masses).
        a1 : float, optional
            The dimensionless spin magnitude of the first binary component. Default = 0.
        a2 : float, optional
            The dimensionless spin magnitude of the second binary component. Default = 0.
        tilt1 : float, optional
            Zenith angle between S1 and LNhat (rad). Default = 0.
        tilt2 : float, optional
            Zenith angle between S2 and LNhat (rad). Default = 0.
        phi12 : float, optional
            Difference in azimuthal angle between S1 and S2 (rad). Default = 0.

        Returns
        -------
        dict: 
            Dictionary of:

            * m_f: float
                Final mass of the remnant (in solar masses).
            * a_f: float
                Final dimensionless spin of the remnant.    

        """        
                
        # Use the following final mass and spin fits to calculate fISCO
        Mf_fits = ["UIB2016", "HL2016"]
        af_fits = ["UIB2016", "HL2016", "HBR2016"]

        # Final mass computation does not use phi12, so we set it to zero
        Mf = nr.bbh_average_fits_precessing(
            m1,
            m2,
            a1,
            a2,
            tilt1,
            tilt2,
            phi12=np.array([0.0]),
            quantity="Mf",
            fits=Mf_fits,
        )
        af = nr.bbh_average_fits_precessing(
            m1, m2, a1, a2, tilt1, tilt2, phi12=phi12, quantity="af", fits=af_fits
        )
        return dict(m_f=float(Mf), a_f=float(af))


    def f_RD_dimensionless(self, a_f):
        """
        Return the dimensionless fundamental RingDown frequency.

        Parameters
        ----------
        a_f : float
            Final dimensionless spin magnitude of the remnant.

        Returns
        -------
        float
            Dimensionless fundamental RingDown frequency. 

        """ 

        f1 = 1.5251
        f2 = -1.1568
        f3 = 0.1292
        FRD = f1 + f2*(1 - a_f)**f3
        return FRD


    def f_RD_BBH(self, m1, m2, chi1=0, chi2=0, tilt1=0, tilt2=0, phi12=0):   
        """
        Fundamental RingDown frequency calculated from the Berti, Cardoso and
        Will (gr-qc/0512160) value for the omega_220 QNM frequency:
        <https://arxiv.org/pdf/gr-qc/0512160.pdf>

        Parameters
        ----------
        m1 : float or numpy.array
            The mass of the first component object in the binary (in solar masses).
        m2 : float or numpy.array
            The mass of the second component object in the binary (in solar masses).
        a1 : float, optional
            The dimensionless spin magnitude of the first binary component. Default = 0.
        a2 : float, optional
            The dimensionless spin magnitude of the second binary component. Default = 0.
        tilt1 : float, optional
            Zenith angle between S1 and LNhat (rad). Default = 0.
        tilt2 : float, optional
            Zenith angle between S2 and LNhat (rad). Default = 0.
        phi12 : float, optional
            Difference in azimuthal angle between S1 and S2 (rad). Default = 0.

        Returns
        -------
        float
            Ringdown Frequency (in Hz).

        """  

        remnant_prms = self.remnant_prms_estimate(m1, m2, chi1, chi2, tilt1, tilt2, phi12)
        m_f = remnant_prms['m_f']
        a_f = remnant_prms['a_f']
        FRD_dimless = self.f_RD_dimensionless(a_f)
        fac = (c_SI**3)/(2*np.pi*G_SI*m_f*Msun_SI)
        FRD = fac * FRD_dimless
        return FRD

    def match_wfs_fd(self, wf1, wf2, psd=None, f_low = 20., f_high=None, subsample_interpolation=True, is_asd_file=False):
        """
        Computes match (overlap maximised over phase and time) between two frequency domain WFs.

        Parameters
        ----------
        wf1 : pycbc.types.frequencyseries.FrequencySeries object
            PyCBC time domain Waveform.
        wf2 : pycbc.types.frequencyseries.FrequencySeries object
            PyCBC time domain Waveform.
        psd: {None, str}
            PSD file to use for computing match. Default = None.
            Predefined_PSDs: {'aLIGOZeroDetHighPower'}
        f_low : {None, float}, optional
            The lower frequency cutoff for the match computation. Default = 20.
        f_high : {None, float}, optional
            The upper frequency cutoff for the match computation. Default = None.
        subsample_interpolation : ({False, bool}, optional)
            If True the peak will be interpolated between samples using a simple quadratic fit. 
            This can be important if measuring matches very close to 1 and can cause discontinuities if you don’t use it as matches move between discrete samples. 
            If True the index returned will be a float instead of int. Default = True.
        is_asd_file : bool, optional
            Is psd provided corresponds to an asd file? Default = False.

        Returns
        -------
        match_val : float
            match value Phase to rotate complex waveform to get the match, if desired.
        index_shift : float
            The number of samples to shift to get the match.
        phase_shift : float
            Phase to rotate complex waveform to get the match, if desired.   

        """ 

        flen = max(len(wf1), len(wf2))
        wf1.resize(flen)
        wf2.resize(flen)

        delta_f = wf1.delta_f
        if psd is not None:
            if psd=='aLIGOZeroDetHighPower':
                psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
            else:
                psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low, is_asd_file=is_asd_file)   
        # match_val, index_shift, phase_shift = match( wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high, subsample_interpolation=subsample_interpolation, return_phase=True )
        match_val, index_shift = match( wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high )
        return match_val, index_shift #, phase_shift   

    def match_wfs_td(self, wf1, wf2, psd=None, f_low = 20., f_high=None, subsample_interpolation=True, is_asd_file=False):    
        """
        Computes match (overlap maximised over phase and time) between two time domain WFs.

        Parameters
        ----------
        wf1 : pycbc.types.timeseries.TimeSeries object
            PyCBC time domain Waveform.
        wf2 : pycbc.types.timeseries.TimeSeries object
            PyCBC time domain Waveform.
        psd: {None, str}
            PSD file to use for computing match. Default = None.
            Predefined_PSDs: {'aLIGOZeroDetHighPower'}
        f_low : {None, float}, optional
            The lower frequency cutoff for the match computation. Default = 20.
        f_high : {None, float}, optional
            The upper frequency cutoff for the match computation. Default = None.
        subsample_interpolation : ({False, bool}, optional)
            If True the peak will be interpolated between samples using a simple quadratic fit. 
            This can be important if measuring matches very close to 1 and can cause discontinuities if you don’t use it as matches move between discrete samples. 
            If True the index returned will be a float instead of int. Default = True.
        is_asd_file : bool, optional
            Is psd provided corresponds to an asd file? Default = False.

        Returns
        -------
        match_val : float
            match value Phase to rotate complex waveform to get the match, if desired.
        index_shift : float
            The number of samples to shift to get the match.
        phase_shift : float
            Phase to rotate complex waveform to get the match, if desired.    

        """   

        tlen = max(len(wf1), len(wf2))
        wf1.resize(tlen)
        wf2.resize(tlen)

        delta_f = wf1.delta_f
        flen = tlen//2+1
        
        if psd is not None:
            if psd=='aLIGOZeroDetHighPower':
                psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
            else:
                psd = pycbc.psd.from_txt(psd, flen, delta_f, f_low, is_asd_file=is_asd_file)      
        # match_val, index_shift, phase_shift = match(wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high, subsample_interpolation=subsample_interpolation, return_phase=True)
        match_val, index_shift = match(wf1, wf2, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        return match_val, index_shift #, phase_shift
        

    def cumulative_phase(self, phase_array, period=np.pi):
        """
        Returns the unwrapped phase.

        Parameters
        ----------
        phase_array : np.array of floats
            Array containing wrapped phase values.
        period : float, optional
            Period of the signal. Default = np.pi.

        Returns
        -------
        np.array
            Array containing the unwrapped phase values.

        Example
        ------- 
        theta_arr = np.linspace(0, 30*np.pi, 1000)
        zs = np.exp(1j*theta_arr)   # rotation of a unit vector in a complex plane (should result in a linear increase in phase)
        plt.plot(np.angle(zs))
        plt.title('wrapped')
        plt.show()
        plt.plot(cumulative_phase(np.angle(zs), period=2*np.pi)/np.pi)
        plt.title('unwrapped')
        plt.show()     

        """ 

        unwrapped_phase = np.zeros(len(phase_array), dtype=complex)
        k = 0
        for i in range(len(phase_array)-1):
            unwrapped_phase[i] = k*(period) + phase_array[i]
            diff = np.abs(phase_array[i] - phase_array[i+1])
            if diff > period/2:
                k += 1
        unwrapped_phase[-1] =  k*(period) + phase_array[-1]    
        return np.real(unwrapped_phase)

            
    def wf_phase(self, wf):
        """
        Returns the phase of a GW waveform.

        Parameters
        ----------
        wf : {complex np.array, pycbc.types.frequencyseries.FrequencySeries object}
            Array containing the complex amplitude of the signal, or an object of type pycbc.types.frequencyseries.

        Returns
        -------
        complex np.array
            Phase of the GW signal (in rads).

        """

        wf = np.array(wf)
        phase = np.zeros(len(wf))
        for i in range(len(wf)):
            if np.abs(wf[i]) == 0:
                phase[i] = 0
            else:
                phase[i] = np.arctan(np.imag(wf[i])/np.real(wf[i]))
        return phase

    # Ref.: FINDCHIRP (arXiv:0509116)
    def tchirp_2PN(self, mtot, eta, f_low=20):
        """
        Chirp time of a GW signal assuming 2PN approximation.
        Reference: Bruce Allen, et al. - `FINDCHIRP <https://arxiv.org/abs/gr-qc/0509116>`

        Parameters
        ----------
        mtot : float
            _description_
        eta : float
            _description_
        f_low : int, optional
            _description_, by default 20

        Returns
        -------
        float
            Chirp time.

        """   

        fac = G_SI*mtot*Msun_SI/c_SI**3
        v_low = np.power(fac*np.pi*f_low, 1/3)
        tchirp = fac*(5/(256*eta))*(v_low**(-8) + ((743/252) + (11/3)*eta)*v_low**(-6) - (32*np.pi/5)*v_low**(-5) + ((3058673/508032) + (5429/504)*eta + (617/72)*eta**2)*v_low**(-4))
        return tchirp

    # more accurate function than above
    def rough_wf_duration(self, m1, m2, f_low=20., threshold=1.):
        """
        Returns the rough WF duration for a given binary component masses and f_low. 
        Duration is computated via actual generation of the WF using the WF approximant IMRPhenomPv2, and defined as:
        Chirp_duraion = duration between the times where strain amplitude reached $threshold (%) of the peak amplitude around the time of peak amplitude.

        Parameters
        ----------
        m1 : float 
            The mass of the first component object in the binary (in solar masses).
        m2 : float 
            The mass of the second component object in the binary (in solar masses).
        f_low : float, optional
            The lower frequency cutoff for the generation of WF. Default = 20.
        threshold : float, optional
            Threshold (%) to defined the start and end of a WF as a fraction of the peak amplitude. It represents the ratio between the start of a WF to the peak of the WF (in percetage).
            Default = 1 (i.e., the duration will be defined as the time when the amplitude first became 1% of the peak amplitude untill the time after which it again went below that).
            Caution: Very low threshold values will result in error, so values below 0.1% are NOT recommended. Values such as 0.5% is ideal, while 1% (default) works decently well.
        Returns
        -------
        dict: 
            Dictionary of:

            * hp : pycbc.types.timeseries.TimeSeries
                Pure polarised time-domain WF.
            * hc : pycbc.types.timeseries.TimeSeries
                Pure polarised time-domain WF.    
            * insp_to_merger_duration : float   
                Rough WF duration in the Inspiral-Merger Phase.
            * post_merger_duration : float
                Rough WF duration in the Post-Merger Phase.
            * chirp_duration : float
                Rough WF duration for whole WF in the IMR Phase.
            * rough_duration_for_PE : int
                Rough WF duration to use for PE.

        """  

        # threshold represents the ratio between the start of a WF to the peak of the WF (in percetage)
        # Very low threshold value will result in error. Don't give below 0.1%, 0.5% is ideal, while 1% (default) is decently well.
        # returns the chirp duration of a WF
        # this assumes trigger is at t=0 (true for pycbc WFs)
        hp, hc = get_td_waveform(approximant="IMRPhenomPv2",
                                    mass1 = m1,
                                    mass2 = m2,
                                    delta_t = 1./2048,
                                    f_lower = f_low)
        
        hp = hp.cyclic_time_shift(-0.2)
        
        sr = 1./hp.delta_t
        max_hp = max(hp)
        len_hp = len(hp)
        
        # duration from Inspiral till Merger
        ind = 0
        d_ind = round(sr/128) #sampling every (1/128)th of a second
        while hp[ind]/max_hp < threshold*1e-2 and ind < len_hp:
            ind += d_ind

        insp_to_merger_duration = -1*hp.sample_times[ind-d_ind]   

        # duration of Post-Merger Signal
        ind = -1
        d_ind = round(sr/512) #sampling every (1/512)th of a second
        while hp[ind]/max_hp < threshold*1e-2 and ind < len_hp:
            ind -= d_ind
        
        post_merger_duration = hp.sample_times[ind+d_ind]
        
        # total Duration of the WF
        chirp_duration = insp_to_merger_duration + post_merger_duration
        
        # duration of the WF for PE
        if insp_to_merger_duration < 0.5:
            rough_dur = 2
        else:
            rough_dur = np.power(2, np.ceil( np.log2( chirp_duration ) ) ) #2*round(chirp_duration/2+0.5) + 2
        
        # storing and returning all results as dictionary 
        res = dict(hp=hp, hc=hc, insp_to_merger_duration = insp_to_merger_duration, post_merger_duration = post_merger_duration, \
                    chirp_duration = chirp_duration, rough_duration_for_PE=rough_dur)
        return res   


    def jframe_to_l0frame(self, mass_1, mass_2, f_ref, phi_ref=0., theta_jn=0., phi_jl=0., a_1=0., a_2=0.,
                            tilt_1=0., tilt_2=0., phi_12=0.):  
        """
        [This function is inherited from PyCBC and lalsimulation.]
        Function to convert J-frame coordinates (which Bilby uses for PE) to L0-frame coordinates (that Pycbc uses for waveform generation).
        J stands for the total angular momentum while L0 stands for the orbital angular momentum.

        Parameters
        ----------
        mass_1 : float 
            The mass of the first component object in the binary (in solar masses).
        mass_2 : float 
            The mass of the second component object in the binary (in solar masses).
        f_ref : float
            The reference frequency (in Hz).
        phi_ref : float, optional
            The orbital phase at ``f_ref``. Default = 0.
        theta_jn : float, optional
            Angle between the line of sight and the total angular momentume J. Default = 0.
        phi_jl : float, optional
            Azimuthal angle of L on its cone about J. Default = 0.
        a_1 : float, optional
            The dimensionless spin magnitude. Default = 0.
        a_2 : float, optional
            The dimensionless spin magnitude. Default = 0.
        tilt_1 : float, optional
            Angle between L and the spin magnitude of object 1. Default = 0.
        tilt_2 : float, optional
            Angle between L and the spin magnitude of object 2. Default = 0.
        phi_12 : float, optional
            Difference between the azimuthal angles of the spin of the 
        object 1 and 2. Default = 0.

        Returns
        -------
        dict :
            Dictionary of:

            * inclination : float
                Inclination (rad), defined as the angle between the orbital angular momentum L and the line-of-sight at the reference frequency.
            * spin1x : float
                The x component of the first binary component's dimensionless spin.
            * spin1y : float
                The y component of the first binary component's dimensionless spin.
            * spin1z : float
                The z component of the first binary component's dimensionless spin.
            * spin2x : float
                The x component of the second binary component's dimensionless spin.
            * spin2y : float
                The y component of the second binary component's dimensionless spin.
            * spin2z : float
                The z component of the second binary component's dimensionless spin.

        """ 

        inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = \
            lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn, phi_jl, tilt_1, tilt_2, phi_12,
                a_1, a_2, mass_1*lal.MSUN_SI, mass_2*lal.MSUN_SI, f_ref,
                phi_ref)
        out_dict = {'inclination': inclination,
                    'spin1x': spin1x,
                    'spin1y': spin1y,
                    'spin1z': spin1z,
                    'spin2x': spin2x,
                    'spin2y': spin2y,
                    'spin2z': spin2z}
        return out_dict

    def l0frame_to_jframe(self, mass_1, mass_2, f_ref, phi_ref=0., inclination=0.,
                            spin1x=0., spin1y=0., spin1z=0.,
                            spin2x=0., spin2y=0., spin2z=0.):
        """
        [This function is inherited from PyCBC and lalsimulation.]
        Function to convert L-frame (that Pycbc uses for waveform generation) coordinates to J-frame coordinates (which Bilby uses for PE).
        J stands for the total angular momentum while L0 stands for the orbital angular momentum.

        Parameters
        ----------
        mass_1 : float 
            The mass of the first component object in the binary (in solar masses).
        mass_2 : float 
            The mass of the second component object in the binary (in solar masses).
        f_ref : float
            The reference frequency (in Hz).
        phiref : float
            The orbital phase at ``f_ref``.
        inclination : float
            Inclination (rad), defined as the angle between the orbital angular momentum L and the
            line-of-sight at the reference frequency. Default = 0.
        spin1x : float
            The x component of the first binary component's. Default = 0.
            dimensionless spin.
        spin1y : float
            The y component of the first binary component's. Default = 0.
            dimensionless spin.
        spin1z : float
            The z component of the first binary component's. Default = 0.
            dimensionless spin.
        spin2x : float
            The x component of the second binary component's. Default = 0.
            dimensionless spin.
        spin2y : float
            The y component of the second binary component's. Default = 0.
            dimensionless spin.
        spin2z : float
            The z component of the second binary component's. Default = 0.
            dimensionless spin.

        Returns
        -------
        dict :
            Dictionary of:

            * theta_jn : float, optional
                Angle between the line of sight and the total angular momentume J. 
            * phi_jl : float, optional
                Azimuthal angle of L on its cone about J. 
            * a_1 : float, optional
                The dimensionless spin magnitude. 
            * a_2 : float, optional
                The dimensionless spin magnitude. 
            * tilt_1 : float, optional
                Angle between L and the spin magnitude of object 1.
            * tilt_2 : float, optional
                Angle between L and the spin magnitude of object 2.
            * phi_12 : float, optional
                Difference between the azimuthal angles of the spin of the object 1 and 2. 

        """    

        thetajn, phijl, s1pol, s2pol, s12_deltaphi, spin1_a, spin2_a = \
            lalsimulation.SimInspiralTransformPrecessingWvf2PE(
                inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                mass_1, mass_2, f_ref, phi_ref)
        out = {'theta_jn': thetajn,
                'phi_jl': phijl,
                'a_1': spin1_a,
                'a_2': spin2_a,
                'tilt_1': s1pol,
                'tilt_2': s2pol,
                'phi_12': s12_deltaphi
                }
        return out

    def GW_WF_approximants(self):
        """
        Prints available Wf approximants in the time and frequency domain.

        Returns
        -------
        dict:
            Dictionary of:

            *  fd_approximants : list
                List of available FD approximants. 
            *  td_approximants : list
                List of available TD approximants.       

        """      

        total_apxs = lalsimulation.NumApproximants
        list_of_fd_apxs = []
        list_of_td_apxs = []
        for i in range(total_apxs):
            if lalsimulation.SimInspiralImplementedFDApproximants(i):
                list_of_fd_apxs.append(lalsimulation.GetStringFromApproximant(i))

            if lalsimulation.SimInspiralImplementedTDApproximants(i):
                list_of_td_apxs.append(lalsimulation.GetStringFromApproximant(i)) 

        approximants = dict(fd_approximants=list_of_fd_apxs, td_approximants=list_of_td_apxs)
        return approximants    


    def cyclic_time_shift_of_WF(self, wf, rwrap=0.2):
        """
        Inspired by PyCBC's function pycbc.types.TimeSeries.cyclic_time_shift(), 
        it shifts the data and timestamps in the time domain by a given number of seconds (rwrap). 
        Difference between this and PyCBCs function is that this function preserves the sample rate of the WFs while cyclically rotating, 
        but the time shift cannot be smaller than the intrinsic sample rate of the data, unlike PyCBc's function.
        To just change the time stamps, do ts.start_time += dt.
        Note that data will be cyclically rotated, so if you shift by 2
        seconds, the final 2 seconds of your data will now be at the
        beginning of the data set.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            The waveform for cyclic rotation.
        rwrap : float, optional
            Amount of time to shift the vector. Default = 0.2.

        Returns
        -------
        pycbc.types.TimeSeries
            The time shifted time series.

        """        

        # This function does cyclic time shift of a WF.
        # It is similar to PYCBC's "cyclic_time_shift" except for the fact that it also preserves the Sample Rate of the original WF.
        if rwrap is not None and rwrap != 0:
            sn = abs(int(rwrap/wf.delta_t))     # number of elements to be shifted 
            cycles = int(sn/len(wf))

            cyclic_shifted_wf = wf.copy()

            sn_new = sn - int(cycles * len(wf))

            if rwrap > 0:
                epoch = wf.sample_times[0] - sn_new * wf.delta_t
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[-sn_new:]
                    tmp_wf_p2 = wf_arr[:-sn_new] 
                    shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)
            else:
                epoch = wf.sample_times[sn_new]
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[sn_new:] 
                    tmp_wf_p2 = wf_arr[:sn_new]
                    shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)  

            for i in range(cycles):        
                    epoch = epoch - np.sign(rwrap)*wf.duration
                    wf_arr = np.array(cyclic_shifted_wf)[:]
                    cyclic_shifted_wf = pycbc.types.TimeSeries(wf_arr, delta_t = wf.delta_t, epoch = epoch)

            assert len(cyclic_shifted_wf) == len(wf), 'Length mismatch: cyclic time shift added extra length to WF.'
            return cyclic_shifted_wf
        else:
            return wf  

    def Hz_to_mass_rescaled_freq(self, f, M):
        fac = lal.G_SI * lal.MSUN_SI / (lal.C_SI**3) 
        return f * M * fac
    
    def mass_rescaled_freq_to_Hz(self, fgeo, M):
        fac = lal.G_SI * lal.MSUN_SI / (lal.C_SI**3)
        return fgeo/M/fac 

    
class cbc_parameter_conversions:
    """
    Contains functions related to parameter conversions relevant for binaries.

    """
        
    def m1m2_to_mchirp(self, m1, m2):
        """
        Returns the chirp mass corresponding to the binary component masses.

        Parameters
        ----------
        m1 : float
            Mass of the first component of the binary, in Msun.
        m2 : float
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        float
            Chirp mass, in Msun.

        """

        return (m1*m2)**(3/5.)/(m1+m2)**(1/5.)

    def m1m2_to_q(self, m1, m2):
        """
        Returns the mass ratio of the binary component masses.
        Convention: q = m_secondary/m_primary < 1

        Parameters
        ----------
        m1 : float
            Mass of the first component of the binary, in Msun.
        m2 : float
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        float, < 1.0
            Mass ratio.

        """

        return min(m1, m2)/max(m1, m2)


    def mchirp_q_to_m1m2(self, m_chirp, q):
        """
        Converts a given chirp mass and mass ratio to the binary component masses.

        Parameters
        ----------
        m_chirp : float
            Chirp mass of the binary.
        q : float
            Mass ratio of the binary.

        Returns
        -------
        float, float
            The binary component masses    

        """

        m2=(m_chirp*(1.+q)**(1/5.))/q**(3/5.)
        m1=q*m2
        return m1, m2

    def m1m2_to_eta(self, m1, m2):   #symmetric mass ratio
        """
        Returns the symmetric mass ratio for a given binary system.

        Parameters
        ----------
        m1 : float
            Mass of the first component of the binary, in Msun.
        m2 : float
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        float, < 0.25
            The symmetric mass ratio.

        """

        return m1*m2 / (m1+m2)**2

    def m1m2_szs_to_effective_spin(self, m1, m2, sz1, sz2):   #effective spin
        """
        Returns the effective spin as a function of the componenet masses and aligned spin vlaues.

        Parameters
        ----------
        m1 : float
            Mass of the first component of the binary, in Msun.
        m2 : float
            Mass of the second component of the binary, in Msun.
        sz1 : float, < 1.
            z-component of the spin of the first component of the binary.
        sz2 : float, < 1.
            z-component of the spin of the second component of the binary.

        Returns
        -------
        float
            The effective spin (chi_eff) of the binary.

        """

        return (m1*sz1+m2*sz2)/(m1+m2)    

    def mchirp_q_szs_to_effective_spin(self, m_chirp, q, sz1, sz2):   #effective spin
        """
        Returns the effective spin as a function of the chirp mass and mass ratio of the binary.

        Parameters
        ----------
        m_chirp : float
            The chirp mass of the binary.
        q : float
            The mass ratio of the binary.
        sz1 : float, < 1.
            z-component of the spin of the first component of the binary.
        sz2 : float, < 1.
            z-component of the spin of the second component of the binary.

        Returns
        -------
        float
            The effective spin (chi_eff) of the binary.

        """

        m1, m2 = self.mchirp_q_to_m1m2(m_chirp, q)
        return (m1*sz1+m2*sz2)/(m1+m2) 
    
    def mchirp_eta_to_m1m2(self, m_chirp, eta):
        """
        Converts a given chirp mass and symmetric mass ratio of the binary to the individual componenet masses.

        Parameters
        ----------
        m_chirp : float
            The chirp mass of the binary.
        eta : float
            The symmetric mass ratio of the binary (0< eta <=0.25).

        Returns
        -------
        float, float
            Binary componenet masses

        """           

        assert eta<=0.25, 'Symmetric mass ratio should lie between 0 and 0.25, but the given eta was {}'.format(eta)
        mtot=m_chirp*eta**(-3/5)
        m1=0.5*mtot*(1+(1-4*eta)**0.5)
        m2=0.5*mtot*(1-(1-4*eta)**0.5)
        return m1, m2    

    def mtot_q_to_m1m2(self, mtot, q): 
        """
        Converts a given total mass and mass ratio of the binary to the individual componenet masses.

        Parameters
        ----------
        mtot : float
            The total mass of the binary.
        q : float
            The mass ratio of the binary.

        Returns
        -------
        float, float
            Binary componenet masses.

        """        

        m1=mtot/(1+q)
        m2=q*m1
        return max(m1, m2),  min(m1, m2)

    def mtot_q_to_mchirp(self, mtot, q): 
        """
        Converts a given total mass and mass ratio of the binary to its chirp mass.

        Parameters
        ----------
        mtot : float
            The total mass of the binary.
        q : float
            The mass ratio of the binary.

        Returns
        -------
        float
            Chirp mass of the binary.

        """        

        m1=mtot/(1+q)
        m2=q*m1
        return self.m1m2_to_mchirp(m1, m2)
    
    def m1m2_to_mtot_q(self, m1, m2):
        """
        Converts a given binary componenet masses to its total mass and mass ratio.

        Parameters
        ----------
        m1 : float
            Mass of the first component of the binary, in Msun.
        m2 : float
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        total mass : float
            Total mass of the binary.
        mass ratio : float   
            Mass ratio of the binary. 

        """        

        if m1>=m2:
            q=m2/m1
        else:
            q=m1/m2
        return m1+m2, q     
    
    
class CBC_parms_domain:
    """
    Contains functions relevant for fitting factor computation.

    """
    def wrap_reflective(self, x, x1, x2):
        """
        Function to wrap and reflect a real number around the points x1 and x2. 
        Example - For spins, we will have 'wrap_reflective(1.1, -1, 1) = 0.9', 'wrap_reflective(-1.1, -1, 1) = -0.9', and so on.

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
        Example - (i) For spins, we will have 'wrap(2*np.pi + x, 0, 2*np.pi, 'periodic') = x',
        'wrap(1.1, -1, 1, 'reflective') = 0.9', 'wrap(-1.1, -1, 1, 'reflective') = -0.9'.

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
            Allowed keywords for boundary are: {'reflective', 'periodic'}

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
    
        
    
    
class misc_funcs:
    """
    Miscallaneous functions that can make your life simpler and effective.

    """

    def print_dict(self, a_dict):
        """
        Function to print a dictionary element by element.

        """        

        for i in range(len(a_dict)):
            print(list(a_dict.items())[i][0],':',list(a_dict.items())[i][1].__repr__()) 
        
    def distribute(self, array, rank, ncpus):  
        """
        Function to efficiently distribute an array/list among different processors.
        Useful for parallel computations, such as MPI.
        It maintains original order of the array while distributing. 
        Caution: empty arrays can be checked using array.size == 0 before doing any operation to avoid errors.
        
        Example
        -------
        distribute(array=[1,2,3,4,5,6,7], rank=0, ncpus=3) = [1, 2, 3]  # Extra elements get distributed starting from rank=0 to n-1.
        distribute(array=[1,2,3,4,5,6,7], rank=1, ncpus=3) = [4, 5]
        distribute(array=[1,2,3,4,5,6,7], rank=2, ncpus=3) = [6, 7]

        Parameters
        ----------
        array : numpy.array or list
            Array to be distributed.
        rank : int
            Rank of the process. An integer between [0, ncpus-1].
        ncpus : int
            Total number of processors.

        Returns
        -------
        numpy.array
            A segment of the provided array assigned to rank=rank.

        """      

        array = np.array(array)
        array_len = len(array)
        q = int(array_len/ncpus)
        rem = array_len - q*ncpus

        if rank < rem:
            d_array = array[rank*(q+1):(rank+1)*(q+1)]
        elif rem <= rank <= ncpus-1:
            d_array = array[rank*q + rem:(rank+1)*q + rem]
        else:
            d_array = np.array([])
        return d_array           

    def find_nearest(array, value):
        """
        Returns nearest element in an array to a given value.

        Parameters
        ----------
        array : numpy.array/list
        value : value to search for

        Returns
        -------
        int
            index of the nearest element.
        float
            Neearest element found in the array.

        """        

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def is_num(self, x):
        """
        Function to check whether an input is a number or not.
        There is scope of improving the function as it can sometimes give false positives.

        Parameters
        ----------
        x : python object
            Object to check.

        Returns
        -------
        Bool

        """        

        if isinstance(x, np.ndarray):   
            # to take care of cases where a scalar is defined as np.array(num)
            try:
                len(x)
                return False
            except TypeError:
                return True
        else: 
            is_list_type = isinstance(x, (np.ndarray, dict, collections.abc.Sequence)) #[collections.abc.Sequence, np.ndarray, dict, np.nan])
            if is_list_type:
                return False
            elif math.isnan(x):
                return False    
            else:    
                return isinstance(x, (float, int, np.floating, np.complexfloating)) 

    def str_num(self, n, r):
        """
        Function to covert an integer (n) to a string with specified number of digits (r). 
        r < n will be treated as n itself.
        r > n will apply extra 0's in the beginning. 

        Parameters
        ----------
        n : int
            Integer object to convert.
        r : int
            Number of digits in the string. 

        Returns
        -------
        str
            String corresponding to the provided integer.

        """        

        i=1
        while n/10**i>=1:
            i+=1
        return '0'*(r-i)+str(n)

    def str_float(self, n, r):
        """
        Function to covert a float (n) to a string up to `r` decimal places. 

        Parameters
        ----------
        n : float
            Float object to convert.
        r : int
            Number of decimal places to consider while conversion.

        Returns
        -------
        str
            String corresponding to the provided float valid upto r decimal places.

        """        

        dec = round(n%1,r)
        str_dec = self.str_num(int(round(10**r*dec)), r)
        if dec == 1:
            n+=1
            str_dec=str_dec[1:]
        return str(int(n)) + 'p' + str_dec

    def str_y(self, n):
        """
        Returns string version for a number up to two decimal places.

        """

        return self.str_float(n, 2)

    def str_m(self, n):
        """
        Returns string version for a number up to one decimal place.

        """

        return  self.str_float(n, 1)    

    def str_to_float(self, sf, separater='p'):
        """
        Returns the number corresponding to a string assuming decimals are denoted by dot (".").

        """

        # returns the float value corresponding to a number in string with decimals separated by the separator, like '123p89' 
        return float(sf.replace(separater, '.'))

    
class lensed_wf_gen(point_lens, cosmology, GW_analysis, cbc_parameter_conversions, CBC_parms_domain, misc_funcs):
    """
    A Child Class inherited from all of the above classes. 
    Contains functions related to WF generation (with or w/o noise), snr computations, snr_to_distance or vice versa estimates, etc.
    
    # An example dict for this class with default values:
    init_prms = dict(f_low=20., f_high=None, f_ref=20., sample_rate=2048, wf_approximant="IMRPhenomXP", ifo_list = ['H1', 'L1', 'V1'])
    lens_prms = dict(m_lens=0., y_lens=5., z_lens=0., Ff_data=None)
    cbc_prms =  dict(mass_1=m1, mass_2=m2, a_1=a_1, a_2=a_1, tilt_1=tilt_1, tilt_2=tilt_2, 
                phi_jl=phi_jl, phi_12=phi_12, theta_jn=theta_jn, luminosity_distance=dist, 
                ra=0., dec=0., polarization=0., coa_phase=0., trig_time=0.,  
                mode_array=None, lambda1=None, lambda2=None, rwrap = 0.)
    misc_prms = dict(save_data=False, data_outdir = './', data_label='data', data_channel='PyCBC-Injection')            
    psd_prms = dict(Noise=True, psd_H1=psd_H1, psd_L1=psd_L1, psd_V1=psd_V1)   #for no-noise, comment this line or set Noise=False.
    where, if Noise == {True, 'True', 'true'} then noise will be added based on provided files for each detector, 
    i.e., psd_H1 = Path to the file containing PSD information of H1 detector, and so on.

    prms = {**init_prms, **lens_prms, **cbc_prms, **psd_prms, **misc_prms}
    
    Caution: Provide data in either the L-frame or J-frame (explained below). Not both!

    prms : 
    Dictionary of:

        Initial basic prms:

            * f_low : ({20., float}), optional 
                Starting frequency for waveform generation (in Hz). 
            * f_high : ({None., float}), optional 
                Maximum frequency for matched filter computations (in Hz).     
            * f_ref : ({20., float}), optional  
                Reference frequency (in Hz). 
            * sample_rate : ({2048, int}), optional   
                Sample rate of WFs to generate. Default = 2048 (works for most of BBH parameters excpet when binary mass < ~20).
            * approximant : str, optional
                Name of LAL WF approximant to use for WF generation. Default="IMRPhenomXP".
            * ifo_list : list of strings
                List of interferometers to consider. Default =  ['H1', 'L1', 'V1']

        Microlensing Parameters:

            * m_lens : ({0.,float}), optional 
                Point-Lens Mass (in solar masses).
            * y_lens : ({5.,float}), optional 
                The dimensionless impact parameter between the lens and the source.
            * z_lens : ({0.,float}), optional. 
                Lens redshift. Default = 0. (i.e., m_lens will then represent the redhifted lens mass.)
            * Ff_data : float, optional 
                F(f) data to use for modifying WF. If given, {m_lens, y_lens, z_lens} will be ignored. Default=None.

        CBC Parameters:

            * mass_1 : float 
                The mass of the first component object in the binary (in solar masses).
            * mass_2 : float 
                The mass of the second component object in the binary (in solar masses).
            * mode_array : {None, list of lists}
                Modes to consider while generating WF. Default = None, contains all the modes available in the WF approximant. 
                For specific modes, use: mode_array = [[2,2], [4,4], etc.]    
            * lambda1 : float, optional
                The tidal deformability parameter of the first component object in the binary (in solar masses).
            * lambda2 : float, optional
                The tidal deformability parameter of the second component object in the binary (in solar masses).    

        (J-Frame Coordinates):    

            * a_1 : float, optional
                The dimensionless spin magnitude. 
            * a_2 : float, optional
                The dimensionless spin magnitude. 
            * tilt_1 : ({0.,float}), optional
                Angle between L and the spin magnitude of object 1.
            * tilt_2 : float, optional
                Angle between L and the spin magnitude of object 2.
            * phi_12 : float, optional
                Difference between the azimuthal angles of the spin of the object 1 and 2. 
            * phi_jl : float, optional
                Azimuthal angle of L on its cone about J.     
            * theta_jn : float, optional
                Angle between the line of sight and the total angular momentum J.  

        (L-Frame Coordinates):

            * inclination : float
                Inclination (rad), defined as the angle between the orbital angular momentum L and the
                line-of-sight at the reference frequency. Default = 0.
            * spin1x : float
                The x component of the first binary component. Default = 0.
            * spin1y : float
                The y component of the first binary component. Default = 0.
            * spin1z : float
                The z component of the first binary component. Default = 0.
            * spin2x : float
                The x component of the second binary component. Default = 0.
            * spin2y : float
                The y component of the second binary component. Default = 0.
            * spin2z : float
                The z component of the second binary component. Default = 0.

        Other (Extrinsic) Parameters:  

            * luminosity_distance : ({100.,float}), optional
                Luminosity distance to the binary (in Mpc).
            * ra : ({0.,float}), optional
                Right ascension of the source (in rad).
            * dec : ({0.,float}), optional
                Declination of the source (in rad).
            * polarization : ({0.,float}), optional
                Polarisation angle of the source (in rad).
            * coa_phase : ({0.,float}), optional
                Coalesence phase of the binary (in rad).
            * trig_time : ({0.,float}), optional
                Trigger time of the GW event (GPS time).
            * rwrap : ({0.,float}), optional
                Cyclic time shift value (in sec).
            
        Noise Parameters:

            * Noise : str, use either of these {True, 'True', 'true'} for setting up the noise.
                Boolian type value indicating whether to add Noise to the projected signals or not.
            * psd_H1, psd_L1, psd_V1 : str 
                Path to the respective PSD files. Default = 'O4' (PSDs corresponding to O4 are pre-saved and linked to the keyword `O4`.)

        Misc Parameters:
            * save_data : str, use either of these {True, 'True', 'true'} for saving the data.
                A boolian type value to save the detector data as a frame file. Default = False.
            * data_outdir: str
                Output directory where data will be saved. Default = './', i.e., the current directory from where the command will be run.
            * data_label: str
                Data label to use while saving. Default = 'data'.
            * data_channel: str,
                Detector channel name (will be used while reading the data). Default = 'PyCBC_Injection'.
                
    """

    def __init__(self):      
        # These values can be modified by attribute referencing, i.e., after importing via module.class.variable_name = New_Val
        self.f_low = 20.
        self.f_high = None
        self.f_ref = 20.
        self.sample_rate = 2048 
        self.wf_approximant = "IMRPhenomXP"
        self.ifo_list = ['H1', 'L1', 'V1']
    
    def lensed_pure_polarized_wf_gen(self, **prms): 
        """
        Function to generate pure polarised lensed waveform.

        Parameters
        ----------
        prms : Dictionary of parameters as described in the definition of this class. 
        To quickly generate WFs with default settings, use:
        prms = dict(mass_1=m1, mass_2=m2, luminosity_distance=100, theta_jn=0)


        Returns
        -------
        Dictionary of:
        * lensed_FD_WF_hp, lensed_FD_WF_hc : pycbc.types.FrequencySeries
            Lensed pure polarized frequency-domain (FD) waveforms.
        * lensed_TD_WF_hp, lensed_TD_WF_hc : pycbc.types.TimeSeries
            Lensed pure polarized time-domain (TD) waveforms.
        * unlensed_FD_WF_hp, unlensed_FD_WF_hc : pycbc.types.FrequencySeries
            Unlensed pure polarized frequency-domain (FD) waveforms.        
        * unlensed_TD_WF_hp, unlensed_TD_WF_hp : pycbc.types.TimeSeries
            Unlensed pure polarized time-domain (TD) waveforms.
            
        """        

        prms_default = dict(f_low=self.f_low, f_high=self.f_high, f_ref=self.f_ref, sample_rate=self.sample_rate, 
                        wf_approximant=self.wf_approximant, ifo_list = self.ifo_list,
                        m_lens=0., y_lens=5., z_lens=0., Ff_data=None, 
                        ra=0., dec=0., polarization=0., coa_phase=0., trig_time=0., 
                        mode_array=None, lambda1=None, lambda2=None, rwrap = 0.
                        )
        prms_default.update(prms)
        prms = prms_default.copy()
        
        # checks if a key exists in the provided dictionary "prms"
        prms_key_bool = lambda x: bool(x in prms.keys())

        # checks if a spin key is provided, otherwise assigns 0.
        lframe_related_keys = np.array(['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'inclination'])
        jframe_related_keys = np.array(['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'theta_jn'])

        key_bool_vals_lframe = list(map(prms_key_bool, lframe_related_keys))
        key_bool_vals_jframe = list(map(prms_key_bool, jframe_related_keys))

        if np.sum(key_bool_vals_lframe) > 0 and np.sum(key_bool_vals_jframe) == 0:
            spin_related_keys = lframe_related_keys
            key_bool_vals = key_bool_vals_lframe
            provided_spin_keys = spin_related_keys[key_bool_vals]
            not_provided_spin_keys = spin_related_keys[[not bool_vals for bool_vals in key_bool_vals]]
            for key in not_provided_spin_keys:
                prms[key] = 0
        elif np.sum(key_bool_vals_lframe) == 0 and np.sum(key_bool_vals_jframe) > 0:
            spin_related_keys = jframe_related_keys
            key_bool_vals = key_bool_vals_jframe
            provided_spin_keys = spin_related_keys[key_bool_vals]
            not_provided_spin_keys = spin_related_keys[[not bool_vals for bool_vals in key_bool_vals]]
            for key in not_provided_spin_keys:
                prms[key] = 0    
            lframe_dict = self.jframe_to_l0frame(mass_1=prms['mass_1'], mass_2=prms['mass_2'], 
                                           f_ref=prms['f_ref'], phi_ref=prms['coa_phase'], theta_jn=prms['theta_jn'],\
                                           phi_jl=prms['phi_jl'], a_1=prms['a_1'], a_2=prms['a_2'], tilt_1=prms['tilt_1'], tilt_2=prms['tilt_2'], phi_12=prms['phi_12'])  
           
            prms = {**prms, **lframe_dict}  #prms.update(lframe_dict) #{**prms, **lframe_dict}
            
        elif np.sum(key_bool_vals_lframe) + np.sum(key_bool_vals_jframe) == 0:
            raise Exception('Please provide parameters related to either L-Frame or J-Frame.')
        
        else:
            raise Exception('Provide parameter values in either the L-Frame or the J-Frame, NOT both!')
        
        #Waveform Generation 
        dt = 1./prms['sample_rate']
        hp, hc = get_td_waveform(
            approximant = prms['wf_approximant'],
            mass1 = prms['mass_1'],
            mass2 = prms['mass_2'],
            spin1x = prms['spin1x'],
            spin1y = prms['spin1y'],
            spin1z = prms['spin1z'],
            spin2x = prms['spin2x'],
            spin2y = prms['spin2y'],
            spin2z = prms['spin2z'],
            lambda1 = prms['lambda1'],
            lambda2 = prms['lambda2'],
            distance =  prms['luminosity_distance'],
            inclination = prms['inclination'],
            coa_phase = prms['coa_phase'],
            mode_array = prms['mode_array'],
            f_lower = prms['f_low'],
            f_ref = prms['f_ref'],
            delta_t = dt)

        tlen = len(hp)
        d_f = hp.delta_f
        flen = tlen//2 + 1

        # performing cyclic time shift to avoid wrapping up of WFs
        hp = self.cyclic_time_shift_of_WF(hp, prms["rwrap"])  #hp.cyclic_time_shift(rwrap)
        hc = self.cyclic_time_shift_of_WF(hc, prms["rwrap"])  #hc.cyclic_time_shift(rwrap)

        #Modifying WF
        fdwf_hp = hp.to_frequencyseries(delta_f=hp.delta_f)
        fdwf_hc = hc.to_frequencyseries(delta_f=hc.delta_f)   

        # Checks if the amplification factor, F(f), data is provided to modify the WF.
        # The data should be a three columned array containing {frequencies, Re[F(f)], Im[F(f)]}
        if prms['Ff_data'] is not None:   
            Ffs = prms["Ff_data"]
            if np.array(Ffs).size > 1 and np.array(Ffs[0]).size == 3:
                iFf = interp1d(Ffs[:,0], Ffs[:,1] + 1j*Ffs[:,2], kind='linear')
                fs = fdwf_hp.sample_frequencies
                Ff = np.array(list(map(iFf, fs)))
            else:
                raise Exception("Please provide 'Ff_data' in a correct format. The data should be a three columned array containing {frequencies, Re[F(f)], Im[F(f)]}. "\
                                "If you didn't intend using 'Ff_data' but rather point lens parameters, you can set it to 'None'.")
        else:
            m_lens, y_lens, z_lens = prms['m_lens'], prms['y_lens'], prms['z_lens']
            if round(m_lens) == 0:
                res_ul = {'unlensed_FD_WF_hp':fdwf_hp, 'unlensed_FD_WF_hc':fdwf_hc, 'unlensed_TD_WF_hp':hp, 'unlensed_TD_WF_hc':hc}
                res_ml = {'lensed_FD_WF_hp':fdwf_hp, 'lensed_FD_WF_hc':fdwf_hc, 'lensed_TD_WF_hp':hp, 'lensed_TD_WF_hc':hc}
                return {**res_ul, **res_ml} 
            else:
                fs = fdwf_hp.sample_frequencies
                wfs = self.w_of_f(fs, m_lens, z_lens)
                Ff = self.point_Fw_eff_map(wfs, y_lens)   

        lfd_hp = Ff*fdwf_hp
        lfdwf_hp = types.FrequencySeries(lfd_hp, delta_f = hp.delta_f)
        lwf_hp = lfdwf_hp.to_timeseries(delta_t=hp.delta_t)

        fdwf_hc = hc.to_frequencyseries(delta_f=hc.delta_f)
        lfd_hc = Ff*fdwf_hc
        lfdwf_hc = types.FrequencySeries(lfd_hc, delta_f = hc.delta_f)
        lwf_hc = lfdwf_hc.to_timeseries(delta_t=hc.delta_t)

        res_ul = {'unlensed_FD_WF_hp':fdwf_hp, 'unlensed_FD_WF_hc':fdwf_hc, 'unlensed_TD_WF_hp':hp, 'unlensed_TD_WF_hc':hc}
        res_ml = {'lensed_FD_WF_hp':lfdwf_hp, 'lensed_FD_WF_hc':lfdwf_hc, 'lensed_TD_WF_hp':lwf_hp, 'lensed_TD_WF_hc':lwf_hc}
        return {**res_ul, **res_ml} 
     
    def wf_len_mod_start(self, wf, extra=1, **prms):
        """
        Function to modify the starting of a WF so that it starts on an integer GPS time (in sec) + add extra length as specified by the user.

        Parameters
        ----------
        wf :  pycbc.types.TimeSeries
            WF whose length is to be modified.
        extra : int, optional
            Extra length to be added in the beginning after making the WF to start from an integer GPS time (in sec). Default = 1.

        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Modified waveform starting form an integer time.

        """      

        sr = prms['sample_rate']
        olen = len(wf)   
        diff = wf.sample_times[0]-np.floor(wf.sample_times[0])  
        #nlen = round(olen+sr*(extra+diff))
        dlen = round(sr*(extra+diff))
        wf_strain = np.concatenate((np.zeros(dlen), wf))
        t0 = wf.sample_times[0]
        dt = wf.delta_t
        n = dlen
        tnn = t0-(n+1)*dt
        wf_stime = np.concatenate((np.arange(t0-dt,tnn,-dt)[::-1], np.array(wf.sample_times)))
        nwf = pycbc.types.TimeSeries(wf_strain, delta_t=wf.delta_t, epoch=wf_stime[0])
        return nwf

    def wf_len_mod_end(self, wf, extra=2, **prms): #post_trig_duration
        """
        Function to modify the end of a WF so that it ends on an integer GPS time (in sec) + add extra length as specified by the user.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            WF whose length is to be modified.
        extra : int, optional
            Extra length to be added towards the end after making the WF to end from an integer GPS time (in sec). 
            Default = 2, which makes sure post-trigger duration is of at least 2 seconds.

        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Modified waveform ending on an integer time.

        """        

        sr = prms['sample_rate']
        olen = len(wf)   
        dt = abs(wf.sample_times[-1] - wf.sample_times[-2])
        diff = np.ceil(wf.sample_times[-1]) - (wf.sample_times[-1] + dt)   #wf.sample_times[-1]-int(wf.sample_times[-1])  
        nlen = round(olen + sr*(extra+diff))
        wf.resize(nlen)
        return wf    
    
    def make_len_power_of_2(self, wf):
        """
        Function to modify the length of a waveform so that its duration is a power of 2.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            WF whose length is to be modified.
            Modified waveform with duration a power of 2.
        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Returns the waveform with length a power of 2.

        """    

        dur = wf.duration  
        wf.resize( int(round(wf.sample_rate * np.power(2, np.ceil( np.log2( dur ) ) ))) )
        wf = self.cyclic_time_shift_of_WF(wf, rwrap = wf.duration - dur )
        return wf
        
    def sim_lensed_wf_gen(self, **prms):
        """
        Simulated lensed WF generation projected onto detectors.

        Parameters
        ----------
        prms : Dictionary of parameters as described in the definition of this class. 
        To quickly generate WFs with default settings, use:
        prms = dict(mass_1=m1, mass_2=m2, luminosity_distance=100, theta_jn=0)

        Returns
        -------
        Dictionary of :
        * pure_polarized_wfs : dict 
            A dictionary containing plus and cross polarizaions of WF. keys = ['hp', 'hc'].
        * pure_ifo_signal : dict
            A dictionary containing projected WFs onto detectors without noise. keys = ifo_list.

        """   

        prms_default = dict(f_low=self.f_low, f_high=self.f_high, f_ref=self.f_ref, sample_rate=self.sample_rate, 
                        wf_approximant=self.wf_approximant, ifo_list=self.ifo_list, ra=0., dec=0., polarization=0., trig_time=0.,
                        extra_padding_at_start=1, extra_padding_at_end=1)
        prms_default.update(prms)
        prms = prms_default.copy()

        # Choose a GPS end time, sky location, and polarization phase for the merger
        # NOTE: Right ascension and polarization phase runs from 0 to 2pi
        #       Declination runs from pi/2 to -pi/2 with the poles at pi/2 and -pi/2.

        end_time = prms['trig_time']
        res_data = self.lensed_pure_polarized_wf_gen(**prms)
        hp = res_data['lensed_TD_WF_hp']
        hc = res_data['lensed_TD_WF_hc']
        hp.start_time += end_time
        hc.start_time += end_time

        # projection onto detectors
        det = dict()
        ifo_signal = dict()
        for ifo in prms['ifo_list']:
            det[ifo] = Detector(ifo)
            ifo_signal[ifo] = det[ifo].project_wave(hp, hc, prms['ra'], prms['dec'], prms['polarization'])
            ifo_signal[ifo] = waveform.utils.taper_timeseries(ifo_signal[ifo], tapermethod='TAPER_STARTEND', return_lal=False)  #remove edge effects

        # We modify the length of the WF so that its time starts and ends in integer seconds. 
        # For this, we first make ends integer then add some extra seconds towards the end and the start.
        for ifo in prms['ifo_list']:
            wf = deepcopy( ifo_signal[ifo] )
            wf = self.wf_len_mod_start(wf, extra=prms['extra_padding_at_start'], **prms) 
            wf = self.wf_len_mod_end(wf, extra=prms['extra_padding_at_end'], **prms) # extra=1 => post-trigger duration is 2 seconds.
            wf = self.make_len_power_of_2(wf)  # making total segment lenght a power of 2 by adding zeros towards the start of the WF.
            ifo_signal[ifo] = wf
        res = dict(pure_polarized_wfs={'hp':hp, 'hc':hc}, pure_ifo_signal=ifo_signal )    
        return res
    
    # def psd_aLIGO(self, duration):
    #     # The color of the noise matches a PSD which you provide
    #     f_low = prms['f_low']
    #     sr = prms['sample_rate']
    #     delta_f = 1.0 / duration
    #     flen = round(np.floor(sr / delta_f) + 1)
    #     psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)
    #     psd = psd_remove_zero(psd)
    #     return psd

    def psd_remove_zero(self, psd):
        """
        Function to replace zeros with rather small values in a given psd.

        """

        for i in range(len(psd)):
            if psd[i]==0:
                psd[i]=1e-52 
        return psd  
    

        #make psd_dict: duration, files, default=O4 psds
    def psd_gen(self, psd_file, psd_sample_rate = None, psd_duration = 32, **prms):
        """
        A tailored function to generate PSD.

        """     

        # The PSD will be interpolated to the requested frequency spacing    
        delta_f = 1.0 / psd_duration
        length = int(psd_sample_rate / delta_f)
        low_frequency_cutoff = prms['f_low']
        psd = pycbc.psd.from_txt(psd_file, length, delta_f, low_frequency_cutoff, is_asd_file=False)
        psd = self.psd_remove_zero(psd)
        return psd
    
    
    # Adds noise to a WF
    def add_noise(self, wf, psd, noise_seed=127):  # takes a pure timeseries WF as input
        """
        Function to add noise to a given WF based on provided PSD and a realization seed. 

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            _description_
        psd : _type_
            pycbc.types.timeseries.TimeSeries
        noise_seed : int, optional
            Seed value to generate noise realisation. Default = 127.

        Returns
        -------
        pycbc.types.TimeSeries
            WF with added noise.
            
        """        

        #remove edge effects
        wf = waveform.utils.taper_timeseries(wf, tapermethod='TAPER_STARTEND', return_lal=False)   
        #generate a noise timeseries with duration equal to that of template
        delta_t = wf.delta_t
        t_samples = len(wf) 
        ts = pycbc.noise.noise_from_psd(t_samples, delta_t, psd, seed=noise_seed)  
        noisy_sig = types.TimeSeries(np.array(wf)+np.array(ts), delta_t=delta_t, epoch=wf.start_time) #adding noise to the pure wf
        return noisy_sig
    
    
    # simulated lensed noisy signal
    def sim_lensed_noisy_wf_gen(self, **prms):
        """
        Simulated lensed WF generation projected onto detectors with added noise.

        Parameters
        ----------
        prms : Dictionary of parameters as described in the definition of this class. 
        To quickly generate WFs with added Noise and default settings, use:
        cbc_prms = dict(mass_1=m1, mass_2=m2, luminosity_distance=100, theta_jn=0)
        psd_prms = dict(Noise=True, psd_H1=psd_H1, psd_L1=psd_L1, psd_V1=psd_V1)   #for no-noise, comment this line or set Noise=False.
        where, if Noise = {True, 'True', 'true'} then noise will be added based on the provided files for each detector, 
        i.e., psd_H1 = Path to the file containing PSD information of H1 detector, and so on.
        prms = {**cbc_prms, **psd_prms}


        Returns
        -------
        Dictionary of :
        * pure_polarized_wfs : dict 
            A dictionary containing plus and cross polarizaions of WF. Keys = ['hp', 'hc'].
        * pure_ifo_signal : dict
            A dictionary containing projected WFs onto detectors without noise. Keys = ifo_list
        * noisy_ifo_signal : dict
            A dictionary containing projected WFs onto detectors with added noise. Keys = ifo_list
        * psd : dict
            A dictionary containing generated PSDs in each detector. Keys = ifo_list

        """ 

        # n_sig_h1, n_sig_l1, n_sig_v1 : pycbc.types.timeseries.TimeSeries  objects
        #     Injected signal on three detectors.
        # signal_h1, signal_l1, signal_v1 : pycbc.types.TimeSeries
        #     Projected WFs onto H1, L1, V1, respectively. 
        # psd_h1, psd_l1, psd_v1: pycbc.types.TimeSeries
        #      Generated PSD data of three detectors.
        # hp, hc : pycbc.types.TimeSeries
        #     Unlensed pure polarized time-domain (TD) waveforms.


        prms_default = dict(f_low=self.f_low, f_high=self.f_high, f_ref=self.f_ref, sample_rate=self.sample_rate, 
                        wf_approximant=self.wf_approximant, ifo_list = self.ifo_list,
                        Noise=False, psd_H1=None, psd_L1=None, psd_V1=None, gen_seed=127,   # just a default random value
                        ra=0., dec=0., polarization=0., coa_phase=0., trig_time=0.,
                        save_data=False, data_outdir = './', data_label= 'signal_data', data_channel='PyCBC_Injection')
        prms_default.update(prms)  
        prms = prms_default.copy()
        
        wfs_res = self.sim_lensed_wf_gen(**prms)
        pure_ifo_signal = wfs_res['pure_ifo_signal']
        
        
        
        #psd_h1 = self.psd_gen(psd_file = filename_h1, psd_sample_rate=1./signal_h1.delta_t, psd_duration=signal_h1.duration, **prms)
        #psd_l1 = self.psd_gen(psd_file = filename_l1, psd_sample_rate=1./signal_l1.delta_t, psd_duration=signal_l1.duration, **prms)
        #psd_v1 = self.psd_gen(psd_file = filename_v1, psd_sample_rate=1./signal_v1.delta_t, psd_duration=signal_v1.duration, **prms)
       
        #df_psd_h1 = np.loadtxt(filename_h1)[:,0][1]-np.loadtxt(filename_h1)[:,0][0]
        #df_psd_l1 = np.loadtxt(filename_l1)[:,0][1]-np.loadtxt(filename_l1)[:,0][0]
        #df_psd_v1 = np.loadtxt(filename_v1)[:,0][1]-np.loadtxt(filename_v1)[:,0][0]
        #psd_h1 = from_txt(filename = filename_h1, delta_f=df_psd_h1, length=len(signal_h1), low_freq_cutoff=prms['f_low'])
        #psd_l1 = from_txt(filename = filename_l1, delta_f=df_psd_l1, length=len(signal_l1), low_freq_cutoff=prms['f_low'])
        #psd_v1 = from_txt(filename = filename_v1, delta_f=df_psd_v1, length=len(signal_v1), low_freq_cutoff=prms['f_low'])


        if prms['psd_H1']=='default' or prms['psd_H1']=='O4':
            prms['psd_H1'] = cwd + '../data/PSDs/O4_target_psds/psd_aLIGO_O4high.txt'
   
        if prms['psd_L1']=='default' or prms['psd_L1']=='O4':
            prms['psd_L1'] = cwd + '../data/PSDs/O4_target_psds/psd_aLIGO_O4high.txt'

        if prms['psd_V1']=='default' or prms['psd_V1']=='O4':
            prms['psd_V1'] = cwd + '../data/PSDs/O4_target_psds/psd_aVirgo_O4high_NEW.txt' 

        psd_file_dict = dict(H1=prms['psd_H1'], L1=prms['psd_L1'], V1=prms['psd_V1'])
        psd = dict()
        for ifo in prms['ifo_list']:
            if psd_file_dict[ifo] != None:
                psd[ifo] = self.psd_gen(psd_file = psd_file_dict[ifo], psd_sample_rate=1./pure_ifo_signal[ifo].delta_t, 
                psd_duration=pure_ifo_signal[ifo].duration, **prms)
            else:
                psd[ifo] = None
                
        if prms['Noise']==True or prms['Noise']=='True' or prms['Noise']=='true':
            noisy_ifo_signal = dict()   
            for ifo in prms['ifo_list']:
                n_sig = self.add_noise(pure_ifo_signal[ifo], psd[ifo], noise_seed=prms['gen_seed'])  
                noisy_ifo_signal[ifo] = deepcopy(n_sig)  
        else:
            noisy_ifo_signal = pure_ifo_signal.copy()

        wfs_res.update(noisy_ifo_signal=noisy_ifo_signal, psd=psd)

        if prms['save_data']==True or prms['save_data']=='True' or prms['save_data']=='true':
            for ifo in prms['ifo_list']:
                print('Saving Data : %s'%ifo) 
                noisy_sig = wfs_res['noisy_ifo_signal'][ifo]
                data = pycbc.types.TimeSeries(np.array(noisy_sig), delta_t=1/noisy_sig.sample_rate, epoch=round(noisy_sig.sample_times[0]))
                frame.write_frame(prms['data_outdir'] + prms['data_label'] + '_' + ifo +'.gwf', ifo + ":" + prms['data_channel'], data) 
        return wfs_res

    # # simulated lensed noisy signal
    # def sim_lensed_noisy_wf_gen(self, **prms):
    #     """
    #     Simulated lensed WF generation projected onto detectors with added noise.

    #     Parameters
    #     ----------
    #     prms : Dictionary of parameters as described in the definition of this class. 
    #     To quickly generate WFs with added Noise and default settings, use:
    #     cbc_prms = dict(mass_1=m1, mass_2=m2, luminosity_distance=100, theta_jn=0)
    #     psd_prms = dict(Noise=True, psd_H1=psd_H1, psd_L1=psd_L1, psd_V1=psd_V1)   #for no-noise, comment this line or set Noise=False.
    #     where, if Noise = {True, 'True', 'true'} then noise will be added based on the provided files for each detector, 
    #     i.e., psd_H1 = Path to the file containing PSD information of H1 detector, and so on.
    #     prms = {**cbc_prms, **psd_prms}


    #     Returns
    #     -------
    #     Dictionary of :
    #     * pure_polarized_wfs : dict 
    #         A dictionary containing plus and cross polarizaions of WF. Keys = ['hp', 'hc'].
    #     * pure_ifo_signal : dict
    #         A dictionary containing projected WFs onto detectors without noise. Keys = ifo_list
    #     * noisy_ifo_signal : dict
    #         A dictionary containing projected WFs onto detectors with added noise. Keys = ifo_list
    #     * psd : dict
    #         A dictionary containing generated PSDs in each detector. Keys = ifo_list

    #     """ 

    #     # n_sig_h1, n_sig_l1, n_sig_v1 : pycbc.types.timeseries.TimeSeries  objects
    #     #     Injected signal on three detectors.
    #     # signal_h1, signal_l1, signal_v1 : pycbc.types.TimeSeries
    #     #     Projected WFs onto H1, L1, V1, respectively. 
    #     # psd_h1, psd_l1, psd_v1: pycbc.types.TimeSeries
    #     #      Generated PSD data of three detectors.
    #     # hp, hc : pycbc.types.TimeSeries
    #     #     Unlensed pure polarized time-domain (TD) waveforms.


    #     prms_default = dict(f_low=self.f_low, f_high=self.f_high, f_ref=self.f_ref, sample_rate=self.sample_rate, 
    #                     wf_approximant=self.wf_approximant, ifo_list = self.ifo_list,
    #                     Noise=False, psd_H1=None, psd_L1=None, psd_V1=None, gen_seed=127,   # just a default random value
    #                     ra=0., dec=0., polarization=0., coa_phase=0., trig_time=0.,
    #                     save_data=False, data_outdir = './', data_label= 'signal_data', data_channel='PyCBC_Injection')
    #     prms_default.update(prms)  
    #     prms = prms_default.copy()
        
    #     wfs_res = self.sim_lensed_wf_gen(**prms)
    #     pure_ifo_signal = wfs_res['pure_ifo_signal']
        
        
        
    #     #psd_h1 = self.psd_gen(psd_file = filename_h1, psd_sample_rate=1./signal_h1.delta_t, psd_duration=signal_h1.duration, **prms)
    #     #psd_l1 = self.psd_gen(psd_file = filename_l1, psd_sample_rate=1./signal_l1.delta_t, psd_duration=signal_l1.duration, **prms)
    #     #psd_v1 = self.psd_gen(psd_file = filename_v1, psd_sample_rate=1./signal_v1.delta_t, psd_duration=signal_v1.duration, **prms)
       
    #     #df_psd_h1 = np.loadtxt(filename_h1)[:,0][1]-np.loadtxt(filename_h1)[:,0][0]
    #     #df_psd_l1 = np.loadtxt(filename_l1)[:,0][1]-np.loadtxt(filename_l1)[:,0][0]
    #     #df_psd_v1 = np.loadtxt(filename_v1)[:,0][1]-np.loadtxt(filename_v1)[:,0][0]
    #     #psd_h1 = from_txt(filename = filename_h1, delta_f=df_psd_h1, length=len(signal_h1), low_freq_cutoff=prms['f_low'])
    #     #psd_l1 = from_txt(filename = filename_l1, delta_f=df_psd_l1, length=len(signal_l1), low_freq_cutoff=prms['f_low'])
    #     #psd_v1 = from_txt(filename = filename_v1, delta_f=df_psd_v1, length=len(signal_v1), low_freq_cutoff=prms['f_low'])


    #     if prms['Noise']==True or prms['Noise']=='True' or prms['Noise']=='true':
    #         if prms['psd_H1']=='default' or prms['psd_H1']=='O4':
    #             filename_h1 = '../data/PSDs/O4_target_psds/psd_aLIGO_O4high.txt'
    #             # filename_h1 = '/home/anuj/git_repos/git_GWs/data/O4_target_psds/psd_aLIGO_O4high.txt'
    #             # filename_h1 = '/mnt/home/student/canujm/pyGW/data_gen/O4_target_psds/psd_aLIGO_O4high.txt'
    #         else:
    #             filename_h1 = prms['psd_H1']

    #         if prms['psd_L1']=='default' or prms['psd_L1']=='O4':
    #             filename_l1 = '../data/PSDs/O4_target_psds/psd_aLIGO_O4high.txt'
    #             # filename_l1 = '/home/anuj/git_repos/git_GWs/data/O4_target_psds/psd_aLIGO_O4high.txt'
    #             # filename_l1 = '/mnt/home/student/canujm/pyGW/data_gen/O4_target_psds/psd_aLIGO_O4high.txt'
    #         else:
    #             filename_l1 = prms['psd_L1']

    #         if prms['psd_V1']=='default' or prms['psd_V1']=='O4':
    #             filename_v1 = '../data/PSDs/O4_target_psds/psd_aVirgo_O4high_NEW.txt' 
    #             # filename_v1 = '/home/anuj/git_repos/git_GWs/data/O4_target_psds/psd_aVirgo_O4high_NEW.txt'
    #             # filename_v1 = '/mnt/home/student/canujm/pyGW/data_gen/O4_target_psds/psd_aVirgo_O4high_NEW.txt'
    #         else:
    #             filename_v1 = prms['psd_V1']
    #         psd_file_dict = dict(H1=filename_h1, L1=filename_l1, V1=filename_v1)
    #         psd = dict()
    #         for ifo in prms['ifo_list']:
    #             psd[ifo] = self.psd_gen(psd_file = psd_file_dict[ifo], psd_sample_rate=1./pure_ifo_signal[ifo].delta_t, 
    #             psd_duration=pure_ifo_signal[ifo].duration, **prms)
    #         noisy_ifo_signal = dict()   
    #         for ifo in prms['ifo_list']:
    #             n_sig = self.add_noise(pure_ifo_signal[ifo], psd[ifo], noise_seed=prms['gen_seed'])  
    #             noisy_ifo_signal[ifo] = deepcopy(n_sig)  
    #     else:
    #         noisy_ifo_signal = pure_ifo_signal.copy()
    #         psd = dict()
    #         for ifo in prms['ifo_list']:
    #             psd[ifo] = None

    #     wfs_res.update(noisy_ifo_signal=noisy_ifo_signal, psd=psd)

    #     if prms['save_data']==True or prms['save_data']=='True' or prms['save_data']=='true':
    #         for ifo in prms['ifo_list']:
    #             print('Saving Data : %s'%ifo) 
    #             noisy_sig = wfs_res['noisy_ifo_signal'][ifo]
    #             data = pycbc.types.TimeSeries(np.array(noisy_sig), delta_t=1/noisy_sig.sample_rate, epoch=round(noisy_sig.sample_times[0]))
    #             frame.write_frame(prms['data_outdir'] + prms['data_label'] + '_' + ifo +'.gwf', ifo + ":" + prms['data_channel'], data) 
    #     return wfs_res
    

    def sim_lensed_noisy_wf_gen_with_snr(self, **prms):
        """
        Simulated lensed WF generation projected onto detectors with added noise. Also returns SNR.
        # Computing SNRs:
        # In general, since s(t) = h(t) + n(t),
        # matched_filter_SNR = (s|h_T) = (s|h)/sqrt(h|h), where h_T is normalized template of a waveform h.
        # optimal_matched_filter_SNR = (h|h_T) = (h|h)/sqrt(h|h)
        # Since we assume noise is gaussian, this MF_SNR is usually a gaussian with mean at the optimal SNR (because of the term (n|h) ).
        # Optimal SNR becomes important in cases of "very low SNRs" when (n|h) dominates over (h|h). In that case, 
        # (n|h) dominates over (h|h) significantly thus biasing the MF_value. 
        # However, optimal SNR still returns the correct value as it is only weighted by the psd rather than contianing the inner product with it (unlike MF).

        Parameters
        ----------
        prms : Dictionary of parameters as described in the definition of this class. 
        To quickly generate WFs with added Noise and default settings, use:
        cbc_prms = dict(mass_1=m1, mass_2=m2, luminosity_distance=100, theta_jn=0)
        psd_prms = dict(Noise=True, psd_H1=psd_H1, psd_L1=psd_L1, psd_V1=psd_V1)   #for no-noise, comment this line or set Noise=False.
        where, if Noise = {True, 'True', 'true'} then noise will be added based on the provided files for each detector, 
        i.e., psd_H1 = Path to the file containing PSD information of H1 detector, and so on.
        prms = {**cbc_prms, **psd_prms}

        Returns
        -------
        dict :
            Dictionary of :
            * pure_polarized_wfs : dict 
                A dictionary containing plus and cross polarizaions of WF. Keys = ['hp', 'hc'].
            * pure_ifo_signal : dict
                A dictionary containing projected WF onto detector(s) without noise. Keys = ifo_list.
            * noisy_ifo_signal : dict
                A dictionary containing projected WF onto detector(s) with added noise. Keys = ifo_list.
            * psd : dict
                A dictionary containing generated PSD(s) in each detector. Keys = ifo_list.
            * signal_templates : dict   
                A dictionary containing template(s) corresponding to the projected signal(s), i.e., signals with peak at one end of the WF (using cyclic shifting).  
                The peak of the template WF needs to be at its edge (start/end) to recover the correct trigger time of the event.  Keys = ifo_list.   
            * optimal_snr : dict 
                A dictionary containing the Optimal SNR of the injected signal(s).  
            * match_filter_snr : dict
                A dictionary containing the Matched-filter SNR of the injected signal(s).  
            * network_optimal_snr : float
                Network optimal SNR.
            * network_matched_filter_snr : float
                Network Matched Filter SNR.                

        """ 

        prms_default = dict(f_low=self.f_low, f_high=self.f_high, f_ref=self.f_ref, sample_rate=self.sample_rate, 
                        wf_approximant=self.wf_approximant, ifo_list = self.ifo_list,
                        ra=0., dec=0., polarization=0., coa_phase=0., trig_time=0.)
        prms_default.update(prms)
        prms = prms_default.copy()

        wfs_res = self.sim_lensed_noisy_wf_gen(**prms)

        signal_templates = dict()
        match_filter_snr_timeseries = dict()
        match_filter_snr = dict()
        optimal_snr = dict()
        for ifo in prms['ifo_list']:
            signal = deepcopy( wfs_res['pure_ifo_signal'][ifo] )
            dt_end = signal.sample_times[-1] - prms['trig_time'] + (signal.sample_times[1] - signal.sample_times[0]) 
            signal_templates[ifo] = self.cyclic_time_shift_of_WF(signal, rwrap=dt_end)
            template, pure_data, noisy_data, psd = signal_templates[ifo], wfs_res['pure_ifo_signal'][ifo], wfs_res['noisy_ifo_signal'][ifo], wfs_res['psd'][ifo]
            mf_snr_ts = pycbc.filter.matchedfilter.matched_filter(template, noisy_data, psd=psd, low_frequency_cutoff=prms['f_low'], high_frequency_cutoff=prms['f_high'])
            match_filter_snr_val = max(np.abs(mf_snr_ts))
            opt_snr_ts = pycbc.filter.matchedfilter.matched_filter(template, pure_data, psd=psd, low_frequency_cutoff=prms['f_low'], high_frequency_cutoff=prms['f_high'])
            opt_snr_val = max(np.abs(opt_snr_ts))
            match_filter_snr_timeseries[ifo] = mf_snr_ts
            match_filter_snr[ifo] = match_filter_snr_val
            optimal_snr[ifo] = opt_snr_val
            
        network_optimal_snr = np.linalg.norm( np.array(list(optimal_snr.items()), dtype=object)[:,1] )
        network_matched_filter_snr =  np.linalg.norm( np.array(list(match_filter_snr.items()), dtype=object)[:,1] )

        wfs_res.update({'signal_templates': signal_templates, 'match_filter_snr': match_filter_snr, 'optimal_snr':optimal_snr})
        wfs_res.update({'network_optimal_snr': network_optimal_snr, 'match_filter_snr_timeseries':match_filter_snr_timeseries,
        'network_matched_filter_snr': network_matched_filter_snr})
        return wfs_res
    
    
    
    def network_optimal_snr_to_distance(self, net_optimal_snr, **prms):
        """
        Converts a given net_opt_snr value to effective distance for a given set of binary and lens prms.

        Parameters
        ----------
        net_optimal_snr : float
            Required Network optimal SNR of the signal.
        prms : dict
            Dictionary of parameters to generate WFs as described in the definition of this class.

        Returns
        -------
        float :
            Distance corresponding to the provided Network optimal SNR.

        """       

        tmp_dist = 100  # assigning a fiducial value
        prms['luminosity_distance'] = tmp_dist
        wfs_res = self.sim_lensed_noisy_wf_gen_with_snr(**prms)
        req_dist = tmp_dist*wfs_res['network_optimal_snr'] / net_optimal_snr
        return req_dist

        # if prms.get('luminosity_distance') == None:
        #     tmp_dist = prms['luminosity_distance'] = 100
        # else:
        #     tmp_dist = prms['luminosity_distance']

        # wf_res = self.sim_lensed_noisy_wf_gen_with_snr(**prms)
        # opt_snr_H1 = wf_res['optimal_snr_H1']
        # opt_snr_L1 = wf_res['optimal_snr_L1']
        # opt_snr_V1 = wf_res['optimal_snr_V1']
        # tmp_net_opt_snr = np.linalg.norm([opt_snr_H1, opt_snr_L1, opt_snr_V1])
        # dist = tmp_dist*tmp_net_opt_snr/net_optimal_snr
        # return dist


    # converts a given net_matched_filter_snr value to effective distance for a given set of binary and lens prms.
    def network_matched_filter_snr_to_distance(self, net_mf_snr, thresh=10, **prms):
        """
        Converts a given net_opt_snr value to effective distance for a given set of binary and lens prms.

        Parameters
        ----------
        net_mf_snr : float
            Required Network matched-filter SNR of the signal.
        thresh : float
            Threshold on the relative error to reach before stop searching (in %). Default = 10.
        prms : dict
            Dictionary of parameters to generate WFs as described in the definition of this class.

        Returns
        -------
        float :
            Distance corresponding to the provided Network matched-filter SNR.

        """ 

        dist_opt = self.network_optimal_snr_to_distance(net_mf_snr, **prms)
        prms['luminosity_distance'] = dist_opt
        wfs_res = self.sim_lensed_noisy_wf_gen_with_snr(**prms)
        tmp_net_mf_snr = wfs_res['network_matched_filter_snr'] 

        # binary search around dist_opt
        dist_min, dist_max = dist_opt*np.array([0.5, 1.5])  
        dist_avg = dist_opt
        k = 0
        rel_err = np.abs( (tmp_net_mf_snr - net_mf_snr) / net_mf_snr)
        while( (rel_err > thresh/100)):
            k+=1
            print('iter: ', k)
            rel_err = np.abs( (tmp_net_mf_snr - net_mf_snr) / net_mf_snr)
            print('relativr_error: ', rel_err)
            tmp_net_mf_snr_0 = tmp_net_mf_snr
            if tmp_net_mf_snr > net_mf_snr:
                dist_min = dist_avg
                dist_avg = (dist_min + dist_max)/2
                prms['luminosity_distance'] = dist_avg
                wfs_res = self.sim_lensed_noisy_wf_gen_with_snr(**prms)
                tmp_net_mf_snr = wfs_res['network_matched_filter_snr'] 
            else:
                dist_max = dist_avg
                dist_avg = (dist_min + dist_max)/2
                prms['luminosity_distance'] = dist_avg
                wfs_res = self.sim_lensed_noisy_wf_gen_with_snr(**prms)
                tmp_net_mf_snr = wfs_res['network_matched_filter_snr']     
            del_mfr = np.abs(tmp_net_mf_snr_0 - tmp_net_mf_snr) 
            if del_mfr < 1e-4:
                print('Warning: Required matched filter value {:.4f} is lower than the minimum value {:.4f} possible for given f_low. \
                \nReturning distance correpsonding to the optimal SNR instead.'.format(net_mf_snr, tmp_net_mf_snr))
                return dist_opt   
        return dist_avg
    
