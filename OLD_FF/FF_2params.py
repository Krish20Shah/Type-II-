#!/home/krish.shah/anaconda3/envs/Teobresums/bin/python3

import numpy as np
import scipy
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt


import sys
#sys.path.append('../src/')
sys.path.append('/home/krish.shah/GW_Lensing/FF_Computation/')
import py_lgw
lgw = py_lgw.lensed_wf_gen()
import FF_computation
import pylab
from matplotlib.colors import Normalize
import pycbc
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform,get_fd_waveform, fd_approximants


q_inj = np.linspace(2,10,5) #mass ratio values 
l_inj = [np.pi/6,np.pi/4,np.pi/3,5*np.pi/12,np.pi/2]
n_iters = 10 #number of start-points/niters
mt_inj = 100.0 #Total mass injected
FF_MV_IQ = np.zeros((len(q_inj),len(l_inj))) #Intializing an array to store all values of FF in grid
coordinates = [] #Initializing th egrid matrix of q and inclination
for i in range(len(q_inj)):
    for j in range(len(l_inj)):
        #Injection parameters
        
        inj_params = dict(approximant="IMRPhenomXPHM",
                             mass1=mt_inj * q_inj[i]/ (1 + q_inj[i]), 
                             mass2=mt_inj  / (1 + q_inj[i]), 
                             delta_t=1.0/4096, 
                             f_lower=30.0,
                             spin1z = 0,
                             spin2z = 0, 
                             inclination=l_inj[j],
                             distance = 100,
                             delta_f =1.)
        #Wf generation
        hp,hc = get_fd_waveform(**inj_params)
        #Lensing Shift
        #function = np.sqrt(1.0)*np.exp( -1j*np.pi/2)
        #hp *= function

        #kwargs for FF 
        kwargs = dict(Mtot = 100, q = q_inj[i],coa_phase = -np.pi/4,inclination = l_inj[j],
             max_wait_per_iter = 1e2, default_value = 0.0 )
        ##FF Calculation
        FF_res = FF_computation.compute_fitting_factor(hp,apx="IMRPhenomXPHM",f_low=30, f_high=None, psd=None,
                                                n_iters=n_iters, xatols=['default'], max_iters=['default'], 
                                                branch_num=None, branch_depth=None,
                                                method='serial', **kwargs)
        #Storing values in the FF_MV_IQ matrix
        FF_MV_IQ[i,j] = FF_res[0,0]
        #Storing grid point for which the value is 
    #    for i in range(len(q_inj)):
    #for j in range(len(l_inj)):

#Printing values in the file opened above
np.savetxt('FF_value_HM_NL.txt',FF_MV_IQ)
#Plotting 
coordinates = np.array(coordinates)
plt.figure(figsize = (6,8))
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=FF_MV_IQ.flatten(), cmap='plasma_r')
cbar = plt.colorbar()
plt.xlabel('Mass ratio')
plt.ylabel('Incilnation')
cbar.set_label('FF_Value')
plt.savefig('FF_over2params_m_i_NL.png')
