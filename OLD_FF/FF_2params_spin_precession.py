#!/home/krish.shah/anaconda3/envs/Teobresums/bin/python3

import numpy as np
import scipy
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt


import sys
#sys.path.append('../src/')
#sys.path.append('../FF_Analysis/')
sys.path.append('/home/krish.shah/GW_Lensing/FF_Computation/')

import py_lgw
lgw = py_lgw.lensed_wf_gen()
import FF_computation_spin_precession
import pylab
from matplotlib.colors import Normalize
import pycbc
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform,get_fd_waveform, fd_approximants


q_inj = 1/2
mt_inj = 100
l_inj = np.pi/6

chi_eff_inj = np.linspace(0,q_inj/(1+q_inj),5)
chi_p_inj =   np.linspace(0,1,10)

niters = 15
FF_MV_IQ = np.zeros((len(chi_eff_inj),len(chi_p_inj)))
coordinates = []
for i in range(len(chi_eff_inj)):
    for j in range(len(chi_p_inj)):

        inj_params = dict(approximant="IMRPhenomXPHM",
                         mass1=mt_inj / (1 + q_inj), 
                         mass2=q_inj*mt_inj  / (1 + q_inj), 
                         delta_t=1.0/4096, 
                         f_lower=30.0,
                         spin1z = 0,
                         spin2z = chi_eff_inj[i]*(1+q_inj)/q_inj,
                         spin1x = chi_p_inj[j]/np.sqrt(2),
                         spin1y = chi_p_inj[j]/np.sqrt(2),
                         spin2x = 0,
                         spin2y = 0, 
                         inclination=l_inj,
                         distance = 100,
                         delta_f =1.)

        #Wf generation
        hp,hc = get_fd_waveform(**inj_params)

        #Lensing Shift
        #function = np.sqrt(1.0)*np.exp( -1j*np.pi/2)
        #hp *= function

        kwargs = dict(Mtot = mt_inj, q = q_inj,chi_eff = chi_eff_inj[i],chi_p = chi_p_inj[j],
                         coa_phase = -np.pi/4,inclination = l_inj,
                            max_wait_per_iter = 1e2, default_value = 0.0 )

        FF_res = FF_computation_spin_precession.compute_fitting_factor(hp,apx="IMRPhenomXPHM",f_low=30, f_high=None, psd=None,
                                        n_iters=niters, xatols=['default'], max_iters=['default'], 
                                        branch_num=None, branch_depth=None,
                                        method='serial', **kwargs)

        FF_MV_IQ[i,j] = FF_res[0,0]
        coordinates.append((chi_eff_inj[i], chi_p_inj[j]))
    
np.savetxt('FF_Values_spin_precession_NL.txt',FF_MV_IQ)
        
coordinates = np.array(coordinates)
fig, ax = plt.subplots()
scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=FF_MV_IQ.flatten(), cmap='viridis_r')
cbar = plt.colorbar(scatter)
cbar.set_label('FF_Value')
plt.xlabel('Chi_eff')
plt.ylabel('Chi_precession')
a = l_inj/np.pi
if abs(l_inj) == 0:
    symbolic_q = '0'
elif abs(a) < 1:
    a = np.pi/l_inj
    symbolic_q = r'$\frac{\pi}{' + str(int(a)) + '}$'
elif abs(a) > 1:
    a = l_inj/np.pi
    symbolic_q = str(int(a)) + r'$ \pi ' +  '}$'

plt.title('For q = %i and inclination =  {}'.format(symbolic_q)%(1/q_inj))
plt.savefig('FF_over2params_m_i_spin_precessionq(%i)_inclination{}_NL.png')