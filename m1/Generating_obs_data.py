# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:24:50 2023
@author: Zhengguang Liu
Employer: Central South University 
Email: zhengguang-liu@outlook.com
"""
import numpy as np
import random
import os 
import sys 
sys.path.append("..")
from Forward_modeling_module import Frequency_Domain_MT_Modeling

########### generating the synthetic data with forward modeling code ##########
noise_level=0.01  # the noise level 
phase_error=2     # the error for the phase
data_type=4 # 1: Zxy; 2: rho_xy and phase_xy; 3: Zxy and Zyx;  4: rho_xy, phase_xy, rho_yx and phase_yx
frequencies=np.loadtxt("frequencies.txt")
true_model=np.loadtxt("true_model.txt")
responses=Frequency_Domain_MT_Modeling(data_type,frequencies,true_model)
###############################################################################

############## add Gaussian noise to the modeling data ######################## 
n_f=len(frequencies)
if data_type==1 or data_type==2:
    output=np.zeros([n_f,6])
else:
    output=np.zeros([n_f,10])
for nf in range(n_f):
    rand=random.gauss(0,1)
    output[nf,0]=data_type
    output[nf,1]=frequencies[nf]
    if data_type==1 or data_type==2:
        output[nf,2]=responses[2*nf]*(1+noise_level*rand)
        output[nf,3]=responses[2*nf+1]*(1+noise_level*rand)
        output[nf,4]=abs(complex(responses[2*nf],responses[2*nf+1]))*noise_level
        output[nf,5]=abs(complex(responses[2*nf],responses[2*nf+1]))*noise_level
    elif data_type==4:
        output[nf,2]=responses[4*nf]*(1+noise_level*rand)
        output[nf,3]=responses[4*nf+1]*(1+noise_level*rand)
        output[nf,4]=abs(complex(responses[4*nf],responses[4*nf+1]))*noise_level
        output[nf,5]=phase_error
        output[nf,6]=responses[4*nf+2]*(1+noise_level*rand)
        output[nf,7]=responses[4*nf+3]*(1+noise_level*rand)
        output[nf,8]=abs(complex(responses[4*nf+2],responses[4*nf+3]))*noise_level
        output[nf,9]=phase_error
    else:
        output[nf,2]=responses[4*nf]*(1+noise_level*rand)
        output[nf,3]=responses[4*nf+1]*(1+noise_level*rand)
        output[nf,4]=abs(complex(responses[4*nf],responses[4*nf+1]))*noise_level
        output[nf,5]=abs(complex(responses[4*nf],responses[4*nf+1]))*noise_level
        output[nf,6]=responses[4*nf+2]*(1+noise_level*rand)
        output[nf,7]=responses[4*nf+3]*(1+noise_level*rand)
        output[nf,8]=abs(complex(responses[4*nf+2],responses[4*nf+3]))*noise_level
        output[nf,9]=abs(complex(responses[4*nf+2],responses[4*nf+3]))*noise_level
###############################################################################

###################### output the synthetic noisy data ########################
f_w=open("observation_data.txt","w")
for n_rows in range(output.shape[0]):
    for n_cols in range(output.shape[1]):
        f_w.write(str(output[n_rows,n_cols])+"   ")
    f_w.write("\n")    
f_w.close()
###############################################################################