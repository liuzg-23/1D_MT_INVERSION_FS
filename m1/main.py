# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:24:50 2023
@author: Zhengguang Liu
Employer: Central South University 
Email: zhengguang-liu@outlook.com
"""
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys 
sys.path.append("..")
from Inversion_module import gauss_newton_inv,ADMM_inv,gauss_newton_fourier_expansion,ADMM_fourier_expansion

############## a struct that store the inversion parameters ###################
class inversion_parameters:
    def _init_(self):
        self.t_s=0      # The thickness of seawater
        self.res_s=0.32 # The resistivity of seawater
        self.inv_mode=1 # 1: isotropic inversion 2: anisotropic inversion
        self.a_d=0.001  # amount of model disturbance for jacobian calculation
        self.max_ite=50 # the maximum iterations for inversion
        self.rms_min=1.0 # the terminating condition of data misfit
        self.noise_floor=1e-12 # noise floor for observation data
        self.step_1     # the number of the tried step value for model update
        self.step_2     # the decay factor for step size
        self.ul_1=10000 # upper bound of the resistivity value in inversion
        self.ul_2=1     # lower bound of the resistivity value in inversion
        self.ul_3=1     # Coefficients of upper and lower bound transformation
        self.rp_1=100   # the initial value of regularization parameter for rho_xx
        self.rp_2=0.1   # the initial value of regularization parameter for rho_yy
        self.rp_3=0.01  # threshold of rms for regularization parameter
        self.rp_4=0.001 # attenuation rate of regularization parameter
        self.n_l=50     # the number of layers
        self.t_0=20     # the thickness of the first layer
        self.f_i=1.05   # expansion factor of thickness
        self.m_0=[]     # initial model
        self.anis=[]    # Anisotropies lambda
        self.n_ser=1    # the number of fourier expansion
        self.omega=5    # a parameter of fourier expansion
        self.alpha=1    # the coefficient of L1 penalty term for ADMM
        self.beta=5     # Coefficients of the penalty term in the augmented Lagrangian function
        self.f_name="io_files"  # The name of the folder that including input and output files
###############################################################################
inv_para=inversion_parameters()
file_name="inversion_parameters.txt"
input_para=np.zeros([25])
n_line=0
with open(file_name,'r') as file:
    for line in file:
        value=line.split('#')[0].strip()
        input_para[n_line]=value
        n_line+=1
inv_algorithm=int(input_para[0])
inv_para.inv_mode=int(input_para[1])
inv_para.max_ite=int(input_para[2])
inv_para.rms_min=input_para[3]
inv_para.noise_floor=input_para[4]
inv_para.rp_1=input_para[5]
inv_para.rp_2=input_para[6]
inv_para.rp_3=input_para[7]
inv_para.rp_4=input_para[8]
inv_para.ul_1=input_para[9]
inv_para.ul_2=input_para[10]
inv_para.ul_3=input_para[11]
inv_para.n_l=int(input_para[12])
inv_para.t_0=input_para[13]
inv_para.f_i=input_para[14]
inv_para.t_s=input_para[15]
inv_para.res_s=input_para[16]
inv_para.n_ser=int(input_para[17])
inv_para.omega=input_para[18]
inv_para.alpha=input_para[19]
inv_para.beta=input_para[20]
inv_para.a_d=input_para[21]
inv_para.step_1=int(input_para[22])
inv_para.step_2=input_para[23]
if inv_para.inv_mode==1: # Isotropic inversion
    inv_para.m_0=np.zeros([inv_para.n_l])
else: # Anisotropic inversion
    inv_para.m_0=np.zeros([2*inv_para.n_l])
inv_para.m_0[:]=input_para[24]
start_time=time.perf_counter()
if inv_algorithm==1:
    gauss_newton_inv(inv_para)
elif inv_algorithm==2:
    gauss_newton_fourier_expansion(inv_para)
elif inv_algorithm==3:
    ADMM_inv(inv_para)
else:
    ADMM_fourier_expansion(inv_para) 
end_time=time.perf_counter()
print("The runtime for this inversion is:",'%.2f'%(end_time-start_time),"s.")