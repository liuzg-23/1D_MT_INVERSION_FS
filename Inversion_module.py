# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:24:50 2023
@author: Zhengguang Liu
Employer: Central South University 
Email: zhengguang-liu@outlook.com
"""

import numpy as np
import os
import math
import matplotlib.pyplot as plt
from Forward_modeling_module import Frequency_Domain_MT_Modeling

########################## global variables ###################################
frequencies=[]
d_t=1
n_f=0
n_d=0
n_l=1
inv_mode=1
n_m=1
dep=[]
dep_c=[]
w_m=[[]]
w_d=[[]]
d_obs=[]
###############################################################################

####################### Initialize inversion parameters #######################
def init_inv_para(inv_para):
    global obs_data,frequencies, n_f, n_d,d_t,n_l,n_m,inv_mode,dep,dep_c,w_m,w_d,d_obs
    n_l=inv_para.n_l
    inv_mode=inv_para.inv_mode
    n_m=n_l*inv_mode
    dep_c=np.zeros([n_l])
    for nl in range(n_l):
        if nl==0:
            dep_c[nl]=inv_para.t_0/2
        else:
            dep_c[nl]=dep_c[nl-1]+inv_para.t_0*pow(inv_para.f_i,(nl-1))/2+\
                inv_para.t_0*pow(inv_para.f_i,nl)/2
    if inv_para.t_s==0:
        dep=np.zeros([n_l])
    else:
        dep=np.zeros([n_l+1])
    
    for nl in range(len(dep)):
        if inv_para.t_s==0:
            if nl==0:
                dep[nl]=0
            else:
                dep[nl]=dep[nl-1]+inv_para.t_0*pow(inv_para.f_i,(nl-1))
        else:
            if nl==0:
                dep[nl]=0
            elif nl==1:
                dep[nl]=inv_para.t_s
            else:
                dep[nl]=dep[nl-1]+inv_para.t_0*pow(inv_para.f_i,(nl-2))
    print("Depth of the inverse model:",'%.0f'%dep[len(dep)-1],'m')
    w_m=np.zeros([n_m,n_m])
    if inv_mode==1:
        for nl in range(n_l):
            if nl!=0:
                w_m[nl,nl-1]=-1*np.sqrt(inv_para.rp_1)
                w_m[nl,nl]=1*np.sqrt(inv_para.rp_1)
    else:
        for nl in range(n_l):
            if nl!=0:
                w_m[nl,nl-1]=-1*np.sqrt(inv_para.rp_1)
                w_m[nl,nl]=1*np.sqrt(inv_para.rp_1)
                w_m[nl+n_l,nl+n_l-1]=-1*np.sqrt(inv_para.rp_2)
                w_m[nl+n_l,nl+n_l]=1*np.sqrt(inv_para.rp_2)
    obs_data=np.loadtxt("observation_data.txt")
    d_t=obs_data[0,0]
    frequencies=obs_data[:,1]
    n_f=len(frequencies)
    if d_t==1 or d_t==2 or d_t==5:
        n_d=2*n_f
    else:
        n_d=4*n_f
    d_obs=np.zeros([n_d])
    w_d=np.zeros([n_d,n_d])
    for nd in range(n_f):
        if d_t==1 or d_t==2 or d_t==5:
            d_obs[2*nd]=obs_data[nd][2]
            d_obs[2*nd+1]=obs_data[nd][3]
            w_d[2*nd,2*nd]=1.0/(obs_data[nd][4]+inv_para.noise_floor)
            w_d[2*nd+1,2*nd+1]=1.0/(obs_data[nd][5]+inv_para.noise_floor)
        else:
            d_obs[4*nd]=obs_data[nd][2]
            d_obs[4*nd+1]=obs_data[nd][3]
            w_d[4*nd,4*nd]=1.0/(obs_data[nd][4]+inv_para.noise_floor)
            w_d[4*nd+1,4*nd+1]=1.0/(obs_data[nd][5]+inv_para.noise_floor)
            d_obs[4*nd+2]=obs_data[nd][6]
            d_obs[4*nd+3]=obs_data[nd][7]
            w_d[4*nd+2,4*nd+2]=1.0/(obs_data[nd][8]+inv_para.noise_floor)
            w_d[4*nd+3,4*nd+3]=1.0/(obs_data[nd][9]+inv_para.noise_floor)
###############################################################################  
############ model transformation for upper and lower constraint ##############
def upper_lower_trans(upper,lower,n,t_t,m):
    x=np.zeros([n_m])
    ul=np.zeros([n_m,n_m])
    for i in range(n_m):
        if t_t==1:
            x[i]=(1.0/n)*np.log((m[i]-lower)/(upper-m[i]))
            ul[i,i]=(upper-m[i])*(m[i]-lower)/(upper-lower)
        else:
            exp_v=np.exp(n*m[i])
            if abs(exp_v)>1e8:
                x[i]=upper-1e-8
            elif abs(exp_v)<1e-8:
                x[i]=lower+1e-8
            else:
                x[i]=(lower+upper*exp_v)/(1+exp_v)
            ul[i,i]=(upper-x[i])*(x[i]-lower)/(upper-lower)
    return ul,x        
###############################################################################

########### calculate the jacobian matrix using perturbation method ###########
def cal_jacobian(inv_para,x_p,m_,m_p,d_f):
    JT=np.zeros([n_m,n_d])
    m_r=m_.copy()
    for nl in range (n_m):
        if x_p[nl]==0:
            JT[nl,:]=0
        else:
            m_r[nl]=m_p[nl]
            model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_r)
            d_f2=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            m_r[nl]=m_[nl]
            JT[nl,:]=(d_f2-d_f)/(x_p[nl])
    return JT

def fourier_trans(omega,x_k,n_ser,upper,lower,n):
    f_k=np.zeros([n_m])
    m_k=np.zeros([n_m])
    ul=np.zeros([n_m,n_m])
    a=x_k[:n_ser]
    af=np.zeros([n_m,n_ser])
    if inv_mode==2:
        a2=x_k[n_ser:2*n_ser]
        af=np.zeros([n_m,2*n_ser])
    dep_ln=np.log(dep_c)
    for nl in range(n_l):
        f_k[nl]+=a[0]
        af[nl,0]=1
        for nser in range(n_ser-1):
            Tz=omega*(nser+1)*dep_ln[nl]
            f_k[nl]+=a[nser+1]*2*(np.exp(complex(0,Tz))).real
            af[nl,nser+1]=2*(np.exp(complex(0,Tz))).real
        if inv_mode==2:
            f_k[nl+n_l]+=a2[0]
            af[nl+n_l,n_ser]=1
            for nser in range(n_ser-1):
                Tz=omega*(nser+1)*dep_ln[nl]
                f_k[nl+n_l]+=a2[nser+1]*2*(np.exp(complex(0,Tz))).real
                af[nl+n_l,n_ser+nser+1]=2*(np.exp(complex(0,Tz))).real
    for nm in range(n_m):
        exp_v=np.exp(n*f_k[nm])
        if abs(exp_v)>1e8:
            m_k[nm]=upper-1e-8
        elif abs(exp_v)<1e-8:
            m_k[nm]=lower+1e-8
        else:
            m_k[nm]=(lower+upper*exp_v)/(1+exp_v)
        ul[nm,nm]=(upper-m_k[nm])*(m_k[nm]-lower)/(upper-lower)
    af=ul.dot(af)
    return af,m_k
def cal_jacobian_2(inv_para,x_k,d_f,omega):
    n_i=x_k.shape[0]
    x_p=x_k.copy()
    JT=np.zeros([n_i,len(d_f)])
    for i in range (n_i):
        if x_k[i]==0:
            per=inv_para.a_d
        else:
            per=x_k[i]*inv_para.a_d
        x_p[i]=x_k[i]+per
        a_f,m_p=fourier_trans(omega,x_p,inv_para.n_ser,inv_para.ul_1,inv_para.ul_2,inv_para.ul_3)
        model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_p)
        d_f2=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
        x_p=x_k.copy()
        JT[i,:]=(d_f2-d_f)/per
    return JT               
###############################################################################                
def output_inv_results(rms,r_p,d_obs,d_f,inv_results,Inv_method):
    if not os.path.exists(Inv_method):
        os.makedirs(Inv_method)
    fw_1=open(Inv_method+"/Inv_para.txt","w")
    fw_2=open(Inv_method+"/Data_file.txt","w")
    fw_3=open(Inv_method+"/Model_file.txt","w")
    for i in range(len(rms)):
        fw_1.write(str(i)+"   "+str(rms[i])+"   "+str(r_p[i])+"\n")
    for i in range(len(d_obs)):
        fw_2.write(str(d_obs[i])+"   "+str(d_f[i])+"\n")
    depth=np.zeros([n_l])
    for i in range(n_l):          
        fw_3.write(str(dep_c[i])+"   ")
    if inv_mode==2:
        for i in range(n_l):          
            fw_3.write(str(dep_c[i])+"   ")
    fw_3.write("\n")
    for i in range(len(inv_results)):
        if inv_mode==1:
            for j in range(n_l):
                fw_3.write(str(inv_results[i][j])+"   ")
            fw_3.write("\n")
        else:
            for j in range(n_m):
                fw_3.write(str(inv_results[i][j])+"   ")
            fw_3.write("\n")

def construct_model_parameters(t_s,res_s,m_):
    if t_s==0:
        model=np.zeros([3,n_l])
    else:
        model=np.zeros([3,n_l+1])
    if inv_mode==1:
        if t_s==0: # land model
            model[0,:]=dep.copy()
            model[1,:]=m_.copy()
            model[2,:]=m_.copy()
        else: # marine model
            model[0,:]=dep.copy()
            model[1,0]=res_s
            model[2,0]=res_s
            model[1,1:]=m_.copy()
            model[2,1:]=m_.copy()
    else:
        if t_s==0: # land model
            model[0,:]=dep.copy()
            model[1,:]=m_[:n_l]
            model[2,:]=m_[n_l:]
        else: # marine model
            model[0,:]=dep.copy()
            model[1,0]=res_s
            model[2,0]=res_s
            model[1,1:]=m_[:n_l]
            model[2,1:]=m_[n_l:]
    return model
#################### Inversion using Gauss-Newton method ######################
def gauss_newton_inv(inv_para):
    init_inv_para(inv_para)
    rms=[]
    r_p=[]
    inv_results=[]
    x_k=np.zeros([n_m])
    m_k=inv_para.m_0.copy()
    model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
    A,x_k=upper_lower_trans(inv_para.ul_1,inv_para.ul_2,inv_para.ul_3,1,m_k)
    w_a=w_m.dot(A)
    for k in range(inv_para.max_ite):
        print("The", k+1,"th", "iteration of GN algorithm:")
        inv_results.append(m_k)
        if k==0:
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            r_p.append(1.0)
            rms.append(np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d))
        else:
            if ((rms[k-1]-rms[k])/rms[k-1])<inv_para.rp_4:
                r_p.append(r_p[k-1]*inv_para.rp_3)
            else:
                r_p.append(r_p[k-1])
        print("  rms:",'%.3f'%rms[k])
        print("  lambda:",'%.5f'%r_p[k])
        if rms[k]<inv_para.rms_min:
            print("The data misfit reached the pre-set value, stop iteration.")
            break
        if k==inv_para.max_ite-1:
            print("The maximum iterations is reached, stop iteration.")
            break
        if 3<k:
            if ((rms[k-1]-rms[k])/rms[k-1])<0.01 and ((rms[k-2]-rms[k-1])/rms[k-2])<0.01\
                and ((rms[k-3]-rms[k-2])/rms[k-3])<0.01:
                print("The rms decreases very little three times in a row, stop iteration.")
                break
        x_p=x_k*(1+inv_para.a_d)
        aa,m_p=upper_lower_trans(inv_para.ul_1,inv_para.ul_2,inv_para.ul_3,2,x_p)
        x_p=x_p-x_k
        JT=cal_jacobian(inv_para,x_p,m_k,m_p,d_f)
        gk=((JT.dot(w_d.T)).dot(w_d)).dot(d_f-d_obs)+\
            r_p[k]*((w_a.T).dot(w_m)).dot(m_k)
        Hk=((JT).dot((w_d.T).dot(w_d))).dot(JT.T)+np.eye(n_m)+r_p[k]*(w_a.T).dot(w_a)
        d_k=-(np.linalg.inv(Hk)).dot(gk)
        for kk in range(inv_para.step_1):
            alpha=pow(inv_para.step_2,kk)
            x_kn=x_k+alpha*d_k
            A,m_k=upper_lower_trans(inv_para.ul_1,inv_para.ul_2,inv_para.ul_3,2,x_kn)
            model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            rms_kk=np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d)
            if kk==0:
                x_best=x_kn.copy()
                m_best=m_k.copy()
                A_best=A.copy()
                d_best=d_f.copy()
                alpha_best=alpha
                rms_best=rms_kk
            else:
                if rms_kk<rms_best:
                    x_best=x_kn.copy()
                    m_best=m_k.copy()
                    A_best=A.copy()
                    d_best=d_f.copy()
                    alpha_best=alpha
                    rms_best=rms_kk
            if kk==(inv_para.step_1-1):
                x_k=x_best.copy()
                m_k=m_best.copy()
                A=A_best.copy()
                d_f=d_best.copy()
                rms.append(rms_best)
                print("  The step value:",alpha_best)
        # print(m_k)
        w_a=w_m.dot(A)
    output_inv_results(rms,r_p,d_obs,d_f,inv_results,'GN')                
###############################################################################

#################### Inversion using ADMM method ##############################
def ADMM_inv(inv_para):
    init_inv_para(inv_para)
    rms=[]
    r_p=[]
    inv_results=[]
    x_k=np.zeros([n_m])
    m_k=inv_para.m_0.copy()
    model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
    x_k=upper_lower_trans(inv_para.ul_1,inv_para.ul_2,inv_para.ul_3,1,m_k)
    z_k=np.zeros([n_m])
    u_k=np.zeros([n_m])
    z_k=x_k.copy()
    beta=inv_para.beta
    for k in range(inv_para.max_ite):
        print("The", k+1,"th", "iteration of ADMM algorithm:")
        inv_results.append(m_k)
        if k==0:
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            r_p.append(inv_para.alpha)
            rms.append(np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d))
        else:
            if ((rms[k-1]-rms[k])/rms[k-1])<inv_para.rp_4:
                r_p.append(r_p[k-1]*inv_para.rp_3)
                beta=beta*inv_para.rp_3
            else:
                r_p.append(r_p[k-1])
        print("  rms:",'%.3f'%rms[k])
        print("  lambda:",'%.5f'%r_p[k])
        if rms[k]<inv_para.rms_min:
            print("The data misfit reached the pre-set value, stop iteration.")
            break
        if k==inv_para.max_ite-1:
            print("The maximum iterations is reached, stop iteration.")
            break
        if 3<k:
            if ((rms[k-1]-rms[k])/rms[k-1])<0.01 and ((rms[k-2]-rms[k-1])/rms[k-2])<0.01\
                and ((rms[k-3]-rms[k-2])/rms[k-3])<0.01:
                print("The rms decreases very little three times in a row, stop iteration.")
                break
        x_p=x_k*(1+inv_para.a_d)
        m_p=upper_lower_trans(inv_para.ul_1,inv_para.ul_2,inv_para.ul_3,2,x_p)
        x_p=x_p-x_k
        JT=cal_jacobian(inv_para,x_p,m_k,m_p,d_f)
        gk=((JT.dot(w_d.T)).dot(w_d)).dot(d_f-d_obs)+\
            beta*w_m.T.dot(w_m.dot(x_k)-z_k+u_k)
        Hk=((JT).dot((w_d.T).dot(w_d))).dot(JT.T)+beta*w_m.T.dot(w_m)+np.eye(n_m)
        d_k=-(np.linalg.inv(Hk)).dot(gk)
        for kk in range(inv_para.step_1):
            alpha=pow(inv_para.step_2,kk)
            x_kn=x_k+alpha*d_k
            m_k=upper_lower_trans(inv_para.ul_1,inv_para.ul_2,inv_para.ul_3,2,x_kn)
            model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            rms_kk=np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d)
            if kk==0:
                x_best=x_kn
                m_best=m_k
                d_best=d_f
                alpha_best=alpha
                rms_best=rms_kk
            else:
                if rms_kk<rms_best:
                    x_best=x_kn
                    m_best=m_k
                    d_best=d_f
                    alpha_best=alpha
                    rms_best=rms_kk
            if kk==(inv_para.step_1-1):
                x_k=x_best
                m_k=m_best
                d_f=d_best
                rms.append(rms_best)
                print("  The step value:",alpha_best)
        # print(m_k)
        z_k=np.sign(w_m.dot(x_k)+u_k)*np.maximum(np.abs(w_m.dot(x_k)+u_k)-r_p[k]/beta, 0.0)
        u_k=u_k+w_m.dot(x_k)-z_k
    output_inv_results(rms,r_p,d_obs,d_f,inv_results,'ADMM')                
###############################################################################

############### Inversion using Gauss-newton with L2 roughness#################
def gauss_newton_fourier_expansion(inv_para):
    init_inv_para(inv_para)
    rms=[]
    r_p=[]
    inv_results=[]
    n_i=inv_para.n_ser
    if inv_mode==2:
        n_i=2*n_i
    A=np.zeros([n_l,n_i])
    x_k=np.zeros([n_i])
    m_k=inv_para.m_0.copy()
    x_k[0]=(1.0/inv_para.ul_3)*np.log((m_k[0]-inv_para.ul_2)/(inv_para.ul_1-m_k[0]))
    if inv_mode==2:
        x_k[inv_para.n_ser]=(1.0/inv_para.ul_3)*np.log((m_k[n_l]-inv_para.ul_2)/(inv_para.ul_1-m_k[n_l]))
    omega=inv_para.omega/np.log(dep_c[n_l-1])
    A,m_k=fourier_trans(omega,x_k,inv_para.n_ser,inv_para.ul_1,inv_para.ul_2,inv_para.ul_3)
    model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
    w_a=w_m.dot(A)
    for k in range(inv_para.max_ite):
        print("The", k+1,"th", "iteration of GN-fourier algorithm:")
        inv_results.append(m_k)
        if k==0:
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            r_p.append(1.0)
            rms.append(np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d))
        else:
            if ((rms[k-1]-rms[k])/rms[k-1])<inv_para.rp_4:
                r_p.append(r_p[k-1]*inv_para.rp_3)
            else:
                r_p.append(r_p[k-1])
        print("  rms:",'%.3f'%rms[k])
        print("  lambda:",'%.5f'%r_p[k])
        if rms[k]<inv_para.rms_min:
            print("The data misfit reached the pre-set value, stop iteration.")
            break
        if k==inv_para.max_ite-1:
            print("The maximum iterations is reached, stop iteration.")
            break
        if 3<k:
            if ((rms[k-1]-rms[k])/rms[k-1])<0.01 and ((rms[k-2]-rms[k-1])/rms[k-2])<0.01\
                and ((rms[k-3]-rms[k-2])/rms[k-3])<0.01:
                print("The rms decreases very little three times in a row, stop iteration.")
                break
        JT=cal_jacobian_2(inv_para,x_k,d_f,omega)
        # print(JT)
        gk=((JT.dot(w_d.T)).dot(w_d)).dot(d_f-d_obs)+r_p[k]*w_a.T.dot(w_m.dot(m_k))
        Hk=((JT).dot((w_d.T).dot(w_d))).dot(JT.T)+r_p[k]*w_a.T.dot(w_a)+np.eye(n_i)
        d_k=-(np.linalg.inv(Hk)).dot(gk)
        for kk in range(inv_para.step_1):
            alpha=pow(inv_para.step_2,kk)
            x_kn=x_k+alpha*d_k
            A,m_k=fourier_trans(omega,x_kn,inv_para.n_ser,inv_para.ul_1,inv_para.ul_2,inv_para.ul_3)
            model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            rms_kk=np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d)
            if kk==0:
                x_best=x_kn.copy()
                m_best=m_k.copy()
                A_best=A.copy()
                d_best=d_f.copy()
                alpha_best=alpha
                rms_best=rms_kk
            else:
                if rms_kk<rms_best:
                    x_best=x_kn.copy()
                    m_best=m_k.copy()
                    A_best=A.copy()
                    d_best=d_f.copy()
                    alpha_best=alpha
                    rms_best=rms_kk
            if kk==(inv_para.step_1-1):
                x_k=x_best.copy()
                m_k=m_best.copy()
                A=A_best.copy()
                d_f=d_best.copy()
                rms.append(rms_best)
                print("  The step value:",alpha_best)
        # print(m_k)
        w_a=w_m.dot(A)
    output_inv_results(rms,r_p,d_obs,d_f,inv_results,'GN_FS')
###############################################################################

################# Inversion using ADMM with fourier_expansion #################
def ADMM_fourier_expansion(inv_para):
    init_inv_para(inv_para)
    rms=[]
    r_p=[]
    inv_results=[]
    n_i=inv_para.n_ser
    if inv_mode==2:
        n_i=2*n_i
    A=np.zeros([n_l,n_i])
    x_k=np.zeros([n_i])
    m_k=inv_para.m_0.copy()
    z_k=np.zeros([n_m])
    u_k=np.zeros([n_m])
    x_k[0]=(1.0/inv_para.ul_3)*np.log((m_k[0]-inv_para.ul_2)/(inv_para.ul_1-m_k[0]))
    if inv_mode==2:
        x_k[inv_para.n_ser]=(1.0/inv_para.ul_3)*np.log((m_k[n_l]-inv_para.ul_2)/(inv_para.ul_1-m_k[n_l]))
    omega=inv_para.omega/np.log(dep_c[n_l-1])
    A,m_k=fourier_trans(omega,x_k,inv_para.n_ser,inv_para.ul_1,inv_para.ul_2,inv_para.ul_3)
    model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
    beta=inv_para.beta
    w_a=w_m.dot(A)
    z_k=w_m.dot(m_k)
    for k in range(inv_para.max_ite):
        print("The", k+1,"th", "iteration of ADMM-fourier algorithm:")
        inv_results.append(m_k)
        if k==0:
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            r_p.append(inv_para.alpha)
            rms.append(np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d))
        else:
            if ((rms[k-1]-rms[k])/rms[k-1])<inv_para.rp_4:
                r_p.append(r_p[k-1]*inv_para.rp_3)
                beta=beta*inv_para.rp_3
            else:
                r_p.append(r_p[k-1])
        print("  rms:",'%.3f'%rms[k])
        print("  lambda:",'%.5f'%r_p[k])
        if rms[k]<inv_para.rms_min:
            print("The data misfit reached the pre-set value, stop iteration.")
            break
        if k==inv_para.max_ite-1:
            print("The maximum iterations is reached, stop iteration.")
            break
        if 3<k:
            if ((rms[k-1]-rms[k])/rms[k-1])<0.01 and ((rms[k-2]-rms[k-1])/rms[k-2])<0.01\
                and ((rms[k-3]-rms[k-2])/rms[k-3])<0.01:
                print("The rms decreases very little three times in a row, stop iteration.")
                break
        JT=cal_jacobian_2(inv_para,x_k,d_f,omega)
        # print(JT)
        gk=((JT.dot(w_d.T)).dot(w_d)).dot(d_f-d_obs)+beta*w_a.T.dot(w_m.dot(m_k)-z_k+u_k)
        Hk=((JT).dot((w_d.T).dot(w_d))).dot(JT.T)+beta*w_a.T.dot(w_a)+np.eye(n_i)
        d_k=-(np.linalg.inv(Hk)).dot(gk)
        for kk in range(inv_para.step_1):
            alpha=pow(inv_para.step_2,kk)
            x_kn=x_k+alpha*d_k
            A,m_k=fourier_trans(omega,x_kn,inv_para.n_ser,inv_para.ul_1,inv_para.ul_2,inv_para.ul_3)
            model=construct_model_parameters(inv_para.t_s,inv_para.res_s,m_k)
            d_f=Frequency_Domain_MT_Modeling(d_t,frequencies,model)
            rms_kk=np.sqrt(np.sum((w_d.dot(d_f-d_obs))**2)/n_d)
            if kk==0:
                x_best=x_kn.copy()
                m_best=m_k.copy()
                A_best=A.copy()
                d_best=d_f.copy()
                alpha_best=alpha
                rms_best=rms_kk
            else:
                if rms_kk<rms_best:
                    x_best=x_kn.copy()
                    m_best=m_k.copy()
                    A_best=A.copy()
                    d_best=d_f.copy()
                    alpha_best=alpha
                    rms_best=rms_kk
            if kk==(inv_para.step_1-1):
                x_k=x_best.copy()
                m_k=m_best.copy()
                A=A_best.copy()
                d_f=d_best.copy()
                rms.append(rms_best)
                print("  The step value:",alpha_best)
        w_a=w_m.dot(A)
        z_k=np.sign(w_m.dot(m_k)+u_k)*np.maximum((np.abs(w_m.dot(m_k)+u_k)-(r_p[k]/beta)), np.zeros([n_m]))
        u_k=u_k+w_m.dot(m_k)-z_k
    output_inv_results(rms,r_p,d_obs,d_f,inv_results,'ADMM_FS') 
###############################################################################