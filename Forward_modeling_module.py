# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:24:50 2023
@author: Zhengguang Liu
Employer: Central South University 
Email: zhengguang-liu@outlook.com
"""
import numpy as np
import math
import cmath

def Frequency_Domain_MT_Modeling(data_type,frequencies,true_model):
    inf=float('inf')
    n_l=len(true_model[0])
    n_f=len(frequencies)
    ep=np.zeros([n_l,17])
    for i in range(n_l):
        # the depth of the model
        if i==(n_l-1):
            ep[i,16]=inf
        else:
            ep[i,16]=true_model[0,i+1]
        # the electrical parameters
        ep[i,0]=1.0/true_model[1,i]
        ep[i,1]=1.0/true_model[2,i]
        ep[i,2]=1.0/true_model[1,i]
        ep[i,15]=1.0
    Zxy=np.zeros(2*n_f)
    Zyx=np.zeros(2*n_f)
    Zxy_yx=np.zeros(4*n_f)
    app_pha_xy=np.zeros(2*n_f)
    app_pha_yx=np.zeros(2*n_f)
    app_pha=np.zeros(4*n_f)
    app=np.zeros(2*n_f)
    for nf in range (n_f):
        App,ph,impedence=MT1D(n_l,ep,frequencies[nf])
        Zxy[nf*2]=impedence[0,1].real
        Zxy[nf*2+1]=impedence[0,1].imag
        Zyx[nf*2]=impedence[1,0].real
        Zyx[nf*2+1]=impedence[1,0].imag
        Zxy_yx[nf*4]=Zxy[nf*2]
        Zxy_yx[nf*4+1]=Zxy[nf*2+1]
        Zxy_yx[nf*4+2]=Zyx[nf*2]
        Zxy_yx[nf*4+3]=Zyx[nf*2+1]
        app_pha_xy[nf*2]=App[0,1]
        app_pha_xy[nf*2+1]=ph[0,1]
        app_pha_yx[nf*2]=App[1,0]
        app_pha_yx[nf*2+1]=ph[1,0]
        app_pha[nf*4]=App[0,1]
        app_pha[nf*4+1]=ph[0,1]
        app_pha[nf*4+2]=App[1,0]
        app_pha[nf*4+3]=ph[1,0]
        app[nf*2]=App[0,1]
        app[nf*2+1]=App[1,0]
    if data_type==1.0:
        return Zxy
    elif data_type==2.0:
        return app_pha_xy
    elif data_type==3.0:
        return Zxy_yx
    elif data_type==4.0:
        return app_pha
    elif data_type==5.0:
        return app
def MT1D(n,ep,f):
    Rz=np.zeros([3,3])
    Rx=np.zeros([3,3])
    R_z=np.zeros([3,3])
    cond=np.zeros([3,3])
    epsilon0=8.854187817e-12
    tol=1e-14
    epsilon=np.zeros([n+1,3,3])
    A=np.zeros([n+1,2,2],dtype=complex)
    y_hat=np.zeros([n+1,3,3],dtype=complex)
    z_hat=np.zeros([n+1],dtype=complex)
    depth=np.zeros([n+1])
    h=np.zeros([n+1])
    sigma=np.zeros([n+1,3,3])
    K1=np.zeros(n+1,dtype=complex)
    K2=np.zeros(n+1,dtype=complex)
    omega=2*math.pi*f
    Z=np.zeros([n,2,2],dtype=complex)
    for i in range(1,n+1):
#conductivity tensor
     alfa=ep[i-1,3]/180*math.pi
     beta=ep[i-1,4]/180*math.pi
     gama=ep[i-1,5]/180*math.pi
#初始化主电导率
     cond[0,0]=ep[i-1,0]
     cond[1,1]=ep[i-1,1]
     cond[2,2]=ep[i-1,2]
#初始化旋转角alfa  绕x轴旋转
     Rz[0,0]=math.cos(alfa)
     Rz[0,1]=-math.sin(alfa)
     Rz[1,0]=math.sin(alfa)
     Rz[1,1]=math.cos(alfa)
     Rz[2,2]=1
#初始化旋转角beta   绕y轴旋转
     Rx[0,0]=math.cos(beta)
     Rx[0,2]=math.sin(beta)
     Rx[1,1]=1
     Rx[2,0]=-math.sin(beta)
     Rx[2,2]=math.cos(beta)
#初始化旋转角gamma   绕z轴旋转
     R_z[0,0]=math.cos(gama)
     R_z[0,1]=-math.sin(gama)
     R_z[1,0]=math.sin(gama)
     R_z[1,1]=math.cos(gama)
     R_z[2,2]=1


#计算电导率
    #  sigma[i]=(Rz@Rx@R_z@cond)@(R_z.T@Rx.T@Rz.T)
     sigma[i]=(Rz@Rx@R_z)@cond@(R_z.T@Rx.T@Rz.T)
#初始化磁导率
     epsilon[i,0,0]=ep[i-1,6]
     epsilon[i,0,1]=ep[i-1,7]
     epsilon[i,0,2]=ep[i-1,8]
     epsilon[i,1,0]=ep[i-1,9]
     epsilon[i,1,1]=ep[i-1,10]
     epsilon[i,1,2]=ep[i-1,11]
     epsilon[i,2,0]=ep[i-1,12]
     epsilon[i,2,1]=ep[i-1,13]
     epsilon[i,2,2]=ep[i-1,14]
     epsilon[i]=epsilon[i]*epsilon0
#计算磁化后的电导率
     y_hat[i]=sigma[i]-1j*omega*epsilon[i]
     
#计算A系数
     A[i,0,0]=y_hat[i,0,0]-y_hat[i,0,2]*y_hat[i,2,0]/y_hat[i,2,2]
     A[i,0,1]=y_hat[i,0,1]-y_hat[i,0,2]*y_hat[i,2,1]/y_hat[i,2,2]
     A[i,1,0]=y_hat[i,1,0]-y_hat[i,1,2]*y_hat[i,2,0]/y_hat[i,2,2]
     A[i,1,1]=y_hat[i,1,1]-y_hat[i,1,2]*y_hat[i,2,1]/y_hat[i,2,2]
#Axy为0，则Ayx必须为0
     # if(abs(A[i,1,0])<tol):
     #  if(abs(A[i,0,1])>tol):
     #      break
     # print('error')
#计算电阻
     mu=ep[i-1,15]*4*math.pi*1e-7
     z_hat[i]=1j*omega*mu
#计算深度
     depth[i]=ep[i-1,16]
#计算每层厚度
    for i in range(1,n+1):
     assert(depth[i]>depth[i-1])
     h[i]=depth[i]-depth[i-1]

#计算K1,K2
    a=0
    b=0
    K12=0
    K22=0
    Q1=np.zeros(n+1,dtype=complex)
    Q2=np.zeros(n+1,dtype=complex)
    D1=np.zeros(n+1,dtype=complex)
    D2=np.zeros(n+1,dtype=complex)
    D3=np.zeros(n+1,dtype=complex)
    D4=np.zeros(n+1,dtype=complex)
    D5=np.zeros(n+1,dtype=complex)
    gama1=np.zeros(n+1,dtype=complex)
    gama2=np.zeros(n+1,dtype=complex)
    P=np.zeros([n+1,6,1],dtype=complex)
    U1=np.zeros([n+1,6,1],dtype=complex)
    U2=np.zeros([n+1,6,1],dtype=complex)
    U3=np.zeros([n+1,6,1],dtype=complex)
    U4=np.zeros([n+1,6,1],dtype=complex)
    for i in range(1,n+1):
     if(abs(A[i,1,0])<tol):
      K12=-z_hat[i]*A[i,0,0]
      K22=-z_hat[i]*A[i,1,1]
     else:
      a=(A[i,0,0]-A[i,1,1])*(A[i,0,0]-A[i,1,1])
      b=cmath.sqrt(a+4*A[i,0,1]*A[i,1,0])
      K12=-0.5*z_hat[i]*(A[i,0,0]+A[i,1,1]+b)
      K22=-0.5*z_hat[i]*(A[i,0,0]+A[i,1,1]-b)
     #计算K1
     if(cmath.sqrt(K12).real>0):
      K1[i]=cmath.sqrt(K12)
     else:
      K1[i]=-cmath.sqrt(K12)
     #计算K2    
     if(cmath.sqrt(K22).real>0):
      K2[i]=cmath.sqrt(K22)
     else:
      K2[i]=-cmath.sqrt(K22)
#计算Q1,Q2，gama1,gama2,D1-D5
    h_n=0
    z_hat_n=A10=A11=k1=k2=0
    for i in range(1,n+1):
        h_n=h[i]
        z_hat_n=z_hat[i]
        A10=A[i,1,0]
        A11=A[i,1,1]
        k1=K1[i]
        k2=K2[i]
    #对于解耦层计算Q1,Q2
        if(abs(A10)<tol):
         Q1[i]=0
         Q2[i]=0
        else:
         Q1[i]=(z_hat_n*A10)/(k1*k1+z_hat_n*A11)
         Q2[i]=(z_hat_n*A10)/(k2*k2+z_hat_n*A11)
    #计算D1-D5
        D1[i]=1+cmath.exp(-2*(k1+k2)*h_n)-cmath.exp(-2*k1*h_n)-cmath.exp(-2*k2*h_n)
        D2[i]=1+cmath.exp(-2*(k1+k2)*h_n)+cmath.exp(-2*k1*h_n)+cmath.exp(-2*k2*h_n)
        D3[i]=1-cmath.exp(-2*(k1+k2)*h_n)+cmath.exp(-2*k1*h_n)-cmath.exp(-2*k2*h_n)
        D4[i]=1-cmath.exp(-2*(k1+k2)*h_n)-cmath.exp(-2*k1*h_n)+cmath.exp(-2*k2*h_n)
        D5[i]=4*cmath.exp(-(k1+k2)*h_n)
        d1=D1[i]
        d2=D2[i]
        d3=D3[i]
        d4=D4[i]
        d5=D5[i]
    #计算gama1,gama2    
        gama1[i]=-k1/z_hat_n
        gama2[i]=-k2/z_hat_n
    #计算P，U1-U4
        r1=gama1[i]
        r2=gama2[i]
        if(abs(A[i,1,0])<tol):
         q1=0
         q2=0
         q=0
        else:
         q1=Q1[i]
         q2=1/Q2[i]
         q=q1*q2
   #计算P
        P[i,0,0]=r1*r2*d1*(q-1)
        P[i,1,0]=q1*(r2*d3-r1*d4)
        P[i,2,0]=q*r2*d3-r1*d4
        P[i,3,0]=r2*d3-q*r1*d4
        P[i,4,0]=q2*(r2*d3-r1*d4)
        P[i,5,0]=(q-1)*d2
        qq=(q-1)*r1*r2
  #计算U1
        U1[i,0,0]=(r2*d3-r1*d4)*q2
        U1[i,1,0]=(q*d1*(r1*r1+r2*r2)+((q*q+1)*d5-2*q*d2)*r1*r2)/qq
        U1[i,2,0]=(d1*(q*r2*r2+r1*r1)+(q+1)*(d5-d2)*r1*r2)*q2/qq
        U1[i,3,0]=(d1*(q*r1*r1+r2*r2)+(q+1)*(d5-d2)*r1*r2)*q2/qq
        U1[i,4,0]=(d1*(r1*r1+r2*r2)+2*(d5-d2)*r1*r2)*q2*q2/qq
        U1[i,5,0]=(r2*d4-r1*d3)/(r1*r2)*q2
  #计算U2
        U2[i,0,0]=q*r1*d4-r2*d3
        U2[i,1,0]=(-q1*((q*r1*r1+r2*r2)*d1+(d5-d2)*(q+1)*r1*r2))/qq
        U2[i,2,0]=((q*q+1)*r1*r2*d2-q*((r1*r1+r2*r2)*d1+2*r1*r2*d5))/qq
        U2[i,3,0]=(-d1*(q*q*r1*r1+r2*r2)-2*q*r1*r2*(d5-d2))/qq
        U2[i,4,0]=(-d1*(q*r1*r1+r2*r2)-(d5-d2)*(q+1)*r1*r2)*q2/qq
        U2[i,5,0]=(q*r1*d3-r2*d4)/(r1*r2)
  #计算U3
        U3[i,0,0]=r1*d4-q*r2*d3
        U3[i,1,0]=-(q1*((r1*r1+q*r2*r2)*d1+(d5-d2)*(q+1)*r1*r2))/qq
        U3[i,2,0]=-(d1*(r1*r1+q*q*r2*r2)+2*q*r1*r2*(d5-d2))/qq
        U3[i,3,0]=((q*q+1)*r1*r2*d2-q*((r1*r1+r2*r2)*d1+2*r1*r2*d5))/qq
        U3[i,4,0]=-(d1*(r1*r1+q*r2*r2)+(d5-d2)*(q+1)*r1*r2)*q2/qq
        U3[i,5,0]=(r1*d3-q*r2*d4)/(r1*r2)
 #计算U4
        U4[i,0,0]=q1*(r2*d3-r1*d4)
        U4[i,1,0]=(q1*q1*((r1*r1+r2*r2)*d1+2*r1*r2*(d5-d2)))/qq
        U4[i,2,0]=(q1*((r1*r1+q*r2*r2)*d1+(d5-d2)*(q+1)*r1*r2))/qq
        U4[i,3,0]=(q1*((q*r1*r1+r2*r2)*d1+(d5-d2)*(q+1)*r1*r2))/qq
        U4[i,4,0]=(q*d1*(r1*r1+r2*r2)+r1*r2*((q*q+1)*d5-2*q*d2))/qq
        U4[i,5,0]=q1*(r2*d4-r1*d3)/(r1*r2)
       
 #计算第n层的阻抗
    ZN=np.zeros([2,2],dtype=complex)
    ZN[0,0]=(gama2[n]-gama1[n])*q2
    ZN[0,1]=q*gama1[n]-gama2[n]
    ZN[1,0]=gama1[n]-q*gama2[n]
    ZN[1,1]=q1*(gama2[n]-gama1[n])
    ZN=1/(gama1[n]*gama2[n]*(q-1))*ZN
    Z[n-1]=ZN
    V=np.zeros([6,1],dtype=complex)
    M=np.zeros([2,2],dtype=complex)
    for i in range(n-1,0,-1):
        V[0,0]=Z[i,0,0]*Z[i,1,1]-Z[i,0,1]*Z[i,1,0]
        V[1,0]=Z[i,0,0]
        V[2,0]=Z[i,0,1]
        V[3,0]=Z[i,1,0]
        V[4,0]=Z[i,1,1]
        V[5,0]=1
        #计算det
        det=np.dot(P[i].T,V)
        #计算M
        M[0,0]=np.dot(U1[i].T,V)
        M[0,1]=np.dot(U2[i].T,V)
        M[1,0]=np.dot(U3[i].T,V)
        M[1,1]=np.dot(U4[i].T,V)
        #迭代计算Z
        Z[i-1]=M/det
    # print(Z)
    #计算视电阻率
    App=np.zeros([2,2])
    Phase=np.zeros([2,2])
    for i in range(2):
        for j in range(2):
            App[i,j]=1/(2*math.pi*f*mu)*abs(Z[0,i,j])*abs(Z[0,i,j])
    #计算相位
            Phase[i,j]=math.atan2(Z[0,i,j].imag,Z[0,i,j].real)/math.pi*180
            if 90<=Phase[i,j] and Phase[i,j]<=180:
               Phase[i,j]=Phase[i,j]-90
            elif -180<=Phase[i,j] and Phase[i,j]<=-90:
               Phase[i,j]=Phase[i,j]+180
            elif -90<=Phase[i,j] and Phase[i,j]<=0:
               Phase[i,j]=Phase[i,j]+90
    return App,Phase,Z[0]