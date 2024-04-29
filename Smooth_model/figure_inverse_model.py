# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:06:46 2019

@author: liuzg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from  matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times-Roman']})
rc('text', usetex = True)
#plt.style.use('science')
mr=np.loadtxt('true_model.txt')
m1=np.loadtxt('ADMM_FS_10/Model_file.txt')
m2=np.loadtxt('ADMM_FS_20/Model_file.txt')
m3=np.loadtxt('ADMM_FS_30/Model_file.txt')
m4=np.loadtxt('ADMM_FS_40/Model_file.txt')

n_l=int(m1.shape[1]/2)
dep_c=m1[0,:n_l]/1000
n_it1=m1.shape[0]-1
n_it2=m2.shape[0]-1
n_it3=m3.shape[0]-1
n_it4=m4.shape[0]-1
res_1=m1[n_it1,:]
res_2=m2[n_it2,:]
res_3=m3[n_it3,:]
res_4=m4[n_it4,:]

# Real model
n_r=mr.shape[1]*2
dep_r=np.zeros([n_r])
res_h=np.zeros([n_r])
res_v=np.zeros([n_r])
for i in range(mr.shape[1]):
    if i==mr.shape[1]-1:
        dep_r[2*i]=(mr[0,i]+1)/1000
        dep_r[2*i+1]=dep_c[n_l-1]
    else:
        dep_r[2*i]=(mr[0,i]+1)/1000
        dep_r[2*i+1]=(mr[0,i+1])/1000
    res_h[2*i]=mr[1,i]
    res_h[2*i+1]=mr[1,i]
    res_v[2*i]=mr[2,i]
    res_v[2*i+1]=mr[2,i]

xtick1=[1,10,100,1000,10000]
xtick2=[1,10,100,1000,10000]
ytick1=[1,10,100,1000]
ytick2=[1,10,100,1000]

fig=plt.figure(figsize=(5.0,3.2)) 
plt.subplots_adjust(wspace=0.15,top=0.92,bottom=0.15,right=0.95,left=0.12) 
 
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.spines['right'].set_linewidth('1.5')
ax1.spines['left'].set_linewidth('1.5')
ax1.spines['top'].set_linewidth('1.5')
ax1.spines['bottom'].set_linewidth('1.5')
ax2.spines['right'].set_linewidth('1.5')
ax2.spines['left'].set_linewidth('1.5')
ax2.spines['top'].set_linewidth('1.5')
ax2.spines['bottom'].set_linewidth('1.5')

ax1.plot(res_h,dep_r,'k-',linewidth=1.0,label='True model')
ax1.plot(res_1[:n_l],dep_c,'r-',linewidth=1.0,label=r'$N_{fs}=10$')
ax1.plot(res_2[:n_l],dep_c,'b-',linewidth=1.0,label=r'$N_{fs}=20$')
ax1.plot(res_3[:n_l],dep_c,'g-',linewidth=1.0,label=r'$N_{fs}=30$')
ax1.plot(res_4[:n_l],dep_c,'-',color='orange',linewidth=1.0,label=r'$N_{fs}=40$')
ax1.minorticks_on()
# ax1.xaxis.set_minor_locator(ticker.MultipleLocator(20000))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax1.set_xscale("log")
# ax1.set_yscale("log")
ax1.set_xlim(5,6000)
ax1.set_ylim(0.0,320)
ax1.invert_yaxis()
ax1.set_xlabel('Resistivity $(\Omega\cdot{m})$',fontsize=10)
ax1.set_ylabel('Depth (km)',fontsize=10)
ax1.tick_params( right='on',top='on', direction='out',which="major",width=0.6,length=5)
ax1.tick_params(right='on',top='on',direction='out',which="minor",width=0.4,length=2.5)
ax1.set_title(r'$\rho_{xx}$',pad=10,fontsize=12,color='black',fontweight='bold')
ax1.legend(loc=2,fontsize=8.5,borderaxespad=0.4,frameon=False,ncol=1,labelspacing=0.25)

ax2.plot(res_v,dep_r,'k-',linewidth=1.0,label='True model')
ax2.plot(res_1[n_l:],dep_c,'r-',linewidth=1.0,label='GN')
ax2.plot(res_2[n_l:],dep_c,'b-',linewidth=1.0,label='GN-FS')
ax2.plot(res_3[n_l:],dep_c,'g-',linewidth=1.0,label='ADMM-FS')
ax2.plot(res_4[n_l:],dep_c,'-',color='orange',linewidth=1.0,label=r'$N_{fs}=40$')
ax2.minorticks_on()
# ax2.xaxis.set_minor_locator(ticker.MultipleLocator(20000))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax2.set_xscale("log")
# ax2.set_yscale("log")
ax2.set_xlim(5,6000)
# ax1.set_xticks(xtick)
# ax1.set_yticks(ytick1)
ax2.set_ylim(0.0,320)
ax2.set_yticklabels(())
ax2.invert_yaxis()
ax2.set_xlabel('Resistivity $(\Omega\cdot{m})$',fontsize=10)
# ax2.grid(which="both",ls='-',lw=0.2)
ax2.tick_params( right='on',top='on', direction='out',which="major",width=0.6,length=5)
ax2.tick_params(right='on',top='on',direction='out',which="minor",width=0.4,length=2.5)
# ax2.set_title('(a)',loc='left',pad=3,fontsize=12,fontweight='bold')
ax2.set_title(r'$\rho_{yy}$',pad=10,fontsize=12,color='black',fontweight='bold')
plt.rcParams.update({'font.size': 7})

fig.savefig('Inv_results.pdf', dpi=500)
fig.show()
