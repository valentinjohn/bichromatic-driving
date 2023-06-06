# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:39:38 2023

@author: Zoltan Gyorgy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import pickle
import sys
import os
from utils.notebook_tools import get_data_from
from scipy.constants import physical_constants

fig_size_single = 3.37
fig_size_double = 6.69

muB=physical_constants['Bohr magneton in Hz/T'][0] / 1e9  #Bohr magneton in GHz 
B=0.675                             #magnetic field in Tesla 
g10=0.17439208                      #g-factor of qubit 1 
c1=-0.00104254      #g-factor modulation, a1-b1, 1/mV                   
d=-0.01145871       #\frac{\Omega^2-t^2}{U} term in GHz 
a=0.03582596        #ratio of the lever-arm and the charging energy, 1/meV 
g20=0.27062489      #g-factor of qubit 2
c2=-0.00142594      #g-factor modulation term 
aU=a*10**(-3)       #ratio of the lever-arm and the charging energy, 1/ueV
beta=1.231-0.446     #virtual plunger matrix elements of P2
t_avg=13.3721499     #average t hopping parameter in ueV 
O_avg=7.584879       #average \Omega hopping parameter in ueV
U=2559.3078425900085 #Charging energy in ueV 
alpha=U*a*10**(-3)   #lever-arm 

def get_fq1(VP1):   #this function calculates the frequency of qubit 1 at different detunings (VP2=-VP1)

    return (muB*B*(g10-c1*VP1)+2*d*1/(1-4*a**2*VP1**2))

def get_fq1b(VP1):  #this function calculates fQ1_ frequency at different detunings (VP2=-VP1)

    return (muB*B*(g10-c1*VP1)-2*d*1/(1-4*a**2*VP1**2))

def get_fq2(VP1):   #this function calculates the frequency of qubit 2 at different detunings (VP2=-VP1)

    return (muB*B*(g20+c2*VP1)+2*d*1/(1-4*a**2*VP1**2))

def get_fq2b(VP1):  #this function calculates fQ2_ frequency at different detunings (VP2=-VP1)

    return (muB*B*(g20+c2*VP1)-2*d*1/(1-4*a**2*VP1**2))

def EP2(f): #frequency in GHz, EP2 in ueV, calculates the electric field of plunger P2, without the lever arm  
    cucc=1.36+7.28*np.e**(-0.45*f)
    return cucc*beta*1000 

def curve_AC5(x,b0,b1,chi3):
    return (chi3**2-(2*b0+b1)*x-b0*b1-2*x**2)/(x+b0)

def curve_AC3(x,x0,y0,chi3):
    return y0-chi3**2/(x-x0)

def curve_AC1(x,x0,y0,chi3):
    return y0+chi3**2/(x-x0)

def curve_AC4(x,b0,b1,chi3):
    return (chi3**2+(2*b0+b1)*x-b0*b1-2*x**2)/(b0-x)

def chi3_AC5(freq):
    VP=10000
    return 2*EP2(freq)*2*VP*t_avg*t_avg*aU**2/(U*(1-aU**2*(2*VP)**2)**2)*241799.050402293*10**9*10**(-6)

def chi3_AC3(freq):
    VP=10000
    return 2*EP2(freq)*2*VP*t_avg*O_avg*aU**2/(U*(1-aU**2*(2*VP)**2)**2)*241799.050402293*10**9*10**(-6)

def chi3_AC1(freq):
    VP=10000
    return 2*EP2(1.141)*2*VP*t_avg*t_avg*aU**2/(U*(1-aU**2*(2*VP)**2)**2)*241799.050402293*10**9*10**(-6)

def chi3_AC4(freq):
    VP=10000
    return 2*EP2(1.57)*2*VP*t_avg*O_avg*aU**2/(U*(1-aU**2*(2*VP)**2)**2)*241799.050402293*10**9*10**(-6)

def chi3_AC2(VP1,ratio):
    return 2*EP2(1.141)*2*VP1*ratio*aU**2/(1*(1-aU**2*(2*VP1)**2)**2)

def chi3_AC2_eps(VP1):
    chi3=2*EP2(1.141)*2*VP1*t_avg**2*aU**2/(U*(1-aU**2*(2*VP1)**2)**2)
    chi3=chi3*241799.050402293*10**(-6)*10**(9)
    return chi3

def plotting(x,ax,VP1):          #this function plots resonance lines 
    y1=2*get_fq1(VP1)*10**9-x
    y2=get_fq1(VP1)*10**9+get_fq2(VP1)*10**9/2-x
    y3=get_fq1(VP1)*10**9+get_fq2(VP1)*10**9/2+get_fq1b(VP1)*10**9/2-x
    ax.plot(x*10**(-6),y1*10**(-9),color='cyan',linewidth=0.8,linestyle=(0, (3,3)))
    ax.plot(x*10**(-6),y2*10**(-9),color='cyan',linewidth=0.8,linestyle=(0, (3,3)))
    ax.plot(x*10**(-6),y3*10**(-9),color='cyan',linewidth=0.8,linestyle=(0, (3,3)))
    
def x_1(y_u,y0,x0,A,chi3,fq1,fc):
    return (x0*(y_u-y0)+pow(10,0*(y_u-fq1-fc))*(A*(x0+y_u-y0+A*pow(10,0*(y_u-fq1-fc)))-chi3**2))/(y_u-y0+A*10**(0*(y_u-fq1-fc)))

def x_2(y_d,y0,x0,A,chi3,fq1,fc):
    b=5.0*10**(-9)
    return (x0*(y_d-y0)+pow(10,b*(y_d-fq1-fc))*(A*(x0+y_d-y0+A*pow(10,b*(y_d-fq1-fc)))-chi3**2))/(y_d-y0+A*10**(b*(y_d-fq1-fc)))

Q1=1.514  #transition frequencies
Q1b=1.570
Q2=2.655
Q2b=2.714

def mono(P,f,ax2):         #P is either 2 or 4, f can be Q1, Q2 or Q1+Q2b
    if P==2:  #if plunger P2 is used
        y=f  #the fp2 is constant 
        x0=0
        x=5
        ax2.plot([x0,x],[y,y],color='brown',linewidth=1)  #plots the monochromatic resonance line
    else:       #if plunger P4 is used
        x=f
        y0=0
        y=5
        ax2.plot([x,x],[y0,y],color='brown',linewidth=1)
            
def Q(i):    #function that returns the Q as a function of an index 
    q=0
    if i==1:
        q=Q1
    if i==2:
        q=Q2
    if i==3:
        q=Q1+Q2b
    return q    

def fp2(C2,C4,f,x):
    return f/C2-x*C4/C2  #calculates the equation of the resonance line

def fp4(C2,C4,f,y):
    return f/C4-y*C2/C4

def bichro(C2,C4,i,ax2):   #c2*fp2+c4*fp4=Q_i, Q1, Q2 and Q1+Q2_ are the three frequencies
    if i==1:
        f=Q1
    if i==2:
        f=Q2
    if i==3:
        f=Q1+Q2b
    I=np.array([0,5])
    ax2.plot(I,fp2(C2,C4,f,I),color='orange',zorder=1,linewidth=1) #plot of the resonance line
    
def bichro2(C2,C4,i,ax3):   #c2*fp2+c4*fp4=Q_i, Q1, Q2 and Q1+Q2_ are the three frequencies
    if i==1:            #just slightly redefine the function used before for bichromatic transitions 
        f=Q1
    if i==2:
        f=Q2
    if i==3:
        f=Q1+Q2b
    I=np.array([0,5])
    ax3.plot(I,fp2(C2,C4,f,I),color='orange',zorder=1,linewidth=1) #plot of the resonance line
    
def trichro(line,ax3):  #line has the form [C2,C4,Qi], Qi has value 1,2 or 3
    C2=line[0]
    C4=line[1]
    Qi=Q(line[2])   #we calculate the actual Qi value
    
    I=np.array([0,5])  #the interval of plotting
    I=np.array([0,5])
    y=(Qi-C4*I)/C2  #y interval
    ax3.plot(I,y,color='violet',zorder=1,linewidth=1) #plot of the resonance line

def intersection(trans1,trans2):  #calculates the intersection point of two transitions (both are at least bichromatic)
    C21=trans1[0]                 #trans1 has the form [C21,C41,Qi], trans2 [C22,C42,Qj], Qi 1,2 or 3
    C41=trans1[1]
    Qi=Q(trans1[2])               #we calculate the actual Qi value using the function Q()
    
    C22=trans2[0]
    C42=trans2[1]
    Qj=Q(trans2[2])
    
    x=(C22*Qi-C21*Qj)/(C41*C22-C42*C21)   #calculates the intersection points
    y=(C42*Qi-C41*Qj)/(C21*C42-C22*C41)
    return [x,y]

def intersection_mono_bi(mono,bi):    
    #mono has two elements, first is 2 or 4, which tells us if it is driven by P2 or P4, second is Qi
    #bihas three elements, C2, C4 and Qj
    x=0
    y=0
    Qi=Q(mono[1])
    Qj=Q(bi[2])
    C2=bi[0]
    C4=bi[1]
    if mono[0]==2:
        y=Qi
        x=(Qj-C2*y)/C4
    if mono[0]==4:
        x=Qi
        y=(Qj-C4*x)/C2
    return [x,y]   #intersection point of a monochromatic with a bichromatic 

def dominates(trans1,trans2): #decides if transition 1 dominates transition 2
    t=0          #t is a logical value, which tells us if trans1 dominates trans2
    
    a=trans1[0]  #a*fp2+b*fp4=Qi, similarly for c and d
    b=trans1[1]
    
    c=trans2[0]
    d=trans2[1]
    
    ph_nr1=abs(a)+abs(b)   #photon number of a process 
    ph_nr2=abs(c)+abs(d)
    
    if ph_nr1<ph_nr2:      #if it is a process, with less pohoton numbers, then it dominates 
        t=1
    
    if ph_nr1==ph_nr2:     #if it has the same number of photons, but contains more ep2, then it dominates
        if abs(a)>abs(c):
            t=1
    return t

def anticrossing(trans1,trans2):   #this function tells us if two transitions cross or not, it will give an answer only if the 
    C21=trans1[0]                  #two transition lines intersect at some point 
    C41=trans1[1]                  #trans1 has form [C21,C41,Qi], trans2 [C22,C42,Qj]
    Qi=trans1[2]
    
    anti=0        #if it is anticrossing 
    order=0       #the order of the anticrossing 
    hop=''        #if it is a 1st order anticrossing, then it tells us which hopping parameters are contributing 
    ep=0          #it is 2, if the 1st order anticrossing is mediated by ep2, it is 4, if it is mediated by ep4
    
    C22=trans2[0]
    C42=trans2[1]
    Qj=trans2[2]
    
    ph_nr_diff=abs(C22-C21)+abs(C42-C41)  #and we also calculate photon number differences 
    
    if dominates([C22-C21,C42-C41],[C21,C41]) and dominates([C22-C21,C42-C41],[C22,C42]): #if the difference dominates both
        anti=1                                                                            #then it is an anticrossings
        order=ph_nr_diff                                                                  #order is the photon number difference
    if Qi==Qj:       #if both processes drive the same transition, then it is not an anticrossing
        anti=0
        order=0
    if order==1:             #if we have a 1st order anticrossing
        if abs(C22-C21)==1:  #if the ep2 photon number is 1, then it is mediated by ep2   
            ep=2
        if abs(C42-C41)==1:  #similarly to ep4
            ep=4
    if order==2:             #if the order is 2, then lets look at the photon number 
        if abs(C22-C21)==2:
            ep=2
        if abs(C42-C41)==2:
            ep=4
    if anti==1:              #this chooses the hopping parameters 
        if Qi!=Qj:
            if Qi+Qj==3:
                hop='t2'
            if Qi+Qj==4:
                hop='Ot'
            if Qi+Qj==5:
                hop='Ot'
    return [anti,order,hop,ep]    #return if it is anticrossing, the order, the hopping parameters and the ep2 or ep4

#%% defining style
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.sans-serif']= 'Arial'
plt.rcParams["figure.figsize"] = (fig_size_single, 3)  #it should be 3 
plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.default'] = 'it' #'regular'

plt.rcParams['legend.frameon']= True
plt.rcParams['legend.fontsize']= 'small'
plt.rcParams['legend.scatterpoints']= 1
plt.rcParams['axes.labelpad'] = 4 #-2