# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:39:38 2023

@author: Zoltan Gyorgy
"""

# %% imports

from scipy.constants import physical_constants
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import savgol_filter
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# %% constants

muB = physical_constants['Bohr magneton in Hz/T'][0] / 1e9
B = 0.675
g10 = 0.17439217   # g-factor

c1 = -0.00104254
d = -0.01145871
a = 0.03582596
g20 = 0.27062503
c2 = -0.00142594
aU = a*10**(-3)
beta = 1.231-0.446
U = 2559.3078425900085
alpha = U*a*10**(-3)

Q1 = 1.514  # transition frequencies
Q1b = 1.570
Q2 = 2.655
Q2b = 2.714
GammaU=121.28400326213747

# %% Definitions
def epsP2(f):
    M12=0.446
    M22=1.231
    attenuation=np.loadtxt("attenuation.txt")
    A=attenuation[1,:]*np.sqrt(2)
    freq=attenuation[0,:]
    n=105
    A_first=A[0:n]
    A_second=A[n:len(A)]
    A_filtered1=savgol_filter(A_first,7,3)
    A_filtered2=savgol_filter(A_second,65,3)
    A_filtered1=np.array(A_filtered1)
    A_filtered2=np.array(A_filtered2)

    A_filtered=np.concatenate((A_filtered1,A_filtered2))

    A_final=interp1d(freq,A_filtered,kind='cubic')
    
    lista=A_final(f)*alpha*(M22-M12)*1000
    lista=lista.astype(np.float64)

    return lista

def epsP4(f):
    M14=0.353
    M24=0.234
    attenuation=np.loadtxt("attenuation.txt")
    A=attenuation[1,:]*np.sqrt(2)
    freq=attenuation[0,:]
    n=105
    A_first=A[0:n]
    A_second=A[n:len(A)]
    A_filtered1=savgol_filter(A_first,7,3)
    A_filtered2=savgol_filter(A_second,65,3)
    A_filtered1=np.array(A_filtered1)
    A_filtered2=np.array(A_filtered2)

    A_filtered=np.concatenate((A_filtered1,A_filtered2))

    A_final=interp1d(freq,A_filtered,kind='cubic')
    
    lista=A_final(f)*alpha*(M14-M24)*1000
    lista=lista.astype(np.float64)

    return lista

def Amplitude(f):
    attenuation=np.loadtxt("attenuation.txt")
    A=attenuation[1,:]*np.sqrt(2)
    freq=attenuation[0,:]
    n=105
    A_first=A[0:n]
    A_second=A[n:len(A)]
    A_filtered1=savgol_filter(A_first,7,3)
    A_filtered2=savgol_filter(A_second,65,3)
    A_filtered1=np.array(A_filtered1)
    A_filtered2=np.array(A_filtered2)

    A_filtered=np.concatenate((A_filtered1,A_filtered2))

    A_final=interp1d(freq,A_filtered,kind='cubic')
    
    lista=A_final(f)
    lista=lista.astype(np.float64)

    return lista


def fq1_fit(eps12,g10,c1,d,a):   #theoretical function for fQ1 
    return muB*B*(g10+c1*eps12/2)+2*d*1/(1-a**2*eps12**2)

def fq2_fit(eps12,g20,c2): #theoretical curve for the detuning dependence of fq2
    d=-0.01145871
    a=0.03582596
    return muB*B*(g20+c2*eps12/2)+2*d*1/(1-a**2*eps12**2)

def get_fq1(VP1):  # this function calculates the frequency of qubit 1 at different detunings (VP2=-VP1)

    return (muB*B*(g10-c1*VP1)+2*d*1/(1-4*a**2*VP1**2))


def get_fq1b(VP1):  # this function calculates fQ1_ frequency at different detunings (VP2=-VP1)

    return (muB*B*(g10-c1*VP1)-2*d*1/(1-4*a**2*VP1**2))


def get_fq2(VP1):  # this function calculates the frequency of qubit 2 at different detunings (VP2=-VP1)

    return (muB*B*(g20+c2*VP1)+2*d*1/(1-4*a**2*VP1**2))


def get_fq2b(VP1):  # this function calculates fQ2_ frequency at different detunings (VP2=-VP1)

    return (muB*B*(g20+c2*VP1)-2*d*1/(1-4*a**2*VP1**2))


def EP2(f):  # frequency in GHz, EP2 in ueV, calculates the electric field of plunger P2, without the lever arm
    cucc = 1.36+7.28*np.e**(-0.45*f)
    return cucc*beta*1000


def x_curve_AC5_up(y_up,b0,b1,t):
    eps=20000*alpha
    fQ2=2.655
    C=epsP2(y_up-fQ2)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(y_up-fQ2)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    a1=b0-C
    chi3=D*t**2
    a2=b1+y_up-C
    
    return (-(2*a1+a2)+np.sqrt((2*a1-a2)**2+8*chi3**2))/4

def x_curve_AC5_down(y_down,b0,b1,t):
    eps=20000*alpha
    fQ2=2.655
    C=epsP2(y_down-fQ2)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(y_down-fQ2)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    
    a1=b0-C
    chi3=D*t**2
    a2=b1+y_down-C
    
    return (-(2*a1+a2)-np.sqrt((2*a1-a2)**2+8*chi3**2))/4


def x_curve_AC3(y,x0,y0,tO):  #theoretical curve of the upper part of the attenuation, with 0 attenuation 
    eps=20000*alpha
    fQ2=2.655
    C=epsP2(y-fQ2)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(y-fQ2)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    return D**2*tO**2/(y0-y)+x0+C


def x_curve_AC1(y,x0,y0,t):  #theoretical curve of the upper part of the attenuation, with 0 attenuation 
    eps=20000*alpha
    fQ2=2.655
    C=epsP2(fQ2-y)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(fQ2-y)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    return -D**2*t**4/(y0-y-C)+x0-C


def x_curve_AC4_down(y_down,b0,b1,tO):
    
    eps=20000*alpha
    fQ2=2.655
    C=epsP2(fQ2-y_down)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(fQ2-y_down)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    
    a1=b0-C
    a2=y_down+b1
    chi3=D*tO
    
    return (2*a1+a2+np.sqrt((2*a1-a2)**2+8*chi3**2))/4

def x_curve_AC4_up(y_up,b0,b1,tO):
    
    eps=20000*alpha
    fQ2=2.655
    C=epsP2(fQ2-y_up)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(fQ2-y_up)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    
    a1=b0-C
    a2=y_up+b1
    chi3=D*tO
    
    return (2*a1+a2-np.sqrt((2*a1-a2)**2+8*chi3**2))/4

def x_curve_AC2(y,x0,y0,t,VP):  #theoretical curve of the upper part of the attenuation, with 0 attenuation 
    eps=VP*alpha
    fQ1=1.514
    C=epsP2(y-fQ1)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(y-fQ1)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    return D**2*t**4/(y0-y-C)+x0+C

def shared_objective_AC5(params, x1, x2, data1, data2):
    b0 = params[0]
    b1 = params[1]
    t = params[2]

    model1 = x_curve_AC5_up(x1,b0,b1,t)
    model2 = x_curve_AC5_down(x2,b0,b1,t)

    residuals1 = model1 - data1
    residuals2 = model2 - data2

    return np.sum(residuals1**2) + np.sum(residuals2**2)

def shared_objective_AC4(params, x1, x2, data1, data2):
    b0 = params[0]
    b1 = params[1]
    tO = params[2]

    model1 = x_curve_AC4_up(x1,b0,b1,tO)
    model2 = x_curve_AC4_down(x2,b0,b1,tO)

    residuals1 = model1 - data1
    residuals2 = model2 - data2

    return np.sum(residuals1**2) + np.sum(residuals2**2)


def plotting(x, ax, VP1, color):  # this function plots resonance lines
    y1 = 2*get_fq1(VP1)*10**9-x
    y2 = get_fq1(VP1)*10**9+get_fq2(VP1)*10**9/2-x
    y3 = get_fq1(VP1)*10**9+get_fq2(VP1)*10**9/2+get_fq1b(VP1)*10**9/2-x
    ax.plot(x*10**(-6), y1*10**(-9), color=color,
            linewidth=0.8, linestyle=(0, (3, 3)))
    ax.plot(x*10**(-6), y2*10**(-9), color=color,
            linewidth=0.8, linestyle=(0, (3, 3)))
    ax.plot(x*10**(-6), y3*10**(-9), color=color,
            linewidth=0.8, linestyle=(0, (3, 3)))


def x_1(y, x0, y0, t, VP):
    eps=2*alpha*VP
    fQ1=1.514
    C=epsP2(y-fQ1)**2*U*(U**2+3*eps**2)/(U**2-eps**2)**3*0.2417990504*GammaU
    D=epsP2(y-fQ1)*2*U*eps/(U**2-eps**2)**2*0.2417990504
    return D**2*t**4/(y0-y-C)+x0+C


def mono(P, f, ax2):  # P is either 2 or 4, f can be Q1, Q2 or Q1+Q2b
    if P == 2:  # if plunger P2 is used
        y = f  # the fp2 is constant
        x0 = 0
        x = 5
        # plots the monochromatic resonance line
        ax2.plot([x0, x], [y, y], color='brown', linewidth=1)
    else:  # if plunger P4 is used
        x = f
        y0 = 0
        y = 5
        ax2.plot([x, x], [y0, y], color='brown', linewidth=1)


def Q(i):  # function that returns the Q as a function of an index
    q = 0
    if i == 1:
        q = Q1
    if i == 2:
        q = Q2
    if i == 3:
        q = Q1+Q2b
    return q


def fp2(C2, C4, f, x):
    return f/C2-x*C4/C2  # calculates the equation of the resonance line


def fp4(C2, C4, f, y):
    return f/C4-y*C2/C4


def bichro(C2, C4, i, ax2):  # c2*fp2+c4*fp4=Q_i, Q1, Q2 and Q1+Q2_ are the three frequencies
    if i == 1:
        f = Q1
    if i == 2:
        f = Q2
    if i == 3:
        f = Q1+Q2b
    I = np.array([0, 5])
    ax2.plot(I, fp2(C2, C4, f, I), color='orange', zorder=1,
             linewidth=1)  # plot of the resonance line


def bichro2(C2, C4, i, ax3):  # c2*fp2+c4*fp4=Q_i, Q1, Q2 and Q1+Q2_ are the three frequencies
    if i == 1:  # just slightly redefine the function used before for bichromatic transitions
        f = Q1
    if i == 2:
        f = Q2
    if i == 3:
        f = Q1+Q2b
    I = np.array([0, 5])
    ax3.plot(I, fp2(C2, C4, f, I), color='orange', zorder=1,
             linewidth=1)  # plot of the resonance line


def trichro(line, ax3):  # line has the form [C2,C4,Qi], Qi has value 1,2 or 3
    C2 = line[0]
    C4 = line[1]
    Qi = Q(line[2])  # we calculate the actual Qi value

    I = np.array([0, 5])  # the interval of plotting
    I = np.array([0, 5])
    y = (Qi-C4*I)/C2  # y interval
    # plot of the resonance line
    ax3.plot(I, y, color='violet', zorder=1, linewidth=1)


# calculates the intersection point of two transitions (both are at least bichromatic)
def intersection(trans1, trans2):
    # trans1 has the form [C21,C41,Qi], trans2 [C22,C42,Qj], Qi 1,2 or 3
    C21 = trans1[0]
    C41 = trans1[1]
    # we calculate the actual Qi value using the function Q()
    Qi = Q(trans1[2])

    C22 = trans2[0]
    C42 = trans2[1]
    Qj = Q(trans2[2])

    x = (C22*Qi-C21*Qj)/(C41*C22-C42*C21)  # calculates the intersection points
    y = (C42*Qi-C41*Qj)/(C21*C42-C22*C41)
    return [x, y]


def intersection_mono_bi(mono, bi):
    # mono has two elements, first is 2 or 4, which tells us if it is driven by P2 or P4, second is Qi
    # bihas three elements, C2, C4 and Qj
    x = 0
    y = 0
    Qi = Q(mono[1])
    Qj = Q(bi[2])
    C2 = bi[0]
    C4 = bi[1]
    if mono[0] == 2:
        y = Qi
        x = (Qj-C2*y)/C4
    if mono[0] == 4:
        x = Qi
        y = (Qj-C4*x)/C2
    return [x, y]  # intersection point of a monochromatic with a bichromatic


def dominates(trans1, trans2):  # decides if transition 1 dominates transition 2
    t = 0  # t is a logical value, which tells us if trans1 dominates trans2

    a = trans1[0]  # a*fp2+b*fp4=Qi, similarly for c and d
    b = trans1[1]

    c = trans2[0]
    d = trans2[1]

    ph_nr1 = abs(a)+abs(b)  # photon number of a process
    ph_nr2 = abs(c)+abs(d)

    if ph_nr1 < ph_nr2:  # if it is a process, with less pohoton numbers, then it dominates
        t = 1

    if ph_nr1 == ph_nr2:  # if it has the same number of photons, but contains more ep2, then it dominates
        if abs(a) > abs(c):
            t = 1
    return t


# this function tells us if two transitions cross or not, it will give an answer only if the
def anticrossing(trans1, trans2):
    C21 = trans1[0]  # two transition lines intersect at some point
    C41 = trans1[1]  # trans1 has form [C21,C41,Qi], trans2 [C22,C42,Qj]
    Qi = trans1[2]

    anti = 0  # if it is anticrossing
    order = 0  # the order of the anticrossing
    hop = ''  # if it is a 1st order anticrossing, then it tells us which hopping parameters are contributing
    ep = 0  # it is 2, if the 1st order anticrossing is mediated by ep2, it is 4, if it is mediated by ep4

    C22 = trans2[0]
    C42 = trans2[1]
    Qj = trans2[2]

    # and we also calculate photon number differences
    ph_nr_diff = abs(C22-C21)+abs(C42-C41)

    # if the difference dominates both
    if dominates([C22-C21, C42-C41], [C21, C41]) and dominates([C22-C21, C42-C41], [C22, C42]):
        anti = 1  # then it is an anticrossings
        order = ph_nr_diff  # order is the photon number difference
    if Qi == Qj:  # if both processes drive the same transition, then it is not an anticrossing
        anti = 0
        order = 0
    if order == 1:  # if we have a 1st order anticrossing
        if abs(C22-C21) == 1:  # if the ep2 photon number is 1, then it is mediated by ep2
            ep = 2
        if abs(C42-C41) == 1:  # similarly to ep4
            ep = 4
    if order == 2:  # if the order is 2, then lets look at the photon number
        if abs(C22-C21) == 2:
            ep = 2
        if abs(C42-C41) == 2:
            ep = 4
    if anti == 1:  # this chooses the hopping parameters
        if Qi != Qj:
            if Qi+Qj == 3:
                hop = 't2'
            if Qi+Qj == 4:
                hop = 'Ot'
            if Qi+Qj == 5:
                hop = 'Ot'
    # return if it is anticrossing, the order, the hopping parameters and the ep2 or ep4
    return [anti, order, hop, ep]


# to not count twice the resonance points, which are both horizontal and vertical maximums
def merge_lists(x1, x2, y1, y2):
    # combine x1 and x2 and y1 and y2 into two separate lists
    x = x1 + x2
    y = y1 + y2

    # create a dictionary to keep track of unique pairs
    unique_pairs = {}

    # iterate through both lists simultaneously
    for i in range(len(x)):
        # get the current pair of x and y values
        pair = (x[i], y[i])

        # check if the pair already exists in the dictionary
        if pair in unique_pairs:
            # if it does, skip it (we only want unique pairs)
            continue
        else:
            # if it doesn't, add it to the dictionary
            unique_pairs[pair] = True

    # convert the dictionary keys back into separate lists
    x = [pair[0] for pair in unique_pairs.keys()]
    y = [pair[1] for pair in unique_pairs.keys()]

    return x, y

def t_to_O(t):
    O=np.sqrt(t**2-GammaU)
    return O

def ueV_to_MHz(x):  #convert ueV to MHz 
    return x*241799.0504*1000*10**(-6)

def chi3_AC2(t,eps12,f):
    eps=alpha*eps12
    return abs(epsP2(f)*2*U*eps/(U**2-eps**2)**2*t**2)

def chi3_fit(eps12,t):
    f=1.1
    eps=alpha*eps12
    return abs(epsP2(f)*2*U*eps/(U**2-eps**2)**2*t**2)
