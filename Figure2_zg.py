# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:38:30 2023

@author: Zoltan Gyorgy
"""

#%% imports
from settings2 import *

#%% defining style
plt.rcParams['legend.frameon']= True

fq1=1.514  #resonance frequencies at (-10,10) mV point
fq1b=1.570
fq2=2.655
fq2b=2.714

script_dir = os.path.dirname() #<-- absolute dir the script is in
sys.path.append(script_dir)
save_path = os.path.join(script_dir, 'Figures')

start_time = '2022-07-12\\17-59-02'

end_time = start_time #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
datadir = os.path.join(script_dir, 'measurements')
datfiles, fnames = get_data_from(start_time, end_time, num = 1, rootfolder=datadir, only_complete = False) 
datfile = datfiles[0]

start_time2 = '2022-07-13\\17-27-20'

end_time2 = start_time2 #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
datfiles2, fnames2 = get_data_from(start_time2, end_time2, num = 1, rootfolder=datadir, only_complete = False) 
datfile2 = datfiles2[0]

start_time_rabi_q1dif = '2022-07-13\\15-56-21' # '2022-07-13\\15-26-45' # 
start_time_rabi_q2dif = '2022-07-13\\14-27-14'
start_time_rabi_q2sum = '2022-07-13\\14-51-17'

start_time_rabi_list = [start_time_rabi_q1dif, start_time_rabi_q2dif, start_time_rabi_q2sum]
mixing_regime = ['difference', 'difference', 'sum']
fp4_fp2 = [(2.6, 1.10064), (4.2, 1.5539), (1.4, 1.2472)]

datfile_rabi = {}

for start_time_rabi in start_time_rabi_list:
    end_time = start_time_rabi #'2021-06-24\\18-03-01' #'2021-06-10\\18-32-47'
    datfiles, fnames = get_data_from(start_time_rabi, end_time, num = 1, rootfolder=datadir, only_complete = False) 
    datfile_rabi[start_time_rabi] = datfiles

    
#%% Calibrated Rabi frequencies
with open(os.path.join(script_dir, 'measurements\config_freq_rabi.txt'), "rb") as file:
    config_freq_rabi = pickle.load(file)

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

try:
    fq2p2 = config_freq_rabi['P2'][P2_pwr]['fq2'][(vP1,vP2)]
    fq2 = fq2p2/1e9
    fq2_p2 = config_freq_rabi['P2'][P2_pwr]['fq2_'][(vP1,vP2)]
    fq2_ = fq2_p2/1e9
    fq1p4 = config_freq_rabi['P4'][P4_pwr]['fq1'][(vP1,vP2)]
    fq1 = fq1p4/1e9
except:
    print('Could not find calibrated data in database')
    fq1 = 1.51e9/1e9
    fq2 = 2.66e9/1e9

mixing_regime = 'difference'
fq = fq1
#%% Plotting
figure_size = 'small'

#figsize = {'big':(1.0*fig_size_double,1.0*fig_size_double/3), 'small':(1.0*fig_size_double ,1.0*fig_size_double/3)}

P2_frequency = datfile.sig_gen3_frequency_set.ndarray/1e9
P4_frequency = datfile.sig_gen2_frequency_set.ndarray[0,:]/1e9
P2_frequency2 = datfile2.sig_gen3_frequency_set.ndarray/1e9
P4_frequency2 = datfile2.sig_gen2_frequency_set.ndarray[0,:]/1e9

linestyles = ['-', ':', '--', '-.']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


#fig = plt.figure(figsize=(1.35*fig_size_double, 1.3*fig_size_double/3))
fig = plt.figure(figsize=(1.05*fig_size_double, 0.8*1.05*fig_size_double/3))
#originally 1.05 

gs1 = GridSpec(nrows=1, ncols=3,width_ratios=[1.07,1,1],hspace=0.3)


ax1 = fig.add_subplot(gs1[0,0])
ax2 = fig.add_subplot(gs1[0,1])
ax3= fig.add_subplot(gs1[0,2])


vmin = 0
vmax = 0.6
cm = ax1.pcolor(P4_frequency, P2_frequency, datfile.su0-datfile.su0.min(), shading='auto', cmap='hot', zorder=1, vmin=vmin, vmax=vmax)
ax1.pcolor(P4_frequency2, P2_frequency2[0:-90], datfile2.su0[0:-90]-datfile2.su0[0:-90].min(), shading='auto', cmap='hot', zorder=1, vmin=vmin, vmax=vmax)

ax1.set_ylabel(r'$f_{P2}$ [GHz]',fontsize=8)
ax1.set_xlabel(r'$f_{P4}$ [GHz]',fontsize=8)

ax1.set_ylim(bottom=0)
ax1.set_xlim(left=0)

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ymax = 4.3e9
xmax = 4.3e9
lw = 1

if figure_size == 'big':
    show_1tone = True
    show_2tone_dif = True
    show_2tone_sum = True
    show_2tone_harmonic = True
else:
    show_1tone = True
    show_2tone_dif = True
    show_2tone_sum = True
    show_2tone_harmonic = False

if show_1tone:
    ax1.hlines(fq1, xmin, xmax, lw = lw, ls=linestyles[0], color='black', label = 'fq1 = fp4 or fq1 = fp2', zorder=0)
    ax1.hlines(fq2, xmin, xmax, lw = lw, ls=linestyles[0], color='navy', label = 'fq2 = fp4 or fq2 = fp2', zorder=0)
    
    ax1.vlines(fq1, ymin, ymax, lw = lw, ls=linestyles[0], color='black', zorder=0)
    ax1.vlines(fq2, ymin, ymax, lw = lw, ls=linestyles[0], color='navy', zorder=0)
    
    ax1.vlines(fq2/2, ymin, ymax, lw = lw, ls=linestyles[1], color='navy', label = 'fq2/2 = fp2', zorder=0)


if show_2tone_dif:
    # differences
    ax1.plot([fq1,xmax],[0,xmax-fq1], lw = lw, ls=linestyles[2], color=colors[0], label = 'fq1 = |fp4 - fp2|', zorder=0)
    ax1.plot([fq2,xmax],[0,xmax-fq2], lw = lw, ls=linestyles[2], color=colors[1], label = 'fq2 = |fp4 - fp2|', zorder=0)
    
    ax1.plot([0,ymax-fq1], [fq1,ymax], lw = lw, ls=linestyles[2], color=colors[0], zorder=0)
    ax1.plot([0,ymax-fq2], [fq2,ymax], lw = lw, ls=linestyles[2], color=colors[1], zorder=0)


if show_2tone_sum:
    # sum
    # ax1.plot([0,fq1], [fq1,0], lw = lw, ls=linestyles[2], color=colors[2], label = 'fq1 = fp4 + fp2', zorder=0)
    ax1.plot([0,fq1+fq2_], [fq1+fq2_,0], lw = lw, ls=linestyles[1], color=colors[2], label = 'fq1 + fq2 = fp4 + fp2', zorder=0)
    ax1.plot([0,fq2], [fq2,0], lw = lw, ls=linestyles[2], color=colors[2], label = 'fq2 = fp4 + fp2', zorder=0)


if show_2tone_harmonic:
    ax1.plot([0,2*fq1],[fq1/2,5/2*fq1], lw = lw, ls=linestyles[1], color=colors[5], label = 'fq1 = 2*(fp4 - fp2)', zorder=0)
    
    ax1.plot([0,fq1],[fq1/2,fq1], lw = lw, ls=linestyles[3], color=colors[3], label = 'fq1 = 2*fp4 + fp2', zorder=0)
    ax1.plot([0,4*fq1],[-fq1/2,3/2*fq1], lw = lw, ls=linestyles[3], color=colors[3], label = 'fq1 = -2*fp4 + fp2', zorder=0)
      
    ax1.plot([0,fq1],[-fq1,fq1], lw = lw, ls=linestyles[2], color=colors[3], label = 'fq1 = 2*fp2 + fp4', zorder=0)
    
    ax1.plot([0,fq2],[(fq1+fq2)/2,(fq1+2*fq2)/2], lw = lw, ls=linestyles[1], color=colors[5], label = 'fq1 + fq2 = 2*fp4 + fp2', zorder=0)
    
    ax1.plot([0,fq1],[fq1/3,2/3*fq1], lw = lw, ls=linestyles[1], color=colors[3], label = 'fq1 = 3*fp4 + fp2', zorder=0)
    
    # ax1.plot([0,2*fq1],[fq1/4,5/4*fq1], lw = lw, ls=linestyles[3], color=colors[3], label = 'fq1 = 2*fp4 + fp2', zorder=0)
    
    ax1.plot([0,fq2],[fq2/2,fq2], lw = lw, ls=linestyles[3], color=colors[4], label = 'fq2 = 2*fp4 + fp2', zorder=0)
    ax1.plot([0,fq2],[-fq2,fq2], lw = lw, ls=linestyles[2], color=colors[4], label = 'fq2 = 2*fp2 + fp4', zorder=0)
    

colors = ['lightskyblue', 'purple', 'turquoise']
n = 0
for start_time_rabi in start_time_rabi_list:
    fp4 = np.round(datfile_rabi[start_time_rabi][0].metadata['station']['instruments']['sig_gen2']['parameters']['frequency']['value']/1e9, 3)
    fp2 = np.round(datfile_rabi[start_time_rabi][0].metadata['station']['instruments']['sig_gen3']['parameters']['frequency']['value']/1e9, 3)
    ax1.scatter([fp4],[fp2], s=40, zorder=1, marker='d', color=colors[n], edgecolors='black', clip_on=False)
    n = n + 1


ax1.axis('square')
ax1.set_ylim(0.9,3.5)
ax1.set_xlim(0.9,4.3)

ax1.plot([-100e-3,fq2-100e-3], [fq2,0], lw = 1, ls=linestyles[2], color='turquoise', zorder=2)
ax1.plot([+100e-3,fq2+100e-3], [fq2,0], lw = 1, ls=linestyles[2], color='turquoise', zorder=2)
ax1.plot([fq2-100e-3,2*fq2-100e-3], [0,fq2], lw = 1, ls=linestyles[2], color='violet', zorder=2)
ax1.plot([fq2+100e-3,2*fq2+100e-3], [0,fq2], lw = 1, ls=linestyles[2], color='violet', zorder=2)
ax1.plot([fq1-100e-3,4*fq1-100e-3], [0,3*fq1], lw = 1, ls=linestyles[2], color='lightskyblue', zorder=2)
ax1.plot([fq1+100e-3,4*fq1+100e-3], [0,3*fq1], lw = 1, ls=linestyles[2], color='lightskyblue', zorder=2)

cbar = fig.colorbar(cm, fraction=0.03, pad=0.02,ax=ax1)
cbar.ax.tick_params(labelsize=5,pad=0.5)
# cbar.set_label(r'$1-P_{\downdownarrows}$')

Q1=1.514  #transition frequencies
Q1b=1.570
Q2=2.655
Q2b=2.714

mono(2,Q1,ax2)      #plotting the monochromatic transitions
mono(2,Q2,ax2)
mono(2,Q1+Q2b,ax2)

mono(4,Q1,ax2)
mono(4,Q2,ax2)
mono(4,Q1+Q2b,ax2)

bichro(1,1,1,ax2)    #plotting the bichromatic transitions
bichro(1,1,2,ax2)
bichro(1,1,3,ax2)

bichro(-1,1,1,ax2)
bichro(-1,1,2,ax2)
bichro(-1,1,3,ax2)

bichro(1,-1,1,ax2)
bichro(1,-1,2,ax2)
bichro(1,-1,3,ax2)


mon4=[[0,1,1],[0,1,2],[0,1,3]]   #monochromatic transitions, C2*P2+C4*P4=Q_i, driven by P4 or P2
mon2=[[1,0,1],[1,0,2],[1,0,3]]

bikro=[[1,1,1],[1,1,2],[1,1,3],[1,-1,1],[1,-1,2],[1,-1,3],[-1,1,1],[-1,1,2],[-1,1,3]]  #similar to the monochromatic transitions

bichro_sum=[[1,1,1],[1,1,2],[1,1,3]]        #bichromatic, when the sum of the frequencies is resonant 
bichro_diff=[[1,-1,1],[1,-1,2],[1,-1,3],[-1,1,1],[-1,1,2],[-1,1,3]]   #bichromatic, when the difference is resonant 

#circles around the analysed anticrossings

x=(Q(2)-1*Q(1))/1    #intersection point of a monochromatic driven by P2 and a bichromatic 
y=Q(1)
ax2.scatter(y,x,s=68,marker='o',facecolors='none',edgecolors='red',zorder=2,) #draw circles around the analysed anticrossings

x=(Q(1)-1*Q(2))/(-1)    #intersection point of a monochromatic driven by P2 and a bichromatic 
y=Q(2)
ax2.scatter(y,x,s=68,marker='o',facecolors='none',edgecolors='red',zorder=2,)

x=(Q(2)-1*Q(3))/(-1)    #intersection point of a monochromatic driven by P2 and a bichromatic 
y=Q(3)
ax2.scatter(y,x,s=68,marker='o',facecolors='none',edgecolors='red',zorder=2,)

m=20*2/3 #markersize

for i in bichro_sum:       #intersection of a bichromatic with sum and a bichromatic with diff 
    for j in bichro_diff:
        C21=i[0]
        C41=i[1]
        Qi=Q(i[2])
        
        C22=j[0]
        C42=j[1]
        Qj=Q(j[2])
        
        x=(Qj-C22*Qi/C21)/(C42-C41*C22/C21)  #calculates the intersection point
        y=(Qj-C42*x)/C22

for j in mon4:
    for i in bikro:
        x=(Q(i[2])-i[0]*Q(j[2]))/i[1]   #intersection point of a monochromatic driven by P4 and a bichromatic 
        y=Q(j[2])
        
for j in mon2:
    for i in bikro:
        x=(Q(i[2])-i[1]*Q(j[2]))/i[0]     #intersection point of a monochromatic driven by P2 and a bichromatic 
        y=Q(j[2])
        if i[1]==1:
            ax2.scatter(y,x,s=m,marker='x',color='blue',zorder=2)

twophoton2=[1,2,3]
twophoton4=[1,2,3]

mon4=[1,2,3]
mon2=[1,2,3]
        
x=(Q(3)-1*Q(1))/1    #intersection point of a monochromatic driven by P2 and a bichromatic 
y=Q(1)
#ax1.scatter(y,x,s=m,marker='x',color='blue',zorder=2,label='strong anticrossing, size$\sim$$\Omega t$')  #we put labels 

x=intersection_mono_bi([4,1],[1,1,2])[0]
y=intersection_mono_bi([4,1],[1,1,2])[1] 
#ax1.scatter(x,y,s=m,marker='x',color='purple',zorder=3,label='strong anticrossing, size$\sim$$t^2$')    #labels 

x=intersection_mono_bi([4,1],[1,-1,1])[0]
y=intersection_mono_bi([4,1],[1,-1,1])[1]
#ax1.scatter(x,y,s=m,marker='x',color='black',label='crossing',zorder=3)    #labels 

C21=-1
C41=1
Qi=Q(1)
        
C22=1
C42=1
Qj=Q(3)
        
x=(Qj-C22*Qi/C21)/(C42-C41*C22/C21)
y=(Qj-C42*x)/C22


ax2.set_xlabel(r'$f_{P4}$ [GHz]',fontsize=8)

ax2.text(1.12,2,r'$\mathrm{Q1^{P4}}$',fontsize=9*2/3)
ax2.text(0.9,1.0,r'$\mathrm{Q2^{P2,P4}}$',fontsize=9*2/3,rotation=-45)
ax2.text(2.25,2.95,r'$\mathrm{Q2^{P4}}$',fontsize=9*2/3)
ax2.text(3.0,1.65,r'$\mathrm{Q1^{-P2,P4}}$',fontsize=9*2/3,rotation=45)

ax2.text(3.25,3.25,r'$\mathrm{(Q1+Q2\_)^{P4}}$',fontsize=9*2/3)
ax2.text(3.45,0.95,r'$\mathrm{Q2^{-P2,P4}}$',fontsize=9*2/3,rotation=45)
#ax2.arrow(3.3, 1.16, 0.25, -0.08,
#          head_width = 0.07,
#          width = 0.01,
#          ec ='black')


trikro_l=[[1,2],[2,1],[-1,2],[2,-1],[1,-2],[-2,1]]  #these are the trichromatic processes, without the specification of Qi
trikro_l=np.array(trikro_l)
trikro=[]
for i in trikro_l:
    for j in range(1,4):
        cucc=[]
        cucc=i
        cucc=np.append(cucc,j)
        #print(cucc)
        trikro.append(cucc)  #this adds the Qi term, so we will get 18 transitions, every element of trikro_l will have 3 
                             #different Qi values         

bichro2(1,1,1,ax3)    #plotting the bichromatic transitions
bichro2(1,1,2,ax3)
bichro2(1,1,3,ax3)

bichro2(-1,1,1,ax3)
bichro2(-1,1,2,ax3)
bichro2(-1,1,3,ax3)

bichro2(1,-1,1,ax3)
bichro2(1,-1,2,ax3)
bichro2(1,-1,3,ax3)

m=18*2/3 #markersize

for i in trikro:  #we plot all of the trichromatic transitions
    trichro(i,ax3)

bikro=[[1,1,1],[1,1,2],[1,1,3],[1,-1,1],[1,-1,2],[1,-1,3],[-1,1,1],[-1,1,2],[-1,1,3]]  #similar to the monochromatic transitions

#there are six different kind of resonance lines

bichro_sum=[[1,1,1],[1,1,2],[1,1,3]]        #bichromatic, when the sum of the frequencies is resonant 
bichro_diff=[[1,-1,1],[1,-1,2],[1,-1,3],[-1,1,1],[-1,1,2],[-1,1,3]]   #bichromatic, when the difference is resonant
trichro_big=[[1,-2,2],[1,-2,1],[-1,2,1],[-1,2,2],[-1,2,3]]   #trichromatic transitions with a slope of 2
trichro_small=[[2,-1,3],[2,-1,2],[2,-1,1],[-2,1,1],[-2,1,2]] #trichromatic transitions with a slope of 1/2
trichro_neg1=[[1,2,3]]                                #trichromatic transitions with a slope of -2
trichro_neg2=[[2,1,3]]                                #trichromatic transitions with a slope of -1/2

#lines=[bichro_sum,bichro_diff,trichro_big,trichro_small,trichro_neg1,trichro_neg2]  #all groups of transitions in a single list
lines=[bikro,trikro] #when we are interested only in the intersection of bicchromatic 2photon with bichro 3photon

#numbers=[0,1,2,3,4,5]                #using these numbers we generate all possible pairs of transition groups to intersect
numbers=[0,1]
pairs=[]                           #these are in the pairs 
for i in range(len(numbers)):
    for j in range(i+1, len(numbers)):
        pairs.append([numbers[i],numbers[j]])

for i in pairs:
    set1=lines[i[0]]               #from the pairs we choose two sets of transitions
    set2=lines[i[1]]
    for j in set1:                 #we take two elements, intersect 
        for k in set2:
            anti=anticrossing(j,k)  #see if it is an anticrossing
            x=intersection(j,k)[0]  #calculate intersection points
            y=intersection(j,k)[1]
            
            #if anti[0]==0:
                #ax2.scatter(x,y,s=m,marker='x',color='black',zorder=2)  #it is a crossing
            if anti[0]==1:
                if anti[1]==2:
                    if anti[3]==2:
                        ax3.scatter(x,y,s=m,marker='x',color='green',zorder=2) #second order anticrossing driven by P2
                    if anti[3]==4:
                        ax3.scatter(x,y,s=m,marker='x',color='green',zorder=2) #second order anticrossing driven by P4 
                    if anti[3]==0:
                        ax3.scatter(x,y,s=m,marker='x',color='red',zorder=2) #second order anticrossing driven by P4 
                if anti[1]==1:             #first order anticrossing, driven by P2
                    if anti[3]==2:
                        if anti[2]=='Ot':
                            ax3.scatter(x,y,s=m,marker='x',color='blue',zorder=2)
                        if anti[2]=='t2':
                            ax3.scatter(x,y,s=m,marker='x',color='blue',zorder=2)
                    else:                  #first order anticrossing, driven by P4
                        if anti[2]=='Ot':
                            ax3.scatter(x,y,s=m,marker='x',color='green',zorder=2)
                        if anti[2]=='t2':
                            ax3.scatter(x,y,s=m,marker='x',color='green',zorder=2)                        
                    

x=intersection([2,1,3],[1,1,2])[0]   
y=intersection([2,1,3],[1,1,2])[1]
ax3.scatter(x,y,s=68,marker='o',facecolors='none',edgecolors='red',zorder=2) #draw circles around the analysed anticrossings

x=intersection([-1,1,2],[-2,1,1])[0]   
y=intersection([-1,1,2],[-2,1,1])[1]
ax3.scatter(x,y,s=68,marker='o',facecolors='none',edgecolors='red',zorder=2) #draw circles around the analysed anticrossings

x=intersection([-1,2,1],[1,-1,1])[0]   
y=intersection([-1,2,1],[1,-1,1])[1]

x=intersection([2,1,3],[1,1,2])[0]  
y=intersection([2,1,3],[1,1,2])[1]
ax3.scatter(x,y,s=m,marker='x',color='blue',zorder=2,label='strong anticrossing') 

x=intersection([-1,1,2],[-2,1,1])[0] 
y=intersection([-1,1,2],[-2,1,1])[1]

x=intersection([1,-1,2],[1,-2,1])[0]  
y=intersection([1,-1,2],[1,-2,1])[1]
ax3.scatter(x,y,s=m,marker='x',color='green',zorder=2,label='weak anticrossing') 

ax3.set_xlabel(r'$f_{P4}$ [GHz]',fontsize=8)

ax3.arrow(1.75, 0.5, 0, 0.6,
          head_width = 0.07,
          width = 0.01,
          ec ='black',clip_on=False)

ax3.arrow(4.15, 0.55, 0, 0.67,
          head_width = 0.07,
          width = 0.01,
          ec ='black',clip_on=False)

ax3.text(1,0.35,r'$\mathrm{(Q1+Q2\_)^{2P2,P4}}$',fontsize=9*2/3)
ax3.text(0.35,1.7,r'$\mathrm{Q2^{P2,P4}}$',fontsize=9*2/3)
ax3.text(3.7,1.25,r'$\mathrm{Q2^{-P2,P4}}$',fontsize=9*2/3,rotation=45)
#ax2.text(3.7,0.87,r'$\mathrm{Q1^{-2P2,P4}}$',fontsize=9*2/3,rotation=26.565)
ax3.text(3.7,0.35,r'$\mathrm{Q1^{-2P2,P4}}$',fontsize=9*2/3,rotation=0)

ax2.axis('square')
ax3.axis('square')

ax2.set_ylim(0.9,3.5)
ax2.set_xlim(0.9,4.3)
ax3.set_ylim(0.9,3.5)
ax3.set_xlim(0.9,4.3)

plt.legend(loc='upper center', bbox_to_anchor=(-0.12, 1.25),
          fancybox=True, shadow=False, ncol=2)

plt.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.97,
                    top=0.95,
                    hspace=0.4)

ax1.tick_params(axis='y', pad=0.5)
ax2.tick_params(axis='y', pad=0.5)
ax3.tick_params(axis='y', pad=0.5)

ax2.text(1.6,1.2,r'$\mathrm{AC_1}$',fontsize=9*2/3)
ax2.text(2.28,1.2,r'$\mathrm{AC_2}$',fontsize=9*2/3)
ax2.text(3.83,1.6,r'$\mathrm{AC_3}$',fontsize=9*2/3)

ax3.text(0.9,1.35,r'$\mathrm{AC_4}$',fontsize=9*2/3)
ax3.text(3.4,1.2,r'$\mathrm{AC_5}$',fontsize=9*2/3)

plt.savefig('Figure2.png',format='png', dpi=300,bbox_inches='tight')
plt.savefig('Figure2.pdf',format='pdf',dpi=300,bbox_inches='tight')
plt.savefig('Figure2.svg',format='svg',bbox_inches='tight')
plt.show()
#%%
