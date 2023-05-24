# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:35:33 2023

@author: vjohn
"""

# %% Imports
from utils.settings import *

# %% Save path
save_path = get_save_path('FigureS6')

# %% Calibrated Rabi frequencies

vP1 = -10
vP2 = 10
P2_pwr = -5
P4_pwr = 3

fq1, fq1_, fq2, fq2_ = load_cal_rabi_freq(vP1, vP2, P2_pwr, P4_pwr)

# %% Calculate values of all possible resonances

factors = [0, 1, 2, 3, -1, -2, -3]
factors_pos = [0, 1]

f_res_list = [fq1, fq2, fq2-fq1, (fq1+fq1_+fq2+fq2_)/2]
f_res_dict = {fq1: [1, 0],
              fq2: [0, 1],
              fq2-fq1: [-1, 1],
              (fq1+fq1_+fq2+fq2_)/2: [1, 1]}

# for n in factors_pos:
#     for m in factors_pos:
#         f_res = n*fq1
#         f_res_list.append(f_res)
#         f_res_dict[f_res] = [round(n,2),round(m,2)]

#         f_res = m*fq2
#         f_res_list.append(f_res)
#         f_res_dict[f_res] = [round(n,2),round(m,2)]

#         f_res = abs(n*fq1-m*fq2)
#         f_res_list.append(f_res)
#         f_res_dict[f_res] = [round(n,2),round(-m,2)]

# f_res = (fq1+fq1_+fq2+fq2_)/2
# f_res_list.append(f_res)
# f_res_dict[f_res] = [1,1]

f_res_list.sort()
f_res_list = list(dict.fromkeys(f_res_list))  # remove duplicates


f_res_list = [item for item in f_res_list if item >
              0.8 and item < 5]  # remove small and high values

# %%

# fp4_start = 1.17
# fp4_stop = 1.35

# fp2_start = 1.4
# fp2_stop = 1.8

# fp4 = np.linspace(fp4_start, fp4_stop, 501)

# fig = plt.figure()

# n_harm = 2
# for fres in f_res_list:
#     for i in np.linspace(-n_harm,n_harm,2*n_harm+1, dtype=int):
#         for j in np.linspace(-n_harm,n_harm,2*n_harm+1, dtype=int):
#             fp4 = np.linspace(fp4_start, fp4_stop, 501)
#             if j != 0:
#                 fp2 = 1/j * (fres - i*fp4)
#                 if any(fp2 > fp2_start) and any(fp2 < fp2_stop):
#                     plt.plot(fp4, fp2, label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
#                                                                      f_res_dict[fres][1],
#                                                                      i,j
#                                                                      ))
#             else:
#                 if i != 0:
#                     fp4 = fres/i
#                     if fp4 > fp4_start and fp4 < fp4_stop:
#                         # print()
#                         plt.vlines(fp4, fp2_start, fp2_stop, color='black', label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
#                                                                           f_res_dict[fres][1],
#                                                                           i,j
#                                                                           ))

# plt.xlabel(r'$f_{p4}$ (GHz)')
# plt.ylabel(r'$f_{p2}$ (GHz)')
# # plt.legend(bbox_to_anchor=(1.01, 0.7))
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', bbox_to_anchor=(1.01, 0.7), scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

# plt.ylim(fp2_start,fp2_stop)
# plt.axis('scaled')
# plt.tight_layout()
# plt.show()

# %% Q1 dif: fq1 = fp4 - fq2

fp4_start = 2.4
fp4_stop = 3.8

fp2_start = fp4_start - fq1
fp2_stop = fp4_stop - fq1

fp4_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(4, 4))
ax_leg.set_visible(False)

n_harm = 2

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, 9))

c = 0
for fres in f_res_list:
    for i in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
        for j in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
            fp4 = np.linspace(fp4_start, fp4_stop, 501)
            fres_ref = fp4 - fq1

            n_Q1 = f_res_dict[fres][0]
            n_Q2 = f_res_dict[fres][1]
            n_P2 = j
            n_P4 = i
            if n_Q1 == 0:
                label_Q = f'{n_Q2}Q2'
            elif n_Q2 == 0:
                label_Q = f'{n_Q1}Q1'
            else:
                label_Q = f'({n_Q1}Q1, {n_Q2}Q2)'

            if n_P2 == 0:
                label_P = f'{n_P4}P4'
            elif n_P4 == 0:
                label_P = f'{n_P2}P2'
            else:
                label_P = f'({n_P2}P2, {n_P4}P4)'

            if j != 0:
                fp2 = 1/j * (fres - i*fp4)
                fp2_delta = fp2-fres_ref
                if any(fp2 > fp2_start) and any(fp2 < fp2_stop) and any(abs(fp2_delta) < fp4_delta_span/2):

                    ax.plot(fp2_delta, fp4, color=colors[c], label='{}^{}'.format(label_Q,
                                                                                  label_P
                                                                                  ))
                    c = c + 1

            else:
                if i != 0:
                    fp4 = fres/i
                    # print(fp4)
                    if fp4 > fp4_start and fp4 < fp4_stop:
                        # print(fp4)
                        ax.hlines(fp4, -fp4_delta_span/2, fp4_delta_span/2, color=colors[c], label='{}^{}'.format(label_Q,
                                                                                                                  label_P
                                                                                                                  ))
                        c = c + 1

ax2 = ax.twinx()
ax2.set_ylim(fp2_start, fp2_stop)

# DOUBLE CHECK LINES
# fp2_delta = np.linspace(-0.05, 0.05, 11)
# plt.plot(fp2_delta, fp4)


plt.xlabel(r'$\Delta f_{p2}$ (GHz)')
ax.set_ylabel(r'$f_{p4}$ (GHz)')
ax2.set_ylabel(r'$f_{p2}$ (GHz)')
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', bbox_to_anchor=(2., 1), scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

# plt.ylim(fp2_start,fp2_stop)
# plt.axis('scaled')
plt.xlim(-fp4_delta_span/2, fp4_delta_span/2)

plt.figlegend(loc='right')
plt.title('Q1^(-P2, P4)')
# plt.figlegend(loc=(1.5, 0.4), prop={'size': 6})
fig.tight_layout()
plt.show()

# %% Q2 dif: fq2 = fp4 - fq2

fp4_start = 3.7
fp4_stop = 4.4

fp2_start = fp4_start - fq2
fp2_stop = fp4_stop - fq2

fp4_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(4, 4))
ax_leg.set_visible(False)

n_harm = 3

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, 10))

c = 0
for fres in f_res_list:
    for i in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
        for j in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
            fp4 = np.linspace(fp4_start, fp4_stop, 501)
            fres_ref = fp4 - fq2

            n_Q1 = f_res_dict[fres][0]
            n_Q2 = f_res_dict[fres][1]
            n_P2 = j
            n_P4 = i
            if n_Q1 == 0:
                label_Q = f'{n_Q2}Q2'
            elif n_Q2 == 0:
                label_Q = f'{n_Q1}Q1'
            else:
                label_Q = f'({n_Q1}Q1, {n_Q2}Q2)'

            if n_P2 == 0:
                label_P = f'{n_P4}P4'
            elif n_P4 == 0:
                label_P = f'{n_P2}P2'
            else:
                label_P = f'({n_P2}P2, {n_P4}P4)'

            if j != 0:
                fp2 = 1/j * (fres - i*fp4)
                fp2_delta = fp2-fres_ref
                if any(fp2 > fp2_start) and any(fp2 < fp2_stop) and any(abs(fp2_delta) < fp4_delta_span/2):
                    ax.plot(fp2_delta, fp4, color=colors[c], label='{}^{}'.format(label_Q,
                                                                                  label_P
                                                                                  ))
                    c = c + 1

            else:
                if i != 0:
                    fp4 = fres/i
                    # print(fp4)
                    if fp4 > fp4_start and fp4 < fp4_stop:
                        # print(fp4)
                        ax.hlines(fp4, -fp4_delta_span/2, fp4_delta_span/2, color=colors[c], label='{}^{}'.format(label_Q,
                                                                                                                  label_P
                                                                                                                  ))
                        c = c + 1

ax2 = ax.twinx()
ax2.set_ylim(fp2_start, fp2_stop)

plt.xlabel(r'$\Delta f_{p2}$ (GHz)')
ax.set_ylabel(r'$f_{p4}$ (GHz)')
ax2.set_ylabel(r'$f_{p2}$ (GHz)')
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', loc='center left', bbox_to_anchor=(1.1, 0.75), scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

plt.xlim(-fp4_delta_span/2, fp4_delta_span/2)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
plt.figlegend(loc='right')
# ax.legend(loc='center left', bbox_to_anchor=(1.7, 0.4))
plt.title('Q2^(-P2, P4)')
fig.tight_layout()
plt.show()

# %% Q2 sum fq2 = fp4 + fq2

fp4_start = 1
fp4_stop = 1.65

fp2_start = fq2 - fp4_start
fp2_stop = fq2 - fp4_stop

fp2_delta_span = 0.08

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(4, 4))
ax_leg.set_visible(False)

n_harm = 2

colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0, 1, 15))

c = 0
for fres in f_res_list:
    for i in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
        for j in np.linspace(-n_harm, n_harm, 2*n_harm+1, dtype=int):
            fp4 = np.linspace(fp4_start, fp4_stop, 501)
            fres_ref = fq2 - fp4
            if j != 0:
                fp2 = 1/j * (fres - i*fp4)
                fp2_delta = fp2 - fres_ref

                n_Q1 = f_res_dict[fres][0]
                n_Q2 = f_res_dict[fres][1]
                n_P2 = j
                n_P4 = i
                if n_Q1 == 0:
                    label_Q = f'{n_Q2}Q2'
                elif n_Q2 == 0:
                    label_Q = f'{n_Q1}Q1'
                else:
                    label_Q = f'({n_Q1}Q1, {n_Q2}Q2)'

                if n_P2 == 0:
                    label_P = f'{n_P4}P4'
                elif n_P4 == 0:
                    label_P = f'{n_P2}P2'
                else:
                    label_P = f'({n_P2}P2, {n_P4}P4)'

                if fres == fq2 and i+j == 0:
                    print(i, j, fp2_delta[0], fp2_delta[1])

                if any(fp2 > fp2_stop) and any(fp2 < fp2_start) and any(abs(fp2_delta) < fp2_delta_span/2):
                    ax.plot(fp2_delta, fp4, color=colors[c], label='{}^{}'.format(label_Q,
                                                                                  label_P
                                                                                  ))
                    c = c + 1

            else:
                if i != 0:
                    fp4 = fres/i
                    # print(fp4)
                    if fp4 > fp4_start and fp4 < fp4_stop:
                        # print(fp4)
                        ax.hlines(fp4, -fp2_delta_span/2, fp2_delta_span/2, color=colors[c], label='{}^{}'.format(label_Q,
                                                                                                                  label_P
                                                                                                                  ))
                        c = c + 1

ax2 = ax.twinx()
ax2.set_ylim(fp2_start, fp2_stop)

ax.set_xlabel(r'$\Delta f_{p2}$ (GHz)')
ax.set_ylabel(r'$f_{p4}$ (GHz)')
ax2.set_ylabel(r'$f_{p2}$ (GHz)')
# lgnd = plt.legend(title='             fq1, fq2, fp4, fp2', loc='center left', bbox_to_anchor=(1.1, 0.9), scatterpoints=1, fontsize=12)
# for handle in lgnd.legendHandles:
#     handle._sizes = [100]

# plt.ylim(fp2_start,fp2_stop)
# plt.axis('scaled')
plt.xlim(-fp2_delta_span/2, fp2_delta_span/2)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
plt.title('Q2^(P2, P4)')
plt.figlegend(loc='right')
fig.tight_layout()
# ax.legend(loc='center left', bbox_to_anchor=(1.7, 0.4))
plt.show()
# %%

fp4_start = 4.0
fp4_stop = 4.15

fp2_start = 1.35
fp2_stop = 1.45

fp2_delta = np.linspace(-0.1, 0.1, 101)

fp4 = np.linspace(fp4_start, fp4_stop, 501)

fig = plt.figure()

fq = round(fq1, 2)

for i in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
    for j in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        if j != 0:
            fp2 = 1/j * (fres - i*fp4)
        if any(fp2 > fp2_start) and any(fp2 < fp2_stop):
            plt.plot(fp4, fp2, label='{}, {}, {}, {}'.format(f_res_dict[fres][0],
                                                             f_res_dict[fres][1],
                                                             i, j
                                                             ))

# %%
data_fp4 = []
data_fp2 = []

fp4_start = 4.0
fp4_stop = 4.15

fp2_start = 1.35
fp2_stop = 1.45

ij_list = []
for fp2 in np.linspace(fp2_start, fp2_stop, 201):
    for fp4 in np.linspace(fp4_start, fp4_stop, 201):
        for i in [0, -1, -2, 1, 2, 3, -3, 4, -4]:
            for j in [0, -1, -2, 1, 2, 3, -3, 4, -4]:
                f_p = round(abs(i*fp4+j*fp2), 2)
                for f_res in f_res_list:
                    if abs(f_p - f_res) < 0.0001:
                        data_fp4.append(fp4)
                        data_fp2.append(fp2)
                        ij_list.append(
                            [(i, j, f_res_dict[f_res][0], f_res_dict[f_res][1]), fp4, fp2])

df_ij = pd.DataFrame(
    ij_list, columns=['ijnm_fres', 'fp4', 'fp2']).drop_duplicates()

# %%
fig = plt.figure()

n = 0
# colors = list(mcolors.TABLEAU_COLORS.values())

for i_j_fres in df_ij.ijnm_fres.unique():
    df = df_ij.where(df_ij.ijnm_fres == i_j_fres).dropna()
    plt.scatter(df.fp4, df.fp2, 5, linestyle=':', marker='.', label=i_j_fres)
    n = n + 1
    # plt.show()

plt.xlabel(r'$f_{p4}$ (GHz)')
plt.ylabel(r'$f_{p2}$ (GHz)')
# plt.legend(bbox_to_anchor=(1.01, 0.7))
lgnd = plt.legend(title='             fp4, fp2, fq1, fq2',
                  bbox_to_anchor=(1.01, 1), scatterpoints=1, fontsize=10)
for handle in lgnd.legendHandles:
    handle._sizes = [100]

# plt.xlim(3.6,4.0)
# plt.ylim(1.0,1.3)
plt.axis('scaled')
plt.tight_layout()
fig.savefig('Figures/line_sim/fp4{}to{}_fp2{}to{}.png'.format(fp4_start,
            fp4_stop, fp2_start, fp2_stop))
plt.show()

# %%
# # df = pd.DataFrame()
# fp2 = 1.12
# fp4 = 3.79

# for fp2 in np.linspace(0, 5, 501):
#     for fp4 in np.linspace(0, 5, 501):

#         data = []
#         factors = [0,1/3,1/2,1,2,3,-1/3,-1/2,-1,-2,-3]
#         factors_pos = [0,1/3,1/2,1,2,3]


#         for n in factors:
#             for m in factors_pos:
#                 for i in [0,-1,1]:
#                     for j in [0,1]:
#                         f_res = round(abs(n*fq1+m*fq2),2)
#                         f_p = abs(i*fp2+j*fp4)
#                         data.append([round(n,2),round(m,2),f_res,
#                                      round(i,2),round(j,2),f_p,
#                                      abs(f_res-f_p)])
#                 # qdif = abs(n*fq1-m*fq2)
#                 # print('sum({},{}):     '.format(round(n,2),round(m,2)) + str(round(qsum,2)))
#                 # print('dif({},{}):     '.format(round(n,2),round(m,2)) + str(round(qdif,2)))

#         df = pd.DataFrame(data, columns=['n_fq1', 'm_fq2', 'f_res',
#                                          'i_fp2', 'j_fp4', 'f_p',
#                                          'f_res - f_p'
#                                          ]).sort_values('f_res - f_p')

#         df = df.where(df['f_res - f_p'] < 0.01).dropna()

#         fp4 = np.linspace(0, 5, 501)

#         for k in range(1,len(df)):
#             i = df.i_fp2.iloc[k]
#             j = df.j_fp4.iloc[k]

#             if i == 0:
#                 fp2 = (f_res - j*fp4)
#             else:
#                 fp2 = 1/i * (f_res - j*fp4)
#             plt.plot(fp4,fp2, color='black', lw=0.1)

# plt.xlim(0,5)
# plt.ylim(0,5)
# plt.show()


# #%%

# delta_fp2 = [-0.025, 0, +0.025]
# delta_fp2 = [-0.5, 0, +0.5]
# for k in range(1,len(df)):
#     line = [df.iloc[k].f_p-df.iloc[k].i_fp2*delta_fp2[0],
#             df.iloc[k].f_p,
#             df.iloc[k].f_p+df.iloc[k].i_fp2*delta_fp2[0]
#             ]
#     plt.plot(delta_fp2, line)

# plt.show()
