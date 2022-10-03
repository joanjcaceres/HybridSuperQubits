#%%
from qutip import *
import numpy as np
import importlib
import functions_fbq
import potential_color
from functions_fbq import *
from potential_color import *
importlib.reload(functions_fbq)
importlib.reload(potential_color)
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams.update(plt.rcParamsDefault)
#%%
EL=0.05e9
EC=40e9
EJ=20e9
EDelta = 20e9
#%%
# BLOCHNIUM
phase_list2 = np.linspace(0, 2*np.pi,51)
eval_bloch, eket_bloch = evals_flux_blochnium(phase_list2, EL,EC,EJ, N = 200, eigvals = 0)
#%%
phase_list = np.linspace(-6*np.pi,6*np.pi,300)
phi_ext = 0
pot_bloch = potential_blochnium(phase_list,phi_ext,EL = 0.05, EJ = 20)
#%%
phase_list3 = np.linspace(-6*np.pi,6*np.pi,71)
wfun0_bloch = probability_phi(EC,EL,phase_list3,eket_bloch[0][0])
wfun1_bloch = probability_phi(EC,EL,phase_list3,eket_bloch[0][1])
#%%
fig,ax = plt.subplots(2,1,figsize=(9,8),dpi=300)
ax[0].plot(phase_list/2/np.pi,pot_bloch,'k')
ax[0].plot(phase_list3/2/np.pi,wfun0_bloch*40 + eval_bloch[50,0]*1e-9,'C0', label = r'$|\Psi_0(\varphi)|^2$')
ax[0].plot(phase_list3/2/np.pi,wfun1_bloch*40 + eval_bloch[50,1]*1e-9,'C1', label = r'$|\Psi_1(\varphi)|^2$')
ax[0].plot(phase_list3/2/np.pi,eval_bloch[50,0]*1e-9 * np.ones(len(phase_list3)),'C7')
ax[0].plot(phase_list3/2/np.pi,eval_bloch[50,1]*1e-9 * np.ones(len(phase_list3)),'C7', label='Eigenenergies')
ax[0].legend(fontsize=12)
ax[0].set_xlabel(r'$\varphi/2\pi$ (rad)', fontsize=12)
ax[0].set_ylabel(r'Potential energy (GHz)', fontsize=12)

phase_list8 = np.concatenate([-phase_list2[::-1][0:-1],phase_list2])
ax[1].plot(phase_list8/2/np.pi,eval_bloch[:,:4]*1e-9)
ax[1].set_xlabel(r'$\varphi_{ext}/2\pi$ (rad)', fontsize = 12)
ax[1].set_ylabel(r'Eigenenergies (GHz)', fontsize = 12)
# plt.savefig('potential_blochnium.png')
plt.show()
#%%
# FERMIONIC-BOSONIC QUBIT
phase_list1 = np.linspace(0, 2*np.pi, 51)
#%%

phase_list0, evalsf_list0,eketsf_list0 = eigensystem_flux(0,phase_list1, EL,EC,EDelta, N = 200,eigvals = 0, interpol = False)
#%%
phase_list5, evalsf_list5,eketsf_list5 = eigensystem_flux(0.05,phase_list1, EL,EC,EDelta, N = 200,eigvals = 0, interpol = False)
# np.savetxt('phase_list5.dat',phase_list5)
# np.savetxt('evalsf_list5.dat',evalsf_list5)
#%%
phase_list20, evalsf_list20,eketsf_list20 = eigensystem_flux(0.2,phase_list1, EL,EC,EDelta, N = 200,eigvals = 0, interpol = False)

phase_list90, evalsf_list90,eketsf_list90 = eigensystem_flux(0.9,phase_list1, EL,EC,EDelta, N = 200,eigvals = 0, interpol = False)
#%%
# EIGENENERGIES
fig,ax = plt.subplots(1,4,figsize= (15,8), dpi = 300)
ax[0].plot(phase_list0/2/np.pi, evalsf_list0[:,:6]*1e-9)
ax[0].set_title(r'r=0.0',fontsize = 12)
ax[0].set_ylabel(r'Eigenenergies (GHz)',fontsize = 12)
ax[0].set_xlabel(r'$\varphi_{ext}/2\pi$ (rad)',fontsize = 12)

ax[1].plot(phase_list5/2/np.pi, evalsf_list5[:,:6]*1e-9)
ax[1].set_title(r'r=0.05',fontsize = 12)
ax[1].set_xlabel(r'$\varphi_{ext}/2\pi$ (rad)',fontsize = 12)

ax[2].plot(phase_list20/2/np.pi, evalsf_list20[:,:6]*1e-9)
ax[2].set_title(r'r=0.20',fontsize = 12)
ax[2].set_xlabel(r'$\varphi_{ext}/2\pi$ (rad)',fontsize = 12)

ax[3].plot(phase_list90/2/np.pi, evalsf_list90[:,:6]*1e-9)
ax[3].set_title(r'r=0.90',fontsize = 12)
ax[3].set_xlabel(r'$\varphi_{ext}/2\pi$ (rad)',fontsize = 12)

plt.show()
#%%
#SPIN PURITY
r_list = np.linspace(0,1,80)
#%%
evalsr_list,eketsr_list = eigensystem_r(0,r_list, EL,EC,EDelta, N = 200,eigvals = 0)
#%%
purity_r_0 = [(ptrace(eketsr_list[i][0],1)**2).tr() for i in range(len(r_list))]
purity_r_1 = [(ptrace(eketsr_list[i][1],1)**2).tr() for i in range(len(r_list))]
#%%
purity_r_0 = np.array(purity_r_0)
purity_r_1 = np.array(purity_r_1)
#%%
np.savetxt('data/purity_r_0.dat',purity_r_0)
np.savetxt('data/purity_r_1.dat',purity_r_1)
#%%
purity_r_0 = np.loadtxt('data/purity_r_0.dat')
purity_r_1 = np.loadtxt('data/purity_r_1.dat')
#%%
plt.plot(r_list,purity_r_0)
plt.plot(r_list,purity_r_1)
plt.show()
#%%
purity_phi_0 = [(ptrace(eketsf_list5[i][0],1)**2).tr() for i in range(len(phase_list1))]
purity_phi_1 = [(ptrace(eketsf_list5[i][1],1)**2).tr() for i in range(len(phase_list1))]
#%%
purity_phi_0 = np.array(purity_phi_0)
purity_phi_1 = np.array(purity_phi_1)
#%%
fig,ax = plt.subplots(1,2,figsize=(9,4),dpi = 300)
ax[0].plot(r_list,purity_r_0)
ax[0].plot(r_list,purity_r_1)
ax[0].set_title(r'$\varphi_{ext} = 0$')
ax[0].set_xlabel(r'$r$',fontsize = 12)
ax[0].set_ylabel(r'Spin purity ($\mathcal{P}$)', fontsize = 12)
ax[0].legend([r'$|0\rangle$', r'$|1\rangle$'])

ax[1].plot(phase_list1/2/np.pi,purity_phi_0)
ax[1].plot(phase_list1/2/np.pi,purity_phi_1)
ax[1].set_title(r'$r = 0.05$')
ax[1].set_xlabel(r'$\varphi_{ext}/2\pi$ (rad)',fontsize = 12)
ax[1].legend([r'$|0\rangle$', r'$|1\rangle$'])
plt.show()
#%%
# POTENTIAL FERMIONIC-BOSONIC QUBIT

phi_list=np.linspace(-6*np.pi,6*np.pi,801)
N=200
EL=0.05
EDelta=20
r=0.05
eig=np.array([(0.5*EL*phi**2+(EDelta*(np.cos(phi/2)*sigmaz()+r*np.sin(phi/2)*sigmax()))).eigenstates(eigvals=2, tol=0, maxiter=1000000) for phi in phi_list])

eigenvecp=np.array([eignevec[0,0] for eignevec in eig[:,1,0]])
eigenvecm=np.array([eignevec[1,0] for eignevec in eig[:,1,0]])

Ep=np.array(eig)[:,0,0]
Em=np.array(eig)[:,0,1]
#%%
phase_list5 = np.loadtxt('data/phase_list5.dat')
evalsf_list5 = np.loadtxt('data/evalsf_list5.dat')
#%%
EL=0.05e9
EC=40e9
EJ=20e9
EDelta = 20e9
phi_list=np.linspace(-6*np.pi,6*np.pi,70)

prob0_up = probability_phi_fbq(EC,EL,phi_list,eketsf_list5[0][0])
prob0_down = probability_phi_fbq1(EC,EL,phi_list,eketsf_list5[0][0])
prob1_up = probability_phi_fbq(EC,EL,phi_list,eketsf_list5[0][1])
prob1_down = probability_phi_fbq1(EC,EL,phi_list,eketsf_list5[0][1])
#%%
np.savetxt('data/prob0_up.dat',prob0_up)
np.savetxt('data/prob0_down.dat',prob0_down)
np.savetxt('data/prob1_up.dat',prob1_up)
np.savetxt('data/prob1_down.dat',prob1_down)
#%%
phi_list=np.linspace(-6*np.pi,6*np.pi,70)
prob2_up = probability_phi_fbq(EC,EL,phi_list,eketsf_list5[0][2])
prob2_down = probability_phi_fbq1(EC,EL,phi_list,eketsf_list5[0][2])
#%%

plt.figure(figsize=(9,4))
phi_list=np.linspace(-6*np.pi,6*np.pi,801)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

colorline(phi_list/2/np.pi, Ep, abs(eigenvecp)**2, cmap=cmap, norm=plt.Normalize(0, 1))
colorline(phi_list/2/np.pi, Em, abs(eigenvecm)**2, cmap=cmap, norm=plt.Normalize(0, 1))

plt.xlim(phi_list.min()/2/np.pi, phi_list.max()/2/np.pi)
plt.ylim(Ep.min(), Em.max())
plt.xlabel(r'$\varphi/2\pi$ (rad)', fontsize = 12)
plt.ylabel(r'Potential energy (GHz)', fontsize = 12)
# plt.title(r'r = 0.05', fontsize = 12)
blue_patch = mpatches.Patch(color='blue', label='Spin down')
red_patch = mpatches.Patch(color='red', label='Spin up')
plt.legend(handles=[red_patch, blue_patch])
plt.show()
# %%
plt.figure(figsize=(9,4))
phi_list_aux=np.linspace(-6*np.pi,6*np.pi,70)
plt.plot(phi_list_aux/2/np.pi,prob0_up + evalsf_list5[25,0]*1e-9, label = r'$|\Psi_{0,\uparrow} (\varphi)|^2$')
plt.plot(phi_list_aux/2/np.pi,prob0_down + evalsf_list5[25,0]*1e-9, label = r'$|\Psi_{0,\downarrow} (\varphi)|^2$')
plt.plot(phi_list_aux/2/np.pi,prob1_down + evalsf_list5[25,1]*1e-9, label = r'$|\Psi_{1,\downarrow} (\varphi)|^2$')

plt.plot(phi_list_aux/2/np.pi,prob1_up + evalsf_list5[25,1]*1e-9, label = r'$|\Psi_{1,\uparrow} (\varphi)|^2$')
plt.xlabel(r'$\varphi/2\pi$ (rad)', fontsize = 12)
plt.legend()
plt.show()
#%%
phase_range_pot = np.linspace(-5*np.pi, 5*np.pi, 401)
E_phi_m_0, E_phi_p_0 = potentials(0.05, phase_range_pot,0)
E_phi_m_pi, E_phi_p_pi = potentials(0.05,phase_range_pot,np.pi)
#%%
fig,ax = plt.subplots(1,1,dpi=300)
ax.plot(phase_range_pot, E_phi_m_0,phase_range_pot,E_phi_p_0)
plt.show()
#%%
phase_list3 = np.linspace(-6*np.pi,6*np.pi,71)
wfun0_bloch_pi = probability_phi(EC,EL,phase_list3,eket_bloch[25][0])
wfun1_bloch_pi = probability_phi(EC,EL,phase_list3,eket_bloch[25][1])
fig,ax = plt.subplots(1,1,figsize=(9,4),dpi=300)
ax.plot(phase_list/2/np.pi,pot_bloch_pi)
ax.plot(phase_list3/2/np.pi,wfun0_bloch_pi*40 + eval_bloch[25,0]*1e-9, label = r'$|\Psi_0(\varphi)|^2$')
ax.plot(phase_list3/2/np.pi,wfun1_bloch_pi*40 + eval_bloch[25,1]*1e-9, label = r'$|\Psi_1(\varphi)|^2$')
ax.plot(phase_list3/2/np.pi,eval_bloch[25,0]*1e-9 * np.ones(len(phase_list3)),'k')
ax.plot(phase_list3/2/np.pi,eval_bloch[25,1]*1e-9 * np.ones(len(phase_list3)),'k', label='Eigenenergies')
ax.legend(fontsize=12)
ax.set_xlabel(r'$\varphi/2\pi$ (rad)', fontsize=12)
ax.set_ylabel(r'Potential energy (MHz)', fontsize=12)
# plt.savefig('potential_blochnium.png')
plt.show()
#%%
eval_bloch[0,0]*1e-9
#%%
EL=0.05*1e9
EC=40*1e9
EDelta=20e9
#%%
EL=0.05
EDelta=20
r=0
phi_ext = 0
k =0.5*EL*xvecs**2-EDelta*np.cos((xvecs-phi_ext)/2)
#%%
xvecs = np.linspace(-6*np.pi,6*np.pi,500)
E_phi_m, E_phi_p = potentials(0.05, xvecs,0)
E_phi_m0, E_phi_p0 = potentials(0.00, xvecs,0)
fig,ax = plt.subplots(2,2,figsize=(12,6),dpi=300)
fig.tight_layout()

ax[0,0].plot(xvecs/2/np.pi,E_phi_m)
ax[0,0].plot(xvecs/2/np.pi,E_phi_p)
ax[0,0].set_ylabel(r'Potential energy',fontsize = 12)
ax[0,0].set_title(r'$r = 0.05$',fontsize = 12)
ax[0,0].grid()

ax[1,0].plot(xvecs/2/np.pi,wavefunction(xvecs,ekets_list[0][0]),label = r'$|0\otimes \uparrow \rangle$')
ax[1,0].plot(xvecs/2/np.pi,wavefunction1(xvecs,ekets_list[0][1]),label = r'$|1\otimes \downarrow \rangle$')

ax[1,0].plot(xvecs/2/np.pi,wavefunction(xvecs,ekets_list[0][1]),label = r'$|1 \otimes \uparrow \rangle $')
ax[1,0].plot(xvecs/2/np.pi,wavefunction1(xvecs,ekets_list[0][0]),label = r'$|0 \otimes \downarrow \rangle$')
ax[1,0].set_xlabel(r'$\varphi/2\pi$ (rad)',fontsize = 12)
ax[1,0].set_ylabel(r'$|\Psi(\varphi)|^2$',fontsize = 12)
ax[1,0].legend(fontsize = 12)
ax[1,0].grid()

ax[0,1].plot(xvecs/2/np.pi,E_phi_m0)
ax[0,1].plot(xvecs/2/np.pi,E_phi_p0)
# ax[0,1].set_ylabel(r'Potential energy',fontsize = 12)
ax[0,1].set_title(r'$r = 0.00$',fontsize = 12)
ax[0,1].grid()

ax[1,1].plot(xvecs/2/np.pi,wavefunction(xvecs,ekets_list_r0[0][0]),label = r'$|0\otimes \uparrow \rangle$')
ax[1,1].plot(xvecs/2/np.pi,wavefunction1(xvecs,ekets_list_r0[0][2]),label = r'$|1\otimes \downarrow \rangle$')

ax[1,1].plot(xvecs/2/np.pi,wavefunction(xvecs,ekets_list_r0[0][2]),label = r'$|1 \otimes \uparrow \rangle $')
ax[1,1].plot(xvecs/2/np.pi,wavefunction1(xvecs,ekets_list_r0[0][0]),label = r'$|0 \otimes \downarrow \rangle$')
ax[1,1].set_xlabel(r'$\varphi/2\pi$ (rad)',fontsize = 12)
# ax[1,1].set_ylabel(r'$|\Psi(\varphi)|^2$',fontsize = 12)
ax[1,1].legend(fontsize = 12)
ax[1,1].grid()
plt.show()

#%%
def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

colorline(xvecs/2/np.pi,E_phi_m, abs(eigenvecp)**2, cmap=cmap, norm=plt.Normalize(0, 1))
colorline(xvecs/2/np.pi,E_phi_m, abs(eigenvecm)**2, cmap=cmap, norm=plt.Normalize(0, 1))
plt.show() 
# %%
phase_range,evalsf_list,ekets_list = eigensystem_flux(0.05,0,2*np.pi,51, N = 150, interpol = False)
#%%
r_range,evalsr_list,eketsr_list = eigensystem_r(0.0,0,0.2,51, N = 150, interpol = False)
#%%
purity_r = np.array([(ptrace(eketsr_list[n][0],1)**2).tr() for n in range(len(eketsr_list))])
#%%
purity_phi = np.array([(ptrace(ekets_list[n][0],1)**2).tr() for n in range(len(ekets_list))])
#%%
fig,ax = plt.subplots(1,2,figsize = (9,4),dpi=300)
# fig.tight_layout()
ax[0].plot(r_range, purity_r)
ax[0].set_ylabel(r'Spin purity')
ax[0].set_xlabel(r'r')
ax[0].set_title(r'$\varphi_{ext} = 0$')
# ax[0].grid()
phase_range = np.linspace(0,2*np.pi,51)
ax[1].plot(phase_range/2/np.pi, purity_phi)
# ax[1].set_ylabel(r'Spin purity')
ax[1].set_xlabel(r'$\varphi/2\pi$ (rad)')
ax[1].set_title(r'$r = 0.05$')
# ax[1].grid()
plt.show()
#%%
phase_range_r0,evalsf_list_r0,ekets_list_r0 = eigensystem_flux(0.0,0,2*np.pi,51, N = 150, interpol = False)
#%%
phase_range_b,evalsf_list_b = evals_flux_blochnium(0,2*np.pi,51, N = 200)
#%%
phase_range_pot = np.linspace(-5*np.pi, 5*np.pi, 401)
E_phi_m_0, E_phi_p_0 = potentials(0.05, phase_range_pot,0)
E_phi_m_pi, E_phi_p_pi = potentials(0.05,phase_range_pot,np.pi)
#%%
np.savetxt('fbq_state0.dat', evalsf_list[:,0])   
np.savetxt('fbq_state1.dat', evalsf_list[:,1])
np.savetxt('fbq_state2.dat', evalsf_list[:,2])
np.savetxt('fbq_state3.dat', evalsf_list[:,3])
np.savetxt('fbq_state4.dat', evalsf_list[:,4])
np.savetxt('fbq_state5.dat', evalsf_list[:,5])
np.savetxt('fbq_state6.dat', evalsf_list[:,6])
#%%
np.savetxt('phase_range.dat',phase_range)
#%%
# EIGENVALUES OF THE FERMIONIC-BOSONIC QUBIT
#%%
phase_range,E01f_list,evalsf_list = evals_flux(0.05,0,2*np.pi,41, interpol = False)
#%%
phase_range_r,E01f_list_r,evalsf_list_r = evals_flux(0.0001,0,2*np.pi,41, interpol = False)
#%%
phase_range_b,E01f_list_b,evalsf_list_b = evals_flux_blochnium(0,2*np.pi,51, interpol = False)
# %%
r_range = np.linspace(0,0.02,20)
rho_g_r_pi_list, rho_e_r_pi_list = purity_r(r_range, 0)
#%%
fig,ax = plt.subplots(1,4,figsize=(12,5), dpi = 300)
fig.tight_layout()

for i in range(6): ax[0].plot(phase_range/2/np.pi, evalsf_list_b[:,i]*1e-9)
ax[0].set_ylabel(r'Energy (GHz)')
ax[0].set_xlabel(r'$\Phi_{ext}/2\pi$ (rad)')
# ax[0].set_ylim([-4,12])
ax[0].set_title('Blochnium')


for i in range(6): ax[1].plot(phase_range/2/np.pi, evalsf_list_r0[:,i]*1e-9)
ax[1].set_ylabel(r'Energy (GHz)')
ax[1].set_xlabel(r'$\Phi_{ext}/2\pi$ (rad)')
# ax[1].set_ylim([-4,12])
ax[1].set_title(r'$r=0.0$')

for i in range(6): ax[2].plot(phase_range/2/np.pi, evalsf_list[:,i]*1e-9)
ax[2].set_ylabel(r'Energy (GHz)')
ax[2].set_xlabel(r'$\Phi_{ext}/2\pi$ (rad)')
# ax[2].set_ylim([-4,12])
ax[2].set_title(r'$r = 0.05$')

for i in range(6): ax[3].plot(phase_range/2/np.pi, evalsf_list_r1[:,i]*1e-9)
ax[3].set_ylabel(r'Energy (GHz)')
ax[3].set_xlabel(r'$\Phi_{ext}/2\pi$ (rad)')
# ax[3].set_ylim([-4,12])
ax[3].set_title(r'$r = 0.1$')

plt.show()

# %%
