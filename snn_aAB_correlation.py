# %%
import torch
import matplotlib.pyplot as plt
# snn_module
import centipede as ctp
dtype = torch.float64
device = torch.device("cpu")
# %%
sim_name = 'aAB_correlation'
sim_num = 1

training = False
# neuron parameters

nb_neurons = 110
tau = 0.010  # [s]

# generate indices for all neurons and split into three populations
idcs_neurons = torch.arange(nb_neurons)
nb_ct = 10
nb_a = 50
nb_b = 50 # all-to-all
pop_ct = idcs_neurons < nb_ct
pop_a = (idcs_neurons >= nb_ct) & (idcs_neurons < nb_ct + nb_a)
pop_b = idcs_neurons >= (nb_a + nb_ct)

# simulation parameters
dt = 0.0001  # [s]
beta = (1 - dt / tau)

# input signal parameters
r_ext = 3000
sigma_s = 0.0025 # s
# %%
# weights
w_rec = torch.zeros((nb_neurons, nb_neurons), dtype=dtype)

p_ff = torch.tensor([0.5], dtype=dtype)
str_ff = torch.tensor([0.03], dtype=dtype)
ab_str = (torch.rand((nb_a*nb_b), dtype=dtype) < p_ff)*(str_ff/p_ff)
aa_str = 0.00001 #0.000001 #0.018 max
bb_str = 0.0
ba_str = 0.0
ct_str = 0.9 # max 0.1 
cta_str = 0.001 #0.004

w_rec[pop_ct.unsqueeze(-1) * pop_ct] = ct_str # ct -> ct
w_rec[pop_a.unsqueeze(-1) * pop_ct] = cta_str # ct -> A
w_rec[pop_b.unsqueeze(-1) * pop_a] = ab_str #ab_str
w_rec[pop_a.unsqueeze(-1) * pop_a] = aa_str  # A -> A
w_rec[pop_b.unsqueeze(-1) * pop_b] = bb_str  # B -> B
w_rec[pop_a.unsqueeze(-1) * pop_b] = ba_str  # B -> A


# %%
# input generation
nb_flicker = 500  # flicker values  
f_flicker = 100.0  # [Hz]
r_ext = 3000.0  # [Hz] - the higher, the less irregular is external input
val_min = 0.8  # [unitless, relative to spike threshold 1 with const input]
val_max = 1.1  # [unitless, relative to spike threshold 1 with const input]

dt_flicker = 1.0 / f_flicker
nb_timesteps_flickerstep = int(dt_flicker / dt)
nb_steps = nb_flicker * nb_timesteps_flickerstep
t_max = nb_steps * dt
print(f"max time: {t_max}")

nb_vals = 10
aa_max = 0.012 #0.018
ab_max = 0.05
cta_max = 0.015 #0.01 #0.015
ct_max = 0.11
aa_vals = torch.linspace(0.0, aa_max, nb_vals)
ab_vals = torch.linspace(0.0, ab_max, nb_vals)
cta_vals = torch.linspace(0.0, cta_max, nb_vals)
ct_vals = torch.linspace(0.0, ct_max, nb_vals)
frs = torch.zeros((nb_vals, 2, 3)) # dim1: att/nat, dim2: a, b, ct
cfs = torch.zeros((nb_vals, 2, 3)) # dim1: att/nat, dim2: a, b, ab
chis = torch.zeros((nb_vals, 2, 3)) # dim1: att/nat, dim2: a, b, ct

# %%
for i in range(nb_vals):
    # weights
    print(f'iter {i}, aa_str: {aa_vals[i]:.5f}')
    #print(f'iter {i}, cta_str: {cta_vals[i]:.5f}')
    #print(f'iter {i}, ct_str: {ct_vals[i]:.5f}')
    w_rec = torch.zeros((nb_neurons, nb_neurons), dtype=dtype)
    
    ab_str = (torch.rand((nb_a*nb_b), dtype=dtype) < p_ff)*(str_ff/p_ff)
    aa_str = aa_vals[i] #aa_vals[i] #0.001
    bb_str = 0.0
    ba_str = 0.0
    ct_str = 0.09 #ct_vals[i] #0.09
    cta_str = 0.004 #cta_vals[i] #0.004

    w_rec[pop_ct.unsqueeze(-1) * pop_ct] = ct_str # ct -> ct
    w_rec[pop_a.unsqueeze(-1) * pop_ct] = cta_str # ct -> A
    w_rec[pop_b.unsqueeze(-1) * pop_a] = ab_str #ab_str
    w_rec[pop_a.unsqueeze(-1) * pop_a] = aa_str  # A -> A
    w_rec[pop_b.unsqueeze(-1) * pop_b] = bb_str  # B -> B
    w_rec[pop_a.unsqueeze(-1) * pop_b] = ba_str  # B -> A

    # input
    step_signal = ctp.generate_step_signal(nb_steps, dt, n=dt_flicker, val_min=val_min, val_max=val_max)
    lambdah = dt * r_ext * step_signal
    # att is on
    tmp = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah)
    tmp[pop_b, :] = 0  # input signal goes only to pop A and ct
    tmp_norm = 1 / (r_ext * tau)  # tmp_norm = 1 / (r_ext * dt) * (dt / tau)
    neur_input = tmp * tmp_norm  # normalize properly to keep avg input indep. of r_ext

    mem0 = torch.rand((nb_neurons,), device=device, dtype=dtype)
    spk0 = torch.zeros((nb_neurons,), device=device, dtype=dtype)
    # run simulation
    mem_rec, spk_rec = ctp.sim_snn(neur_input, w_rec, beta, mem_init=mem0, spk_init=spk0, training=False)
    
    # compute metrices
    fr_ct = spk_rec[pop_ct, :].mean() / dt  
    fr_a = spk_rec[pop_a, :].mean() / dt 
    fr_hz_a = spk_rec[pop_a, :].mean(dim=0) / dt
    av_fr_a = ctp.compute_firing_rate(spk_rec[pop_a, :], dt, sigma_s)
    fr_b = spk_rec[pop_b, :].mean() / dt
    fr_hz_b = spk_rec[pop_b, :].mean(dim=0) / dt
    av_fr_b = ctp.compute_firing_rate(spk_rec[pop_b, :], dt, sigma_s)
    cf_b = torch.corrcoef(torch.stack((av_fr_b, step_signal)))[0, 1]
    cf_a = torch.corrcoef(torch.stack((av_fr_a, step_signal)))[0, 1]
    cf_ab = torch.corrcoef(torch.stack((av_fr_a, av_fr_b)))[0, 1]
    print(f'a: {fr_a.item():.2f}, b: {fr_b.item():.2f}, ct: {fr_ct.item():.2f}')
    print(f'cfs: b {cf_b.item():.4f}, a {cf_a.item():.4f}, ab: {cf_ab.item():.4f}')
    cspks = ctp.box_convolve_spikes(spk_rec)
    chi_a = ctp.compute_synchrony(cspks[pop_a, :])
    chi_b = ctp.compute_synchrony(cspks[pop_b, :])
    chi_ct = ctp.compute_synchrony(cspks[pop_ct, :])
    frs[i, 0] = torch.tensor([fr_a, fr_b, fr_ct])
    cfs[i, 0] = torch.tensor([cf_a, cf_b, cf_ab])
    chis[i, 0] = torch.tensor([chi_a, chi_b, chi_ct])

    # att is off
    tmp = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah)
    tmp[pop_b, :] = 0  # input signal goes only to pop A
    tmp[pop_ct, :] = 0 # NO ATTENTION
    tmp_norm = 1 / (r_ext * tau)  # tmp_norm = 1 / (r_ext * dt) * (dt / tau)
    neur_input = tmp * tmp_norm  # normalize properly to keep avg input indep. of r_ext

    mem0 = torch.rand((nb_neurons,), device=device, dtype=dtype)
    spk0 = torch.zeros((nb_neurons,), device=device, dtype=dtype)
    # run simulation
    mem_rec, spk_rec = ctp.sim_snn(neur_input, w_rec, beta, mem_init=mem0, spk_init=spk0, training=False)
    
    # compute metrices
    fr_ct = spk_rec[pop_ct, :].mean() / dt 
    fr_a = spk_rec[pop_a, :].mean() / dt 
    fr_hz_a = spk_rec[pop_a, :].mean(dim=0) / dt
    av_fr_a = ctp.compute_firing_rate(spk_rec[pop_a, :], dt, sigma_s)
    fr_b = spk_rec[pop_b, :].mean() / dt
    fr_hz_b = spk_rec[pop_b, :].mean(dim=0) / dt
    av_fr_b = ctp.compute_firing_rate(spk_rec[pop_b, :], dt, sigma_s)
    cf_b = torch.corrcoef(torch.stack((av_fr_b, step_signal)))[0, 1]
    cf_a = torch.corrcoef(torch.stack((av_fr_a, step_signal)))[0, 1]
    cf_ab = torch.corrcoef(torch.stack((av_fr_a, av_fr_b)))[0, 1]
    print(f'a: {fr_a.item():.2f}, b: {fr_b.item():.2f}, ct: {fr_ct.item():.2f}')
    print(f'cfs: b {cf_b.item():.4f}, a {cf_a.item():.4f}, ab: {cf_ab.item():.4f}')
    cspks = ctp.box_convolve_spikes(spk_rec)
    chi_a = ctp.compute_synchrony(cspks[pop_a, :])
    chi_b = ctp.compute_synchrony(cspks[pop_b, :])
    chi_ct = ctp.compute_synchrony(cspks[pop_ct, :])
    frs[i, 1] = torch.tensor([fr_a, fr_b, fr_ct])
    cfs[i, 1] = torch.tensor([cf_a, cf_b, cf_ab])
    chis[i, 1] = torch.tensor([chi_a, chi_b, chi_ct])   

cfs = torch.nan_to_num(cfs, nan=0.0)
chis = torch.nan_to_num(chis, nan=0.0)
frs = torch.nan_to_num(frs, nan=0.0)
# %%
# %%
is_savefig = False #False #True
x_axis = aa_vals #aa_vals 
#x_axis = cta_vals 
#x_axis = ct_vals
xlbl = 'Recurrent coupling strength'
#xlbl = 'Control to Sender coupling strength'
#xlbl = 'Inter Control coupling strength'
#cta_str = 'vary'
#ct_str = 'vary'
aa_str='vary'
title_lbl = f'AB={str_ff.item():.3f}, cta={cta_str}, ct={ct_str}, AA={aa_str}'

plt.figure(figsize=(9, 4))
plt.plot(x_axis, frs[:, 1, 0], label=r"$r_A$ non-att", color='navy')
plt.plot(x_axis, frs[:, 1, 1], '--', label=r"$r_B$ non-att", color='navy')
plt.plot(x_axis, frs[:, 0, 0], label=r"$r_A$ att", color='crimson')
plt.plot(x_axis, frs[:, 0, 1], '--', label=r"$r_B$ att", color='crimson')

plt.title(title_lbl)
plt.xlabel(xlbl)
plt.ylabel(r'Population firing rate $r$ [Hz]')

plt.grid()
plt.legend()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'fr')
#plt.show()
plt.figure(figsize=(9, 4))
plt.plot(x_axis, cfs[:, 1, 0], label=r"$C(f(x), r_A)$ non-att", color='navy')
plt.plot(x_axis, cfs[:, 0, 0], label=r"$C(f(x), r_A)$ att", color='crimson')
plt.title(title_lbl)
plt.xlabel(xlbl)
plt.ylabel(r'Correlation')
plt.grid()
plt.legend()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'sig_A')
#plt.show()
plt.figure(figsize=(9, 4))
plt.plot(x_axis, cfs[:, 1, 1], label=r"$C(f(x), r_B)$ non-att", color='navy')
plt.plot(x_axis, cfs[:, 0, 1], label=r"$C(f(x), r_B)$ att", color='crimson')
plt.title(title_lbl)
plt.xlabel(xlbl)
plt.ylabel(r'Correlation')
plt.grid()
plt.legend()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'sig_C')
#plt.show()
plt.figure(figsize=(9, 4))
plt.plot(x_axis, chis[:, 1, 0], label=r"$\chi^2_A$ non-att", color='navy')
plt.plot(x_axis, chis[:, 1, 1], '--', label=r"$\chi^2_B$ non-att", color='navy')
plt.plot(x_axis, chis[:, 0, 0], label=r"$\chi^2_A$ att", color='crimson')
plt.plot(x_axis, chis[:, 0, 1], '--',  label=r"$\chi^2_B$ att", color='crimson')

plt.title(title_lbl)
plt.xlabel(xlbl)
plt.ylabel(r'Synchrony $\chi^2$')
plt.grid()
plt.legend()
#plt.show()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'chi')
plt.figure(figsize=(9, 4))
plt.plot(x_axis, cfs[:, 1, 2], label=r"$C(r_A, r_B)$ non-att", color='navy')
plt.plot(x_axis, cfs[:, 0, 2], label=r"$C(r_A, r_B)$ att", color='crimson')

plt.title(title_lbl)
plt.xlabel(xlbl)
plt.ylabel(r'Correlation')
plt.grid()
plt.legend()
#plt.show()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'A_C')
# %%
