# %%
import torch
import matplotlib.pyplot as plt
import snn_module as ctp
import matplotlib.cm as clmps
dtype = torch.float64
device = torch.device("cpu")
# %%
training = False
# neuron parameters
nb_neurons = 100
nb_a = 50
nb_b = 50 
tau = 0.010  # [s]
# generate indices for all neurons and split into two populations
idcs_neurons = torch.arange(nb_neurons)
pop_a = idcs_neurons < (nb_neurons // 2)
pop_b = ~pop_a
# simulation parameters
dt = 0.0001  # [s]
beta = (1 - dt / tau)
# input signal parameters
r_ext = 3000
sigma_s = 0.0025 # s
# weights
w_rec = torch.zeros((nb_neurons, nb_neurons), dtype=dtype)
p_ff = 0.5
p_ff = torch.tensor([0.5], dtype=dtype)
str_ff = torch.tensor([0.030], dtype=dtype)
ab_str = (torch.rand((nb_a*nb_b), dtype=dtype) < p_ff)*(str_ff/p_ff)
aa_str = 0.0 # max 0.018
bb_str = 0.0
ba_str = 0.0
w_rec[pop_b.unsqueeze(-1) * pop_a] = ab_str
w_rec[pop_a.unsqueeze(-1) * pop_a] = aa_str  # A -> A
w_rec[pop_b.unsqueeze(-1) * pop_b] = bb_str  # B -> B
w_rec[pop_a.unsqueeze(-1) * pop_b] = ba_str  # B -> A
plt.imshow(w_rec)
plt.colorbar()

# %%
# input generation
nb_flicker = 500 #0  # flicker values  # 10 for FR target
f_flicker = 100.0  # [Hz]
r_ext = 3000.0  # [Hz] - the higher, the less irregular is external input
val_min = 0.8  # [unitless, relative to spike threshold 1 with const input]
val_max = 1.0  # [unitless, relative to spike threshold 1 with const input]

dt_flicker = 1.0 / f_flicker
nb_timesteps_flickerstep = int(dt_flicker / dt)
nb_steps = nb_flicker * nb_timesteps_flickerstep
t_max = nb_steps * dt
print(f"max time: {t_max}")
t = dt * torch.arange(nb_steps, device=device, dtype=dtype)

# %%
nb_vals = 30
aa_max = 0.012
aa_vals = torch.linspace(0, aa_max, nb_vals)
frs = torch.zeros((nb_vals, 2))
cfs = torch.zeros((nb_vals, 2))
chis = torch.zeros((nb_vals, 2))

for i in range(nb_vals):
    # weights
    print(f'iter {i}, aa_str: {aa_vals[i]:.3f}')
    w_rec = torch.zeros((nb_neurons, nb_neurons), dtype=dtype)
    w_rec[pop_a.unsqueeze(-1) * pop_a] = aa_vals[i] #A -> A
    w_rec[pop_b.unsqueeze(-1) * pop_b] = bb_str  # B -> B
    w_rec[pop_a.unsqueeze(-1) * pop_b] = ba_str  # B -> A
    w_rec[pop_b.unsqueeze(-1) * pop_a] = ab_str # A -> B
    # input
    step_signal = ctp.generate_step_signal(nb_steps, dt, n=dt_flicker, val_min=val_min, val_max=val_max)
    lambdah = dt * r_ext * step_signal
    tmp = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah)
    tmp[pop_b, :] = 0  # input signal goes only to pop A
    tmp_norm = 1 / (r_ext * tau)  # tmp_norm = 1 / (r_ext * dt) * (dt / tau)
    neur_input = tmp * tmp_norm  # normalize properly to keep avg input indep. of r_ext
    mem0 = torch.rand((nb_neurons,), device=device, dtype=dtype)
    spk0 = torch.zeros((nb_neurons,), device=device, dtype=dtype)
    # run simulation
    mem_rec, spk_rec = ctp.sim_snn(neur_input, w_rec, beta, mem_init=mem0, spk_init=spk0, training=False)
    # compute metrices
    fr_a = spk_rec[pop_a, :].mean() / dt 
    fr_hz_a = spk_rec[pop_a, :].mean(dim=0) / dt
    av_fr_a = ctp.compute_firing_rate(spk_rec[pop_a, :], dt, sigma_s)
    fr_b = spk_rec[pop_b, :].mean() / dt
    fr_hz_b = spk_rec[pop_b, :].mean(dim=0) / dt
    av_fr_b = ctp.compute_firing_rate(spk_rec[pop_b, :], dt, sigma_s)
    cf_b = torch.corrcoef(torch.stack((av_fr_b, step_signal)))[0, 1]
    cf_a = torch.corrcoef(torch.stack((av_fr_a, step_signal)))[0, 1]
    print(f'a: {fr_a.item():.2f}, b: {fr_b.item():.2f}')
    print(f'cfs: b {cf_b.item():.4f}, a {cf_a.item():.4f}')
    cspks = ctp.box_convolve_spikes(spk_rec)
    chi_a = ctp.compute_synchrony(spk_rec[pop_a, :])
    chi_b = ctp.compute_synchrony(spk_rec[pop_b, :])
    frs[i] = torch.tensor([fr_a, fr_b])
    cfs[i] = torch.tensor([cf_a, cf_b])
    chis[i] = torch.tensor([chi_a, chi_b])
    
cfs = torch.nan_to_num(cfs, nan=0.0)
chis = torch.nan_to_num(chis, nan=0.0)
frs = torch.nan_to_num(frs, nan=0.0)
# %%
# %%
x_ax = aa_vals
colors_wp = clmps.Accent(torch.linspace(0.0, 1.0, 8))


plt.figure(figsize=(9, 3))
plt.plot(x_ax, chis[:, 0], color=colors_wp[0], label=r'$\chi^2_{A}$')
plt.plot(x_ax, chis[:, 1], color=colors_wp[4], label=r'$\chi^2_{B}$')
plt.ylabel(r'Synchrony $\chi^2$')
plt.xlabel(r'Recurrent coupling strength')
plt.legend()
plt.grid()

plt.figure(figsize=(9, 3))

plt.plot(x_ax, cfs[:, 0], color=colors_wp[0], label=r'$\rho_{A}$')
plt.plot(x_ax, cfs[:, 1], color=colors_wp[4], label=r'$\rho_{B}$')
plt.ylabel(r'Correlation $\rho$')
plt.xlabel(r'Recurrent coupling strength')
plt.legend()
plt.grid()

plt.figure(figsize=(9, 3))
plt.plot(x_ax, frs[:, 0], color=colors_wp[0], label=r'$r_{A}$')
plt.plot(x_ax, frs[:, 1], color=colors_wp[4], label=r'$r_{B}$')
plt.ylabel(r'Firing rate $r$ [Hz]')
plt.xlabel(r'Recurrent coupling strength')
plt.legend()
plt.grid()
