# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
# snn_module
import centipede as ctp
import sympy
import matplotlib.cm as clmps
dtype = torch.float64
device = torch.device("cpu")
# %%
# gain function
nb_neurons = 200
time_step = 1e-4
nb_steps  = 4000
t_max = time_step*nb_steps
r_ext1 = 100
r_ext2= 3000
r_ext3 = 1000000
r_ext = 3000
sig_val = 1.0
lambdah1 = time_step * r_ext1 * sig_val
lambdah2 = time_step * r_ext2 * sig_val
lambdah3 = time_step * r_ext3 * sig_val
t = torch.linspace(0, t_max, nb_steps)
# Membrane time constant for the update equation
tau_mem = 10e-3
beta = 1 - time_step / tau_mem
w_rec = torch.zeros((nb_neurons, nb_neurons), dtype=dtype)
inpt1 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah1)
inpt2 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah2)
#inpt = torch.zeros((nb_neurons, nb_steps)) + sig_val
inpt1 = inpt1/r_ext1/tau_mem
inpt2 = inpt2/r_ext2/tau_mem
inpt1.mean()
inpt2.mean()
# %%
v = sympy.symbols("v", cls=sympy.Function)
t, i = sympy.symbols("t i", real=True)
tau = sympy.symbols("tau", real=True, positive=True)
deq = sympy.Eq(tau * v(t).diff(t), -v(t) + i)
print('Neuron ode: ', deq)
deq_sol = sympy.dsolve(deq, v(t), ics={v(0): 0})
print('Analytical solution: ', deq_sol)
t_fire = sympy.solve(sympy.Eq(deq_sol.rhs, 1), t)[0]
print('Time to fire: ', t_fire)
print('Analytical gain: ', 1 / t_fire)
v_fire_eq = sympy.Eq(i - i*sympy.exp(-t/tau), 1)
v_fire = sympy.solve(v_fire_eq, i)
print('Current to fire: ', v_fire)
gain = sympy.lambdify((i, tau), 1 / t_fire)
i_num = 4 * np.arange(101) / 100
plt.plot(i_num, gain(i_num, 0.03))
### Analytical gain
def compute_an_gain(inps: torch.Tensor, tau: float):
    t_fire = tau * torch.log(inps / (inps - 1))
    return 1 / t_fire


def compute_t_tofire(inps: torch.Tensor, tau: float):
    return tau * torch.log(inps/(inps - 1))


def compute_curr_tofire(tf: float, tau:float):
    cr_time = torch.tensor([tf/tau])
    return torch.exp(cr_time)/(torch.exp(cr_time) - 1)

# %%
# Run the simulation
iters = 100
reps = 1
sig_vals = torch.linspace(0, 3, iters)
#varses = torch.zeros((iters, reps))
fir_rates1 = torch.zeros((reps, iters))
fir_rates2 = torch.zeros((reps, iters))
fir_rates3 = torch.zeros((reps, iters))

for j in range(reps):
    print(j)
    for i in range(iters):
        sig_val = sig_vals[i]
        lambdah1 = time_step * r_ext1 * sig_val
        lambdah2 = time_step * r_ext2 * sig_val
        lambdah3 = time_step * r_ext3 * sig_val
        print('sig val: ', sig_val)
        inpt1 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah1)
        inpt1 = inpt1/r_ext1/tau_mem
        inpt2 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah2)
        inpt2 = inpt2/r_ext2/tau_mem
        inpt3 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah3)
        inpt3 = inpt3/r_ext3/tau_mem

    
        _, spks1 = ctp.sim_snn(inpt1, w_rec, beta)
        spks1 = spks1.detach()
        measured_rate1 = spks1.sum()/t_max/nb_neurons
        print(measured_rate1)
        fir_rates1[j, i] = measured_rate1
        _, spks2 = ctp.sim_snn(inpt2, w_rec, beta)
        spks2 = spks2.detach()
        measured_rate2 = spks2.sum()/t_max/nb_neurons
        print(measured_rate2)
        fir_rates2[j, i] = measured_rate2
        _, spks3 = ctp.sim_snn(inpt3, w_rec, beta)
        spks3 = spks3.detach()
        measured_rate3 = spks3.sum()/t_max/nb_neurons
        print(measured_rate3)
        fir_rates3[j, i] = measured_rate3

# %%
blue_colors = clmps.pink(np.linspace(0.1, 0.5, 3))
an_gain = compute_an_gain(sig_vals, tau_mem)
#blue_colors = ['#8f99fb', '#3d7afd', '#152eff', '#0c1793', '#040273']
plt.plot(sig_vals, fir_rates1.mean(dim=0), label=f'{r_ext1} Hz', color=blue_colors[0], linewidth=1.7, alpha=0.95)
plt.plot(sig_vals, fir_rates2.mean(dim=0), label=f'{r_ext2} Hz', color=blue_colors[1], linewidth=1.7, alpha=0.95)
plt.plot(sig_vals, fir_rates3.mean(dim=0), label=f'{r_ext3} Hz', color=blue_colors[2], linewidth=1.7, alpha=0.8)
plt.plot(sig_vals, an_gain, "--", label='analytical gain', color="#ca0147")

plt.xlabel('Input current $I_{\mathrm{ext}}$') 
plt.ylabel('Firing rate $r$ [Hz]')
plt.grid()
plt.legend(title='External rate $r_{\mathrm{ext}}$')

plt.show()

# %%
sigma_s = 0.0015 # s
nb_flicker = 20 # 2500  # flicker values  # 10 for FR target
f_flicker = 50.0  # [Hz]
r_ext = 3000.0  # [Hz] - the higher, the less irregular is external input
val_min = 0.8  # [unitless, relative to spike threshold 1 with const input]
val_max = 1.1  # [unitless, relative to spike threshold 1 with const input]

dt_flicker = 1.0 / f_flicker
nb_timesteps_flickerstep = int(dt_flicker / time_step)
nb_steps = nb_flicker * nb_timesteps_flickerstep
t_max = nb_steps * time_step
t = time_step * torch.arange(nb_steps, device=device, dtype=dtype)
step_signal = ctp.generate_step_signal(nb_steps, time_step, n=dt_flicker, val_min=val_min, val_max=val_max)
lambdah1 = time_step * r_ext1 * step_signal
lambdah2 = time_step * r_ext2 * step_signal
lambdah3 = time_step * r_ext3 * step_signal
inpt1 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah1)
inpt1 = inpt1/r_ext1/tau_mem
inpt2 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah2)
inpt2 = inpt2/r_ext2/tau_mem
inpt3 = ctp.generate_neur_input(nb_neurons, nb_steps, lambdah3)
inpt3 = inpt3/r_ext3/tau_mem
_, spks1 = ctp.sim_snn(inpt1, w_rec, beta)
spks1 = spks1.detach()
_, spks2 = ctp.sim_snn(inpt2, w_rec, beta)
spks2 = spks2.detach()
_, spks3 = ctp.sim_snn(inpt3, w_rec, beta)
spks3 = spks3.detach()
av_fr1 = ctp.compute_firing_rate(spks1, time_step, sigma_s)
av_fr2 = ctp.compute_firing_rate(spks2, time_step, sigma_s)
av_fr3 = ctp.compute_firing_rate(spks3, time_step, sigma_s)
cf1 = torch.corrcoef(torch.stack((av_fr1, step_signal)))[0, 1]
cf2 = torch.corrcoef(torch.stack((av_fr2, step_signal)))[0, 1]
cf3 = torch.corrcoef(torch.stack((av_fr3, step_signal)))[0, 1]

# %%
blue_colors = clmps.pink(np.linspace(0.1, 0.5, 3))
red_color = "#ca0147"

# === Subplot 1 ===
plt.subplot(311)
plt.plot(t, step_signal, "--", color=red_color)
ax1 = plt.gca()
ax1r = ax1.twinx()
ax1r.plot(t, av_fr1, color=blue_colors[0])
#ax1r.set_ylabel(r"Output rate $r$ [Hz]", color=blue_colors[0])
#ax1.set_ylabel(r"Input current $I_{ext}$ [nA]", color=red_color)
ax1.tick_params(axis="y", colors=red_color)
ax1r.tick_params(axis="y", colors=blue_colors[0])
ax1.set_title(rf"$r_{{ext}} = {r_ext1}, \ \rho = {cf1:.2f}$")
ax1.grid(True)

# === Subplot 2 ===
plt.subplot(312)
plt.plot(t, step_signal, "--", color=red_color)
ax2 = plt.gca()
ax2r = ax2.twinx()
ax2r.plot(t, av_fr2, color=blue_colors[1])
ax2r.set_ylabel(r"Output rate $r$ [Hz]", color=blue_colors[1])
ax2.set_ylabel(r"Input current $I_{ext}$ [nA]", color=red_color)
ax2.tick_params(axis="y", colors=red_color)
ax2r.tick_params(axis="y", colors=blue_colors[1])
ax2.set_title(rf"$r_{{ext}} = {r_ext2}, \ \rho = {cf2:.2f}$")
ax2.grid(True)
ax2.set_xticklabels([])  # hide x tick labels

# === Subplot 3 ===
plt.subplot(313)
plt.plot(t, step_signal, "--", color=red_color)
ax3 = plt.gca()
ax3r = ax3.twinx()
ax3r.plot(t, av_fr3, color=blue_colors[2])
#ax3r.set_ylabel(r"Output rate $r$ [Hz]", color=blue_colors[2])
#ax3.set_ylabel(r"Input current $I_{ext}$ [nA]", color=red_color)
ax3.tick_params(axis="y", colors=red_color)
ax3r.tick_params(axis="y", colors=blue_colors[2])
ax3.set_title(rf"$r_{{ext}} = {r_ext3}, \ \rho = {cf3:.2f}$")
ax3.set_xlabel("Time $t$ [s]")
ax3.grid(True)

plt.tight_layout()
plt.show()

# %%
