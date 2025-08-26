# %%
import torch
import matplotlib.pyplot as plt
import snn_module as ctp
import matplotlib.cm as clmps

dtype = torch.float64
device = torch.device("cpu")

sim_name = 'AB_train_rec'
sim_num = 31
# %%
# neuron parameters
nb_neurons = 100
tau = 0.010  # [s]

# generate indices for all neurons and split into two populations
idcs_neurons = torch.arange(nb_neurons)
pop_a = idcs_neurons < (nb_neurons // 2)
pop_b = ~pop_a
nb_a = 50
nb_b = 50

# simulation parameters
dt = 0.0001  # [s]
verbose = False

# training parameters
learning_rate = 0.001 
nb_epochs = 30

# input parameters
nb_flicker = 100  # flicker values  # 10 for FR target
f_flicker = 100.0  # [Hz]
r_ext = 3000.0  # [Hz] - the higher, the less irregular is external input
val_min = 0.8  # [unitless, relative to spike threshold 1 with const input]
val_max = 1.0  # [unitless, relative to spike threshold 1 with const input]

# other parameters
sigma_s = 0.0025
w_range_init = 0.001

# derived parameters
dt_flicker = 1.0 / f_flicker
nb_timesteps_flickerstep = int(dt_flicker / dt)
nb_timesteps = nb_flicker * nb_timesteps_flickerstep
t_max = nb_timesteps * dt
t = dt * torch.arange(nb_timesteps, device=device, dtype=dtype)

# pre-compute parameters of simulation
beta = 1 - dt / tau

# %%
reps = 2
lossies = torch.zeros((reps, 3, nb_epochs, nb_flicker), device=device, dtype=dtype)
frs = torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
cfs = torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
w_params = torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
seeds = torch.zeros((reps))
bad_seeds = []
j = 0 # repetitions counter

while j < (reps):
    seed = torch.randint(0, 2**10, (1,))

    torch.manual_seed(seed)
    seed = torch.randint(0, 2**10, (1,))
    epoch_counter = 0

    # init weights
    w = torch.empty((nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(w, -w_range_init, w_range_init)
    p_ff = torch.tensor([0.5], dtype=dtype)
    str_ff = torch.tensor([0.030], dtype=dtype)
    ab_str = (torch.rand((nb_a*nb_b), dtype=dtype) < p_ff)*(str_ff/p_ff)
    w.data[pop_b.unsqueeze(-1) * pop_a] = ab_str 
    w.data[pop_b.unsqueeze(-1) * pop_b] = 0.0  # B -> B
    w.data[pop_a.unsqueeze(-1) * pop_b] = 0.0  # B -> A
    w = torch.nn.Parameter(w, requires_grad=True)

    optimizer = torch.optim.Adam([w], lr=learning_rate)
    for i_epoch in range(nb_epochs):

        print(f"Epoch {i_epoch + 1} of {nb_epochs} epochs...")
        skip_epoch = False

        # dry run
        with torch.no_grad():

            step_signal = ctp.generate_step_signal(nb_timesteps, dt, n=dt_flicker, val_min=val_min, val_max=val_max)
            lambdah = dt * r_ext * step_signal
            tmp = ctp.generate_neur_input(nb_neurons, nb_timesteps, lambdah)
            tmp[pop_b, :] = 0  # input signal goes only to pop A
            tmp_norm = 1 / (r_ext * tau)  # tmp_norm = 1 / (r_ext * dt) * (dt / tau)
            neur_input = tmp * tmp_norm  # normalize properly to keep avg input indep. of r_ext
            avg_neur_input = tmp[pop_a, :].mean(dim=0) / (r_ext * dt)
            mem0 = torch.rand((nb_neurons,), device=device, dtype=dtype)
            spk0 = torch.zeros((nb_neurons,), device=device, dtype=dtype)
            mem_rec, spk_rec = ctp.sim_snn(neur_input, w, beta, mem_init=mem0, spk_init=spk0, training=False)
            fr_a = spk_rec[pop_a, :].mean() / dt 
            fr_hz_a = spk_rec[pop_a, :].mean(dim=0) / dt
            av_fr_a = ctp.compute_firing_rate(spk_rec[pop_a, :], dt, sigma_s)
            fr_b = spk_rec[pop_b, :].mean() / dt
            fr_hz_b = spk_rec[pop_b, :].mean(dim=0) / dt
            av_fr_b = ctp.compute_firing_rate(spk_rec[pop_b, :], dt, sigma_s)

            out1_a = torch.reshape(fr_hz_a, [nb_flicker, nb_timesteps_flickerstep]).mean(dim=-1)
            out1_b = torch.reshape(fr_hz_b, [nb_flicker, nb_timesteps_flickerstep]).mean(dim=-1)
            inp = torch.reshape(step_signal, [nb_flicker, nb_timesteps_flickerstep]).mean(dim=-1)

            # cf computations
            cf_b = torch.corrcoef(torch.stack((av_fr_b, step_signal)))[0, 1]
            cf_a = torch.corrcoef(torch.stack((av_fr_a, step_signal)))[0, 1]
            print(f'cfs: b {cf_b.item():.4f}, a {cf_a.item():.4f}')
            print(f'frs: b {fr_b.item():.2f}, a {fr_a.item():.2f}')
            frs[j, :, i_epoch] = torch.tensor([fr_a, fr_b])
            cfs[j, :, i_epoch] = torch.tensor([cf_a, cf_b])
            w_params[j, :, i_epoch] = torch.tensor([w.data[pop_a.unsqueeze(-1) * pop_a].mean(), w.data[pop_a.unsqueeze(-1) * pop_a].std()])

        # wet run
        optimizer.zero_grad()
        # start from same initials
        mem_last = mem0
        spk_last = spk0

        # check nan
        check_nan_tensors = [cf_b, cf_a, fr_a, fr_b]
        contains_nan = any(torch.isnan(el) for el in check_nan_tensors)
        print(f'contains nan: {contains_nan}')
        if contains_nan:
            print(f'NaN detected with seed {seed}')
            bad_seeds.append(seed)
            skip_epoch = True
            break

        elif not skip_epoch:
            print(f'rep num: {j}')

            for i_flicker in range(nb_flicker):

                i_current = i_flicker * nb_timesteps_flickerstep
                tmp_neur_input = neur_input[:, i_current:i_current + nb_timesteps_flickerstep]
                mem_rec, spk_rec = ctp.sim_snn(tmp_neur_input, w, beta, mem_init=mem_last, spk_init=spk_last, training=True)

                mem_last = mem_rec[:, -1].detach()
                spk_last = spk_rec[:, -1].detach()

                out2_a_i_flicker = (spk_rec[pop_a, :].mean(dim=0) / dt).mean()
                out2_b_i_flicker = (spk_rec[pop_b, :].mean(dim=0) / dt).mean()

                assert out2_a_i_flicker == out1_a[i_flicker]
                assert out2_b_i_flicker == out1_b[i_flicker]


                # loss firing rate

                loss_firing = torch.abs(50.0 - out2_a_i_flicker) / 50.0

                # loss pearson between [0, 1]
                loss_pearson = (1.0 - ctp.i_am_a_weird_pearson(out2_b_i_flicker, i_flicker, out1_b, inp)) / 2

                loss = loss_firing + loss_pearson

                if verbose:
                    print(loss)

                lossies[j, 0, i_epoch, i_flicker] = loss.detach()
                lossies[j, 1, i_epoch, i_flicker] = loss_firing.detach()
                lossies[j, 2, i_epoch, i_flicker] = loss_pearson.detach()

                loss.backward()

            epoch_counter += 1
            if epoch_counter == nb_epochs:
                seeds[j] = seed
                j = j + 1
        
            w.grad[pop_b.unsqueeze(-1) * pop_a] = 0.0 # A -> B
            #w.grad[pop_a.unsqueeze(-1) * pop_a] = 0.0  # A -> A
            w.grad[pop_b.unsqueeze(-1) * pop_b] = 0.0  # B -> B
            w.grad[pop_a.unsqueeze(-1) * pop_b] = 0.0  # B -> A
            optimizer.step()

# %%
lossies_av = lossies.mean(dim=0) #torch.zeros((reps, 3, nb_epochs, nb_flicker), device=device, dtype=dtype)
frs_av = frs.mean(dim=0) #torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
cfs_av = cfs.mean(dim=0) #torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
w_params_av = w_params.mean(dim=0) #torch.zeros((reps, 6, nb_epochs), device=device, dtype=dtype)
lossies_std = lossies.std(dim=0) #torch.zeros((reps, 3, nb_epochs, nb_flicker), device=device, dtype=dtype)
frs_std = frs.std(dim=0) #torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
cfs_std = cfs.std(dim=0) #torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
w_params_std = w_params.std(dim=0) #torch.zeros((reps, 6, nb_epochs), device=device, dtype=dtype)
is_savefig = False #False #True
x_ax = torch.arange(nb_epochs)
# %%
plt.figure(figsize=(9, 4))
colors_wp = clmps.Accent(torch.linspace(0.0, 1.0, 8))
plt.plot(x_ax, lossies_av[1].cpu().mean(dim=-1), label='FR loss', color=colors_wp[1])
plt.fill_between(x_ax,
                lossies_av[1].cpu().mean(dim=-1)- lossies_std[1].cpu().mean(dim=-1),
                lossies_av[1].cpu().mean(dim=-1) + lossies_std[1].cpu().mean(dim=-1),
                color=colors_wp[1], alpha=0.1)
plt.plot(x_ax, lossies_av[2].cpu().mean(dim=-1), label='CC loss', color=colors_wp[2])
plt.fill_between(x_ax,
                lossies_av[2].cpu().mean(dim=-1)- lossies_std[2].cpu().mean(dim=-1),
                lossies_av[2].cpu().mean(dim=-1) + lossies_std[2].cpu().mean(dim=-1),
                color=colors_wp[2], alpha=0.1)
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.show()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'loss')
# %%
plt.figure(figsize=(9, 4))

plt.plot(x_ax, cfs_av[0, :], color=colors_wp[0], label=r'$\rho_{A}$')
plt.fill_between(x_ax,
                cfs_av[0, :]- cfs_std[0, :],
                cfs_av[0, :] + cfs_std[0, :],
                color=colors_wp[0], alpha=0.1)
plt.plot(x_ax, cfs_av[1, :], color=colors_wp[4], label=r'$\rho_{B}$')
plt.fill_between(x_ax,
                cfs_av[1, :]- cfs_std[1, :],
                cfs_av[1, :] + cfs_std[1, :],
                color=colors_wp[4], alpha=0.1)
plt.ylabel(r'Correlation $\rho$')
plt.xlabel('epoch')
plt.legend(loc='center left')
plt.grid()
plt.legend(loc='center right')
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'cfs')
# %%
plt.figure(figsize=(9, 4))
plt.plot(x_ax, frs_av[0, :], color=colors_wp[0], label=r'$r_{A}$')
plt.fill_between(x_ax,
                frs_av[0, :]- frs_std[0, :],
                frs_av[0, :] + frs_std[0, :],
                color=colors_wp[0], alpha=0.1)
plt.plot(x_ax, frs_av[1, :], color=colors_wp[4], label=r'$r_{B}$')
plt.fill_between(x_ax,
                frs_av[1, :]- frs_std[1, :],
                frs_av[1, :] + frs_std[1, :],
                color=colors_wp[4], alpha=0.1)
plt.ylabel(r'Firing rate $r$ [Hz]')
plt.xlabel('epoch')
plt.legend()
plt.grid()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'frs')
# %%
plt.figure(figsize=(9, 4))
plt.plot(x_ax, w_params_av[0, :], color=colors_wp[-2], label=r'$\mu_{aa}$')
plt.fill_between(x_ax,
                w_params_av[0, :] - w_params_std[0, :],
                w_params_av[0, :] + w_params_std[0, :],
                color=colors_wp[-2], alpha=0.1)
plt.plot(x_ax, w_params_av[1, :], '--', color=colors_wp[-2], label=r'$SD_{aa}$')
plt.fill_between(x_ax,
                w_params_av[1, :] - w_params_std[1, :],
                w_params_av[1, :] + w_params_std[1, :],
                color=colors_wp[-2], alpha=0.1)
plt.ylabel(r'Coupling strength')
plt.xlabel('epoch')
plt.legend()
plt.grid()
if is_savefig:
    plt.savefig(sim_name+str(sim_num)+'wei')

# %%
