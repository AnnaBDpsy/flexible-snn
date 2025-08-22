# %%
# ctAB model with att
# TRAIN THE NETWORK ========================================================================
import torch
import matplotlib.pyplot as plt
# snn_module
import centipede as ctp
import matplotlib.cm as clmps
import numpy as np

dtype = torch.float64
device = torch.device("cpu")

sim_name = 'aAB_train'
sim_num = 81

# %%
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
verbose = False

# training parameters
learning_rate = 0.1#0.025  # FR target 0.001, nb_epochs = 100
nb_epochs = 50 #30

# input parameters
nb_flicker = 100  # flicker values  # 10 for FR target
f_flicker = 100.0  # [Hz]
r_ext = 3000.0  # [Hz] - the higher, the less irregular is external input
val_min = 0.8  # [unitless, relative to spike threshold 1 with const input]
val_max = 1.0  # [unitless, relative to spike threshold 1 with const input]

# other parameters
sigma_s = 0.0025
w_range_init = 0.1 #0.08 #0.16  # 0.1 for FR target

# derived parameters...
dt_flicker = 1.0 / f_flicker
nb_timesteps_flickerstep = int(dt_flicker / dt)
nb_timesteps = nb_flicker * nb_timesteps_flickerstep
t_max = nb_timesteps * dt
t = dt * torch.arange(nb_timesteps, device=device, dtype=dtype)

# pre-compute parameters of simulation
beta = 1 - dt / tau
# %%
reps = 10
lossies = torch.zeros((reps, 3, nb_epochs, nb_flicker), device=device, dtype=dtype)
frs = torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
cfs = torch.zeros((reps, 2, nb_epochs), device=device, dtype=dtype)
w_params = torch.zeros((reps, 6, nb_epochs), device=device, dtype=dtype)
w2_params = torch.zeros((reps, 6, nb_epochs), device=device, dtype=dtype)
raw_weights = torch.zeros((reps, nb_epochs, nb_neurons, nb_neurons))
map_weights = torch.zeros((reps, nb_epochs, nb_neurons, nb_neurons))
all_spks = torch.zeros((reps, nb_epochs, nb_neurons, nb_timesteps))
all_sigs = torch.zeros((reps, nb_epochs, nb_timesteps), device=device, dtype=dtype)
seeds = torch.zeros((reps))
bad_seeds = []
j = 0

while j < (reps):
    seed = torch.randint(0, 2**10, (1,))
    torch.manual_seed(seed)
    seed = torch.randint(0, 2**10, (1,))
    while seed in bad_seeds:
        seed = torch.randint(0, 2**10, (1,))
    epoch_counter = 0

    # init weights
    w = torch.empty((nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(w, -w_range_init, w_range_init)

    p_ff = torch.tensor([0.5], dtype=dtype)
    str_ff = torch.tensor([0.03], dtype=dtype)
    ab_str = (torch.rand((nb_a*nb_b), dtype=dtype) < p_ff)*(str_ff/p_ff)
    aa_str = 0.001 
    bb_str = 0.0
    ba_str = 0.0
    ct_str = 0.09 
    cta_str = 0.004


    w.data[pop_ct.unsqueeze(-1) * pop_ct] = w.data[pop_ct.unsqueeze(-1) * pop_ct].uniform_(-w_range_init-0.5, w_range_init-0.5)  
    w.data[pop_b.unsqueeze(-1) * pop_ct] = 0.0 # ct -> B
    w.data[pop_a.unsqueeze(-1) * pop_ct] = w.data[pop_a.unsqueeze(-1) * pop_ct].uniform_(-w_range_init-1.5, w_range_init-1.5) 
    w.data[(pop_a+pop_b) * pop_ct.unsqueeze(-1)] = 0.0 #A&B -> ct
    w.data[pop_a.unsqueeze(-1) * pop_a] = w.data[pop_a.unsqueeze(-1) * pop_a].uniform_(-w_range_init-0.4, w_range_init-0.4)
    w.data[pop_b.unsqueeze(-1) * pop_a] = ab_str  # A -> B
    w.data[pop_b.unsqueeze(-1) * pop_b] = 0.0  # B -> B
    w.data[pop_a.unsqueeze(-1) * pop_b] = 0.0  # B -> A
    w = torch.nn.Parameter(w, requires_grad=True)


    optimizer = torch.optim.Adam([w], lr=learning_rate)


    for i_epoch in range(nb_epochs):

        print(f"Epoch {i_epoch + 1} of {nb_epochs} epochs...")
        skip_epoch = False
        # map weights
        with torch.no_grad():
            #mask_all = torch.zeros_like(w)
            #mask_all[pop_a.unsqueeze(-1) * pop_a] = 1.0
            #mask_all[pop_a.unsqueeze(-1) * pop_ct] = 1.0
            #mask_all[pop_ct.unsqueeze(-1) * pop_ct] = 1.0
            mask_aa = torch.zeros_like(w)
            mask_aa[pop_a.unsqueeze(-1) * pop_a] = 1.0
            mask_cta = torch.zeros_like(w)
            mask_cta[pop_a.unsqueeze(-1) * pop_ct] = 1.0
            mask_ct = torch.zeros_like(w)
            mask_ct[pop_ct.unsqueeze(-1) * pop_ct] = 1.0

        with torch.no_grad():
            #w2 = w * (1.0 - mask_all)
            #w2 = w2 + mask_all * torch.abs(w)
            w2 = w * (1.0 - mask_cta)
            w2 = w2 + mask_cta * torch.sigmoid(w) * 0.006  #* 0.008#* torch.abs(w) #* torch.sigmoid(w) * 0.02
            w2 = w2 * (1.0 - mask_aa)
            w2 = w2 + mask_aa * torch.sigmoid(w) * 0.004   #* 0.006#* torch.abs(w) #* torch.sigmoid(w) * 0.02
            w2 = w2 * (1.0 - mask_ct)
            w2 = w2 + mask_ct * torch.sigmoid(w) * 0.12#* torch.abs(w) #* torch.sigmoid(w) * 0.02

        # dry run


        with torch.no_grad():
            #att = torch.tensor([1.0], dtype=dtype) # off
            att = torch.tensor([i_epoch % 2], dtype=dtype)  #torch.tensor([1.0], dtype=dtype) #torch.tensor([i_epoch % 2], dtype=dtype) # #torch.randint(0, 2, (1,), dtype=dtype)
            step_signal = ctp.generate_step_signal(nb_timesteps, dt, n=dt_flicker, val_min=val_min, val_max=val_max)
            #all_sigs[j, i_epoch] = step_signal.clone()
            lambdah = dt * r_ext * step_signal
            tmp = ctp.generate_neur_input(nb_neurons, nb_timesteps, lambdah)
            tmp[pop_ct, :] = tmp[pop_ct, :] * att # att allocation
            tmp[pop_b, :] = 0  # input signal goes only to pop A and ct
            tmp_norm = 1 / (r_ext * tau)  # tmp_norm = 1 / (r_ext * dt) * (dt / tau)
            neur_input = tmp * tmp_norm  # normalize properly to keep avg input indep. of r_ext
            avg_neur_input = tmp[pop_a, :].mean(dim=0) / (r_ext * dt)
            mem0 = torch.rand((nb_neurons,), device=device, dtype=dtype)
            spk0 = torch.zeros((nb_neurons,), device=device, dtype=dtype)

            mem_rec, spk_rec = ctp.sim_snn(neur_input, w2, beta, mem_init=mem0, spk_init=spk0, training=False)
            #all_spks[j, i_epoch] = spk_rec.clone()
            fr_ct = spk_rec[pop_ct, :].mean() / dt  
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
            print(f'frs: b {fr_b.item():.2f}, a {fr_a.item():.2f}, ct: {fr_ct.item():.2f}')
            #with torch.no_grad:
            frs[j, :, i_epoch] = torch.tensor([fr_a, fr_b])
            cfs[j, :, i_epoch] = torch.tensor([cf_a, cf_b])
            w_params[j, :, i_epoch] = torch.tensor([w.data[pop_a.unsqueeze(-1) * pop_a].mean(), 
                                                w.data[pop_a.unsqueeze(-1) * pop_a].std(),
                                                w.data[pop_a.unsqueeze(-1) * pop_ct].mean(),
                                                w.data[pop_a.unsqueeze(-1) * pop_ct].std(),
                                                w.data[pop_ct.unsqueeze(-1) * pop_ct].mean(),
                                                w.data[pop_ct.unsqueeze(-1) * pop_ct].std()])
            w2_params[j, :, i_epoch] = torch.tensor([w2.data[pop_a.unsqueeze(-1) * pop_a].mean(), 
                                                w2.data[pop_a.unsqueeze(-1) * pop_a].std(),
                                                w2.data[pop_a.unsqueeze(-1) * pop_ct].mean(),
                                                w2.data[pop_a.unsqueeze(-1) * pop_ct].std(),
                                                w2.data[pop_ct.unsqueeze(-1) * pop_ct].mean(),
                                                w2.data[pop_ct.unsqueeze(-1) * pop_ct].std()])
            print(f'AA mean {w2.data[pop_a.unsqueeze(-1) * pop_a].mean().item():.3f}')
            print(f'ct-A mean {w2.data[pop_a.unsqueeze(-1) * pop_ct].mean().item():.3f}')
            print(f'ct-ct mean {w2.data[pop_ct.unsqueeze(-1) * pop_ct].mean().item():.3f}')
            raw_weights[j, i_epoch] = w.data.clone()
            map_weights[j, i_epoch] = w2.data.clone()

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


                w2 = w * (1.0 - mask_cta)
                w2 = w2 + mask_cta * torch.sigmoid(w) * 0.006  #* 0.008#* torch.abs(w) #* torch.sigmoid(w) * 0.02
                w2 = w2 * (1.0 - mask_aa)
                w2 = w2 + mask_aa * torch.sigmoid(w) * 0.004   #* 0.006#* torch.abs(w) #* torch.sigmoid(w) * 0.02
                w2 = w2 * (1.0 - mask_ct)
                w2 = w2 + mask_ct * torch.sigmoid(w) * 0.12#* torch.abs(w) #* torch.sigmoid(w) * 0.02

                #w2 = w * (1.0 - mask_all)
                #w2 = w2 + mask_all * torch.abs(w)

                mem_rec, spk_rec = ctp.sim_snn(
                    tmp_neur_input, 
                    w2, 
                    beta, 
                    mem_init=mem_last, 
                    spk_init=spk_last, 
                    training=True)

                mem_last = mem_rec[:, -1].detach()
                spk_last = spk_rec[:, -1].detach()

                out2_a_i_flicker = (spk_rec[pop_a, :].mean(dim=0) / dt).mean()
                out2_b_i_flicker = (spk_rec[pop_b, :].mean(dim=0) / dt).mean()

                assert out2_a_i_flicker == out1_a[i_flicker]
                assert out2_b_i_flicker == out1_b[i_flicker]



                # loss = (out2_b_i_flicker - 150.0) ** 2
                # loss firing rate
                loss_firing = (torch.abs(50.0 - out2_a_i_flicker) / 50.0)
                
                wp_cf_b = ctp.i_am_a_weird_pearson(out2_b_i_flicker, i_flicker, out1_b, inp)
                # loss pearson between [0, 1]
                loss_pearson = ((1.0 -wp_cf_b) / 2)*att + torch.abs(wp_cf_b)*(1.0 - att)

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
            # comment trainable weights

            #w.grad[pop_a.unsqueeze(-1) * pop_a] = 0.0  # A -> A
            #w.grad[pop_a.unsqueeze(-1) * pop_ct] = 0.0 # ct -> A
            #w.grad[pop_ct.unsqueeze(-1) * pop_ct] = 0.0 # ct -> ct
            # never used

            w.grad[pop_b.unsqueeze(-1) * pop_b] = 0.0  # B -> B
            w.grad[pop_a.unsqueeze(-1) * pop_b] = 0.0  # B -> A
            w.grad[pop_b.unsqueeze(-1) * pop_ct] = 0.0 # ct -> B
            w.grad[(pop_a+pop_b) * pop_ct.unsqueeze(-1)] = 0.0 #A&B -> ct
            w.grad[pop_b.unsqueeze(-1) * pop_a] = 0.0  # A -> B

            optimizer.step()