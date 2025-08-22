# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

dtype = torch.float64
device = torch.device("cpu")


def heaviside(input_):
    out = (input_ > 0).type(dtype=dtype)
    return out


class HeavisideATan(torch.autograd.Function):
    """
    Taken from
    https://snntorch.readthedocs.io/en/latest/_modules/snntorch/surrogate.html#ATan
    Surrogate gradient of the Heaviside step function.
    **Forward pass:** Heaviside step function shifted.
        .. math:: / 2 * ctx.alpha * input_).pow_(2))
            * 
            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}
    **Backward pass:** Gradient of shifted arc-tan function.
        .. math::
                S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{π}\\frac{1}{(1+(πU\\frac{α}{2})^2)}
    α defaults to 2, and can be modified by calling \
        ``surrogate.atan(alpha=2)``.
    Adapted from:
    *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang,
    Y. Tian (2021) Incorporating Learnable Membrane Time Constants
    to Enhance Learning of Spiking Neural Networks. Proc. IEEE/CVF
    Int. Conf. Computer Vision (ICCV), pp. 2661-2671.*"""

    @staticmethod
    def forward(ctx, input_, alpha=2.0):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = heaviside(input_)
        return out

    @staticmethod
    def backward(ctx, grad_output):

        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_sur = (
            ctx.alpha
            / 2
            / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2)))
        grad = (
            grad_input
            * grad_sur
        )
        return grad
    

spike_fn = HeavisideATan.apply


def sim_snn(inpt, w_rec, beta, mem_init=None, spk_init=None, training=True):

    # input determines number of neurons and number of timesteps
    nb_timesteps = inpt.shape[-1]
    nb_neurons = inpt.shape[0]

    # if not defined, set random initial membrane pots in [0, 1]
    if mem_init is None:
        mem_init = torch.rand((nb_neurons), device=device, dtype=dtype)
    assert mem_init.shape[0] == nb_neurons, 'mem_init size should be equal to number of neurons'

    # if not defined, set initial spike vector to zero
    if spk_init is None:
        spk_init = torch.zeros((nb_neurons), device=device, dtype=dtype)
    assert spk_init.shape[0] == nb_neurons, 'spk_init size should be equal to number of neurons'

    # arrays for storing membrane potentials and spikes
    mem_rec = torch.zeros((nb_neurons, nb_timesteps), device=device, dtype=dtype)
    spk_rec = []

    # set initial values
    mem = mem_init
    spk = spk_init

    # loop through time
    for t in range(nb_timesteps):

        new_mem = beta * mem + inpt[:, t] + (w_rec @ spk)

        if training is True:
            spk = spike_fn(new_mem - 1.0)
        else:
            spk = heaviside(new_mem - 1.0)
        new_mem[spk == 1] = 0

        # store new membrane potentials and spikes
        mem_rec[:, t] = new_mem
        spk_rec.append(spk)
        mem = new_mem

    spk_rec = torch.stack(spk_rec, dim=1)

    return mem_rec, spk_rec


def generate_neur_input(nb_neurons, nb_steps, lambdah):

    ones = torch.ones((nb_neurons, nb_steps), device=device, dtype=dtype, requires_grad=False)
    spike_trains = torch.poisson(lambdah * ones)

    return spike_trains


def gaussian_kernel(dt, sigma):

    sigma_bins = sigma/dt
    kernel_size = int(sigma_bins*4)
    x = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, (kernel_size // 2) * 2 + 1, device=device, dtype=dtype)  # TODO device missing
    gaussian = torch.exp(-x**2 / (2 * sigma_bins**2))

    return gaussian / gaussian.sum()


def smooth_with_gaussian_convolution(input_tensor, dt, sigma):

    kernel = gaussian_kernel(dt, sigma)
    kernel_size = len(kernel)
    padding = (kernel_size-1) // 2  
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, length)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size)
    smoothed_tensor = torch.nn.functional.conv1d(input_tensor, kernel, padding=padding)

    return smoothed_tensor.squeeze()  # Remove batch and channel dimensions


# st and sigma in s
def compute_firing_rate(spks, dt, sigma):

    # Ensure spks is a 2D tensor with shape (n, time_steps)
    if spks.dim() != 2:
        raise ValueError("spks must be a 2D tensor with shape (n, time_steps)")

    av_spks_count = spks.mean(dim=0) / dt  # Shape: (time_steps,)
    av_spks_smooth = smooth_with_gaussian_convolution(av_spks_count, dt, sigma)

    return av_spks_smooth


def generate_step_signal(num_samples, dt, n=0.02, val_min=0.8, val_max=1.2):

    # n is width of ONE step...
    # dt is discretization of time...
    signal = torch.zeros((num_samples), device=device, dtype=dtype, requires_grad=False)

    num_steps = int(n / dt)
    num_vals = math.ceil(num_samples / num_steps)  # TODO
    values = torch.empty((num_vals)).uniform_(val_min, val_max)
    index = 0
    for value in values:
        signal[index:index + num_steps] = value
        index += num_steps

    return signal


def one_step_pearson(value, i_value, out, inp):

    n = out.shape[0]
    m = inp.shape[0]

    assert n == m, "out and inp should have same number of elements!"
    assert out[i_value] == value, "out[i_value] should match value!"

    inp_mean = inp.mean()
    out_mean = (out.sum() - out[i_value] + value) / n

    nom = 0.0
    denom_out = 0.0
    denom_inp = 0.0

    for i in range(n):
        if i == i_value:
            nom += (value - out_mean) * (inp[i] - inp_mean)
            denom_out += (value - out_mean) ** 2
        else:
            nom += (out[i] - out_mean) * (inp[i] - inp_mean)
            denom_out += (out[i] - out_mean) ** 2
        denom_inp += (inp[i] - inp_mean) ** 2

    pearson = nom / torch.sqrt(denom_out * denom_inp)

    return pearson


def box_convolve_spikes(spike_train, nc=10):

    kernel = torch.ones(nc, dtype=dtype, device=spike_train.device) / nc
    
    # Input shape: (batch_size=1, channels=num_neurons, num_steps)
    spike_train = spike_train.unsqueeze(0)  # Shape: (1, N, nb_steps)
    
    # Kernel shape: (groups=num_neurons, 1, kernel_size)
    kernel = kernel.repeat(spike_train.shape[1], 1, 1)  # Shape: (N, 1, nc)
    
    convolved = F.conv1d(
        spike_train,
        kernel,
        padding=nc//2,
        groups=spike_train.shape[1]
    )
    
    return convolved.squeeze(0)


def compute_synchrony(spike_trains):

    # Compute the global variable (population activity) X(t)
    X_t = spike_trains.mean(dim=0)  # Average across neurons
    # Variance of the global variable
    var_X = torch.var(X_t)
    # Variance of individual neurons
    var_V = torch.var(spike_trains, dim=1)  # Variance across time for each neuron
    # Average variance across neurons
    mean_var_V = torch.mean(var_V)
    # Synchrony measure
    gamma = var_X / mean_var_V

    return gamma


def raster_plot(spks, t_max, dt, t_min=0, color='black'):
    t_sim = t_max - t_min
    num_steps = round(t_sim/dt)
    start_steps = round(t_min/dt)
    end_steps = round(t_max/dt)
    t = torch.linspace(t_min, t_max, num_steps)
    spks = spks[:, start_steps:end_steps]
    spike_mask = spks.bool()
    neuron_idx, spike_times = torch.nonzero(spike_mask, as_tuple=True)
    spike_times = t[spike_times].numpy()
    neuron_idx = neuron_idx.numpy()
    plt.scatter(spike_times, neuron_idx, color=color, marker='|', s=10)