import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_chain(x,name=None,filename=None):
    """
    Produce a trace plot
    """
    fig=plt.figure(figsize=(4,3))
    plt.plot(x,',')
    plt.grid()
    plt.xlabel('iteration')
    if name is not None:
        plt.ylabel(name)
        if filename is None:
            filename=name+'_chain.png'
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close(fig)

def plot_proposal_stats(path):
    """
    Plot acceptance and number of proposed points for each chain
    """
    import glob
    files = glob.glob(path + 'proposal*.dat')
    chains = []
    for f in files:
        chains.append(np.loadtxt(f))

    fig = plt.figure(figsize=(10,6))
    for i, c in enumerate(chains):
        plt.plot((c[:, 2] / c[:, 1]), '.', label='Chain {}'.format(i))
    plt.xlabel('Iteration')
    plt.ylabel('Acceptance ratio')
    plt.legend()
    plt.savefig(path + 'proposal_acceptance.png' ,bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(10,6))
    for i, c in enumerate(chains):
        plt.plot(c[:, 1], '.', label='Chain {} proposed'.format(i))
        plt.plot(c[:, 2], '.', label='Chain {} accepted'.format(i))
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(path + 'proposal_counts.png' ,bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(10,6))
    for i, c in enumerate(chains):
        plt.plot(np.cumsum(c[:, 1]), label='Chain {}'.format(i))
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(path + 'proposal_likelihood_evalutions.png', bbox_inches='tight')
    plt.close(fig)

def plot_corner_samples(samples, filename=None, cmap=True):
    """
    Make a corner plot showing all of the nested_samples
    """
    samples_list = []
    for n in samples.dtype.names:
        samples_list.append(samples[n])
    samples = np.array(samples_list).T
    samples = samples[:, :-2]
    N = samples.shape[0]
    d = samples.shape[-1]
    if cmap:
        c = plt.cm.plasma(np.linspace(0, 1, N))
        c_hist = 'tab:purple'
    else:
        c = 'tab:blue'
        c_hist = c

    fig,ax = plt.subplots(d, d, figsize=(4*d, 4*d))
    for i in range(d):
        for j in range(d):
            if j < i:
                ax[i, j].scatter(samples[:,j], samples[:,i], c=c, marker='.', s=0.5)
            elif j == i:
                ax[i, j].hist(samples[:, j], bins=int(np.sqrt(N)), color=c_hist)
            else:
                ax[i, j].set_axis_off()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close(fig)

def plot_hist(x,name=None,filename=None):
    """
    Produce a histogram
    """
    fig=plt.figure(figsize=(4,3))
    plt.hist(x, density = True, facecolor = '0.5', bins=int(len(x)/20))
    plt.ylabel('probability density')
    if name is not None:
        plt.xlabel(name)
        if filename is None:
            filename=name+'_hist.png'
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close(fig)

def plot_indices(indices, nlive=None, u=None, name=None, filename=None):
    """
    Histogram indices for index insertion tests
    """
    import random
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(indices, density = True, color='tab:blue', linewidth = 1.25,
                histtype='step', bins=len(indices)//50, label = 'produced')
    if u is not None:
        ax.hist(u, density = True, color='tab:orange', linewidth = 1.25,
                    histtype='step', bins=len(u)//50, label = 'expected')
    if nlive is not None:
        ax.axhline(1 / nlive, color='black', linewidth=1.25, linestyle=':',
                label='pmf')

    ax.legend(loc='upper left')
    ax.set_xlabel('insertion indices')
    if name is not None:
        ax.set_xlabel(name)
        if filename is None:
            filename=name+'_hist.pdf'
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_corner(xs,filename=None,**kwargs):
    """
    Produce a corner plot
    """
    import corner
    fig=plt.figure(figsize=(10,10))
    mask = [i for i in range(xs.shape[-1]) if not all(xs[:,i]==xs[0,i]) ]
    corner.corner(xs[:,mask],**kwargs)
    if filename is not None:
        plt.savefig(filename,bbox_inches='tight')
    plt.close(fig)

def plot_flows(model, n_inputs, N=1000, inputs=None, cond_inputs=None, mode='inverse', output='./'):
    """
    Plot each stage of a series of flows
    """
    import matplotlib.pyplot as plt

    if n_inputs > 2:
        raise NotImplementedError('Plotting for higher dimensions not implemented yet!')

    outputs = []

    if mode == 'direct':
        if inputs is None:
            raise ValueError('Can not sample from parameter space!')
        else:
            inputs = torch.from_numpy(inputs).to(model.device)

        for module in model._modules.values():
            inputs, _ = module(inputs, cond_inputs, mode)
            outputs.append(inputs.detach().cpu().numpy())
    else:
        if inputs is None:
            inputs  = torch.randn(N, n_inputs, device=model.device)
            orig_inputs = inputs.detach().cpu().numpy()
        for module in reversed(model._modules.values()):
            inputs, _ = module(inputs, cond_inputs, mode)
            outputs.append(inputs.detach().cpu().numpy())

    n = int(len(outputs) / 2) + 1
    m = 1

    if n > 5:
        m = int(np.ceil(n / 5))
        n = 5

    z = orig_inputs
    pospos = np.where(np.all(z>=0, axis=1))
    negneg = np.where(np.all(z<0, axis=1))
    posneg = np.where((z[:, 0] >= 0) & (z[:, 1] < 0))
    negpos = np.where((z[:, 0] < 0) & (z[:, 1] >= 0))

    points = [pospos, negneg, posneg, negpos]
    colours = ['r', 'c', 'g', 'tab:purple']
    colours = plt.cm.Set2(np.linspace(0, 1, 8))


    fig, ax = plt.subplots(m, n, figsize=(n * 3, m * 3))
    ax = ax.ravel()
    for j, c in zip(points, colours):
        ax[0].plot(z[j, 0], z[j, 1], ',', c=c)
        ax[0].set_title('Latent space')
    for i, o in enumerate(outputs[::2]):
        i += 1
        for j, c in zip(points, colours):
            ax[i].plot(o[j, 0], o[j, 1], ',', c=c)
        ax[i].set_title(f'Flow {i}')
        #ax[i].plot(o[:, 0], o[:, 1], ',')
    plt.tight_layout()
    fig.savefig(output + 'flows.png')

def plot_loss(epoch, history, output='./'):
    """
    Plot a loss function
    """
    fig = plt.figure()
    epochs = np.arange(1, epoch + 1, 1)
    plt.plot(epochs, history['loss'], label='Loss')
    plt.plot(epochs, history['val_loss'], label='Val. loss')
    plt.xlabel('Epochs')
    plt.ylabel('Negative log-likelihood')
    plt.legend()
    plt.tight_layout()
    fig.savefig(output + 'loss.png')
    plt.close('all')


def plot_samples(z, samples, output='./', filename='output_samples.png', names=None, c=None):
    """
    Plot the samples in the latent space and parameter space
    """
    N = samples.shape[0]
    d = samples.shape[-1]

    latent = True
    if z.shape[-1] != d:
        print('Not plotting latent space')
        latent = False

    if names is None:
        names = list(range(d))
    latent_names =  [f'z_{n}' for n in range(d)]

    if c is None:
        c = 'tab:blue'

    samples = samples[np.isfinite(samples).all(axis=1)]

    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d >= 2:
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].scatter(samples[:,j], samples[:,i], c=c, s=1.)
                    ax[i, j].set_xlabel(names[j])
                    ax[i, j].set_ylabel(names[i])
                elif j == i:
                    ax[i, j].hist(samples[:, j], int(np.sqrt(N)), histtype='step')
                    ax[i, j].set_xlabel(names[j])
                else:
                    if latent:
                        ax[i, j].scatter(z[:,j], z[:,i], c=c, s=1.)
                        ax[i, j].set_xlabel(latent_names[j])
                        ax[i, j].set_ylabel(latent_names[i])
                    else:
                        ax[i, j].axis('off')
    else:
        ax.hist(samples, int(np.sqrt(N)), histtype='step')

    plt.tight_layout()
    fig.savefig(output + filename)
    plt.close('all')

def plot_inputs(samples, output='./', filename='input_samples.png', names=None):
    """
    Plot n-dimensional input samples
    """
    N = samples.shape[0]
    d = samples.shape[-1]
    if names is None:
        names = list(range(d))

    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d > 1:
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].plot(samples[:,j], samples[:,i], marker=',', linestyle='')
                    ax[i, j].set_xlabel(names[j])
                    ax[i, j].set_ylabel(names[i])
                elif j == i:
                    ax[i, j].hist(samples[:, j], int(np.sqrt(N)), histtype='step')
                    ax[i, j].set_xlabel(names[j])
                else:
                    ax[i, j].set_axis_off()
    else:
        ax.hist(samples, int(np.sqrt(N)), histtype='step')
    plt.tight_layout()
    fig.savefig(output + filename ,bbox_inches='tight')
    plt.close('all')

def plot_comparison(truth, samples, output='./', filename='sample_comparison.png'):
    """
    Plot the samples in the latent space and parameter space
    """
    d = samples.shape[-1]

    samples = samples[np.isfinite(samples).all(axis=1)]

    xs = [truth, samples]
    labels = ['reference', 'flows']

    fig, ax = plt.subplots(d, d, figsize=(12, 12))
    if d > 1:
        for i in range(d):
            for j in range(d):
                for x, l in zip(xs, labels):
                    if j < i:
                        ax[i, j].plot(x[:,j], x[:,i], marker=',', linestyle='')
                    elif j == i:
                        ax[i, j].hist(x[:, j], int(np.sqrt(samples.shape[0])), histtype='step', lw=2.0)
                    else:
                        ax[i, j].axis('off')
    else:
        for x, l in zip(xs, labels):
            ax.hist(x, int(np.sqrt(samples.shape[0])), histtype='step')
    plt.tight_layout()
    fig.savefig(output + filename)
    plt.close('all')

def generate_contour(r, dims, N=1000):
    """Generate a contour"""
    x = np.array([np.random.randn(N) for _ in range(dims)])
    R = np.sqrt(np.sum(x ** 2., axis=0))
    z = x / R
    return r * z.T

def plot_contours(contours, output='./', filename='contours.png', names=None):
    """Plot contours in the latent space and physical space"""
    d = contours.shape[-1]
    if names is None:
        names = list(range(d))
    latent_names =  [f'z_{n}' for n in range(d)]
    fig, ax = plt.subplots(d, d, figsize=(d*3, d*3))
    if d >= 2:
        for c in contours:
            for i in range(d):
                for j in range(d):
                    if j < i:
                        ax[i, j].scatter(c[1, :,j], c[1, :,i], s=1.)
                    elif j == i:
                        pass
                    else:
                        ax[i, j].scatter(c[0, :,j], c[0, :,i], s=1.)
        for i in range(d):
            for j in range(d):
                if j < i:
                    ax[i, j].set_xlabel(names[j])
                    ax[i, j].set_ylabel(names[i])
                elif j == i:
                    ax[i, j].axis('off')
                else:
                    ax[i, j].set_xlabel(latent_names[j])
                    ax[i, j].set_ylabel(latent_names[i])
        plt.tight_layout()
        fig.savefig(output + filename)
        plt.close('all')
    else:
        pass

