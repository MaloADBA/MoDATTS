import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.autograd import Variable

class mine(nn.Module):
    """
    MINE: Mutual information neural estimator.
    https://arxiv.org/abs/1801.04062
    
    TODO: reduce bias in MINE gradient estimation.
    """
    def __init__(self, estimation_network, rng=None):
        super(mine, self).__init__()
        self.estimation_network = estimation_network
        self.rng = rng if rng else np.random.RandomState()
        
    def forward(self, x, z, z_marginal=None):
        if z_marginal is None:
            permutation = self.rng.permutation(len(z))
            z_marginal = z[permutation]
        joint = self.estimation_network(x, z)
        marginal = self.estimation_network(x, z_marginal)
        assert np.product(joint.size())==len(joint)
        assert np.product(marginal.size())==len(marginal)
        lower_bound = ( torch.mean(joint)
                       -torch.log(torch.sum(torch.exp(marginal)))
                       +np.log(marginal.size(0)))
        return -lower_bound


def _run_mine():
    from matplotlib import pyplot as plt
    import time
    
    # opt
    use_spectral_norm = False
    weight_decay = 0.01
    
    # net
    n_hidden = 400
    n_iter = 10000
    
    # Data parameters.
    N = 10000
    size = 20
    covariance = 0.9
    
    # Covariance matrix.
    cov = np.eye(size*2)
    cov[size:, :size] += np.eye(size)*covariance
    cov[:size, size:] += np.eye(size)*covariance
    
    # Data.
    def sample_data():
        sample = np.random.multivariate_normal(mean=[0]*size*2,
                                               cov=cov,
                                               size=(N,))
        x = sample[:,:size]
        z = sample[:,size:]
        return x, z
    
    # Theoretical mutual information.
    sx = np.linalg.det(cov[:size, :size])
    sz = np.linalg.det(cov[size:, size:])
    s  = np.linalg.det(cov)
    mi_real = 0.5*np.log(sx*sz/s)
    
    # Mutual information estimator.
    def _n(layer):
        if use_spectral_norm:
            return spectral_norm(layer)
        return layer
    class mi_estimation_network(nn.Module):
        def __init__(self, n_hidden):
            super(mi_estimation_network, self).__init__()
            self.n_hidden = n_hidden
            modules = []
            modules.append(_n(nn.Linear(size*2, self.n_hidden)))
            modules.append(nn.ReLU())
            for i in range(0):
                modules.append(_n(nn.Linear(self.n_hidden, self.n_hidden)))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(self.n_hidden, 1))
            self.model = nn.Sequential(*tuple(modules))
        
        def forward(self, x, z):
            out = self.model(torch.cat([x, z], dim=-1))
            return out
    
    mi_estimator = mine(mi_estimation_network(n_hidden=n_hidden))
    model = mi_estimator.estimation_network
    
    # Train
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001,
                                 eps=1e-7,
                                 weight_decay=weight_decay,
                                 amsgrad=True)
    model.cuda()
    model.train()
    fig, ax = plt.subplots(1,1)
    ax.axhline(mi_real, color='red', linestyle='dashed')
    fig.show()
    fig.canvas.draw()
    loss_history = []
    for i in range(n_iter):
        optimizer.zero_grad()
        x, z = sample_data()
        _, z_marginal = sample_data()
        x = torch.from_numpy(x.astype(np.float32)).cuda()
        z = torch.from_numpy(z.astype(np.float32)).cuda()
        z_marginal = torch.from_numpy(z_marginal.astype(np.float32)).cuda()
        loss = mi_estimator(x, z, z_marginal)
                
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        print("Iteration {} - lower_bound={:.2f} (real {:.2f}) norm={}"
              "".format(i, -loss.item(), mi_real, norm))
        loss_history.append(-loss.item())
        if (i)%100==0:
            plt.scatter(range(i+1), loss_history, c='black', s=2)
            fig.canvas.draw()
            
            def get_sv(layer):
                _, s, _ = np.linalg.svd(layer.weight.data.cpu().numpy())
                return np.mean(s), np.min(s), np.max(s)
            for i, m in enumerate(model.model):
                if isinstance(m, nn.Linear):
                    print("Singular values at layer {}: mean={:.2f}, "
                          "min={:.2f}, max={:.2f}".format(i, *get_sv(m)))
                
            
    plt.show(block=True)    # Keep figure until it's closed.


if __name__=='__main__':
    print("\nRUNNING MINE\n")
    _run_mine()
    
