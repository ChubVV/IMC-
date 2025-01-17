"""
Implementation of Cayley gradient descent for the paper
``A Feasible Method for Optimization with Orthogonality Constraints''
by Z Wen, W Yin
"""
import torch


def cayley_transform(X, G, tau, returns='Y'):
    """X and G have shape (n, p)"""
    A = G @ X.T - X @ G.T
    n = X.shape[0]
    Id = torch.eye(n)
    Q = torch.inverse(Id + 0.5 * tau * A) @ (Id - 0.5 * tau * A)

    if returns == 'Y':
        return Q @ X
    return Q


def cayley_gd(loss_fn, X0, tau, n_iter, save_traj=True):
    X = X0

    traj = [X.clone().detach()] if save_traj else None
    losses = [] if save_traj else None

    for _ in range(n_iter):
        loss = loss_fn(X)
        loss.backward()

        with torch.no_grad():
            X.copy_(cayley_transform(X, X.grad, tau))

        X.grad.zero_()

        if save_traj:
            traj.append(X.clone().detach())
            losses.append(loss.item())

    return X, traj, losses
