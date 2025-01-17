import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cayley_gd import cayley_transform, cayley_gd


def test_cayley_transform():
    tau = 1.
    n = 20
    p = 5

    n_tests = 100
    passed = 0
    for i in range(n_tests):
        torch.manual_seed(2 * i)
        X = torch.randn(n, p, requires_grad=True)

        torch.manual_seed(2 * i + 1)
        Z = torch.randn(n, p)

        loss = ((X - Z) ** 2).sum()
        loss.backward()

        Y = cayley_transform(X, X.grad, tau)
        passed += torch.allclose(Y.T @ Y, X.T @ X, rtol=1e-5, atol=1e-5)

    print(f"cayley_transform() passed the test {passed}/{n_tests} times")


def test_cayley_gd():
    """To test the method, find orthogonal matrix X in the QR-decomposition
    M=XR (for known M and R) by minimizing Frobenius norm of XR-M."""
    tau = 1e-2
    n = 20
    p = 5
    n_iter = 100

    torch.manual_seed(0)
    M = torch.randn(n, p)
    Q, R = torch.linalg.qr(M, mode='reduced')

    loss_fn = lambda X: ((X @ R - M) ** 2).sum()

    X0 = torch.eye(n, p, requires_grad=True)
    _, traj, losses = cayley_gd(loss_fn, X0, tau, n_iter)

    Id = torch.eye(p)
    constr_violation = [((X.T @ X - Id) ** 2).sum().item() for X in traj]

    plt.figure(figsize=(8, 6))
    plt.plot(losses, label=r'$\| XR - M \|_F^2$')
    plt.plot(constr_violation, label=r'$\| X^\top X - I \|_F^2$')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    im_name = f'test_cayley_gd.jpg'
    plt.savefig(im_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    test_cayley_transform()
    test_cayley_gd()
