import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from algorithm import run_algorithm


def sample_ortog_mat(n1, n2):
    """Generate an n1 x n2 matrix with ortogonal columns"""
    Q = torch.randn(n1, n2)
    Q = torch.linalg.qr(Q)[0]
    return Q


def generate_matrix(n1, n2, singular_values):
    """Generate an n1 x n2 matrix X with the specified singular values"""
    rank = len(singular_values)

    U = sample_ortog_mat(n1, rank)
    V = sample_ortog_mat(n2, rank)
    D = torch.diag(singular_values)

    return U @ D @ V.T


def generate_template(n1, n2, p):
    """Generate a template"""
    diag1 = torch.bernoulli(p * torch.ones(n1))
    diag2 = torch.bernoulli(p * torch.ones(n2))
    T = torch.diag(diag1) @ torch.ones(n1, n2) @ torch.diag(diag2)
    return T


def choose_optimal_qr(Q, R):
    """Multiply some columns of Q by -1 to make it closer to I.
    Multiply the respective rows of R."""
    I = torch.eye(*Q.shape)
    for j in range(Q.shape[1]):
        if torch.norm(Q[:, j] - I[:, j]) > torch.norm(-Q[:, j] - I[:, j]):
            Q[:, j] = -Q[:, j]
            R[j] = -R[j]
    return Q, R


def perturb_orthog_mat(Q, noise=0.05):
    """Perturb a matrix keeping columns orthogonal"""
    perturbed_Q = Q + noise * torch.randn_like(Q)
    print(f"initial Q rel err = {torch.norm(Q - perturbed_Q, p='fro') / torch.norm(Q, p='fro'):.2f}")
    return perturbed_Q


def plot(errors, fname='convergence.jpg', logsale=True):
    plt.figure(figsize=(8, 6))
    plt.plot(errors, label=r'$\| X - \widetilde{X} \|_F / \| X \|_F$')
    plt.xlabel('Iteration')

    plt.grid()

    if logsale:
        plt.yscale('log')
        plt.grid(True, which="both")

    plt.legend()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def experiment_R(Z, templates, X0, rank):
    """In this experiment, Q is set to Q_true. Only R is updated"""
    # Determine optimal Q
    Q, R = torch.linalg.qr(X0)
    Q, R = choose_optimal_qr(Q[:, :rank], R[:rank])

    # Sample random upper triangular R
    n2 = X0.size(1)
    R = torch.triu(torch.randn(rank, n2))

    n_iter = 5
    mu, tau = 1, 1  # Unused in this case
    errors_true = run_algorithm(Z, Q, R, templates, mu, tau, n_iter, X0,
                                update_z=False, update_R=True, update_Q=False)
    plot(errors_true, fname='convergence_R.jpg', logsale=False)


def experiment_Q(Z, templates, X0, rank):
    """In this experiment, R is set to R_true. Only Q is updated"""
    # Determine optimal R
    Q, R = torch.linalg.qr(X0)
    Q, R = choose_optimal_qr(Q[:, :rank], R[:rank])

    # Sample random orthogonal Q
    n1 = X0.size(0)
    Q = sample_ortog_mat(n1, rank).requires_grad_()

    n_iter = 5000
    mu = 0.5
    tau = 1.5e-4
    errors_true = run_algorithm(Z, Q, R, templates, mu, tau, n_iter, X0,
                                update_z=False, update_R=False, update_Q=True)
    plot(errors_true, fname='convergence_Q.jpg')


def experiment_close(Z, templates, X0, rank):
    """In this experiment, Initial Q is close to Q_true"""
    # Determine optimal Q and perturb it
    Q, R = torch.linalg.qr(X0)
    Q, R = choose_optimal_qr(Q[:, :rank], R[:rank])
    Q = perturb_orthog_mat(Q, noise=0.05).requires_grad_()

    # Sample random upper triangular R
    n2 = X0.size(1)
    R = torch.triu(torch.randn(rank, n2))

    n_iter = 5000
    mu = 0.01
    tau = 2e-4
    errors_true = run_algorithm(Z, Q, R, templates, mu, tau, n_iter, X0,
                                update_z=False, update_R=True, update_Q=True)
    plot(errors_true, fname='convergence_close.jpg')


def main():
    n1 = 10
    n2 = 15
    rank = 3
    condition_number = 1e1

    singular_values = torch.linspace(1, condition_number, rank)
    X0 = generate_matrix(n1, n2, singular_values)

    p = 0.5
    K = 50
    templates = [generate_template(n1, n2, p) for _ in range(K)]
    Z = torch.tensor([(X0 * Tk).sum() for Tk in templates])

    experiment_R(Z, templates, X0, rank)
    experiment_Q(Z, templates, X0, rank)
    experiment_close(Z, templates, X0, rank)


if __name__ == '__main__':
    torch.manual_seed(42)
    main()
