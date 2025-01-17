import torch
from tqdm import tqdm

from cayley_gd import cayley_transform


def vectorize_upper_triangular(A):
    """Extract the upper triangular part of a matrix and vectorize it"""
    upper_triangular_indices = torch.triu_indices(A.size(0), A.size(1), offset=0)
    upper_triangular_vector = A[upper_triangular_indices[0], upper_triangular_indices[1]]
    return upper_triangular_vector


def upper_triangular_from_vector(v, n1, n2):
    """Convert a vectorized upper triangular part of a matrix into an upper triangular matrix"""
    upper_triangular_indices = torch.triu_indices(n1, n2, offset=0)

    upper_triangular_matrix = torch.zeros((n1, n2))
    upper_triangular_matrix[upper_triangular_indices[0], upper_triangular_indices[1]] = v

    return upper_triangular_matrix


def loss_fn(Q, R, mu, z, templates):
    """Compute the loss of Q: \sum_k (z_k - <T_k, QR>)^2 + \mu \| Q - I \|^2"""
    QR = Q @ R
    loss = mu * ((Q - torch.eye(*Q.shape)) ** 2).sum()
    for zk, Tk in zip(z, templates):
        loss += (zk - (Tk * QR).sum()) ** 2

    return loss


def optimize_z(Z, Q, R, templates):
    """Optimize log-likelihood wrt z"""
    QR = Q @ R
    frobeinus_products = [(Tk * QR).sum() for Tk in templates]
    z = (Z + torch.tensor(frobeinus_products)) / 2
    return z


def optimize_R(z, Q, templates):
    """Optimize log-likelihood wrt R"""
    QTk = [vectorize_upper_triangular(Q.T @ Tk) for Tk in templates]
    coeffs = torch.stack(QTk)
    R_vec = torch.linalg.lstsq(coeffs, z).solution
    R = upper_triangular_from_vector(R_vec, Q.size(1), templates[0].size(1))
    return R


def cayley_gd_step(z, Q, R, templates, mu, tau):
    """Cayley_GD step wrt Q"""
    loss = loss_fn(Q, R, mu, z, templates)
    loss.backward()

    with torch.no_grad():
        Q.copy_(cayley_transform(Q, Q.grad, tau))

    Q.grad.zero_()
    return Q


def get_rel_error(Q, R, X0):
    """Relative error of QR ~ X (Frobenius norm)"""
    err = torch.norm(Q @ R - X0, p='fro') / torch.norm(X0, p='fro')
    return err.item()


def run_algorithm(Z, Q, R, templates, mu, tau, n_iter, X0, update_z=True, update_R=True, update_Q=True):
    """Run algorithm"""
    errors_true = [get_rel_error(Q, R, X0)]

    for _ in tqdm(range(n_iter)):
        with torch.no_grad():
            z = optimize_z(Z, Q, R, templates) if update_z else Z
            R = optimize_R(z, Q, templates) if update_R else R

        Q = cayley_gd_step(z, Q, R, templates, mu, tau) if update_Q else Q
        errors_true.append(get_rel_error(Q, R, X0))

    return errors_true
