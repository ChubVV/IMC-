import torch
from tqdm import trange
import pandas as pd


def generate_ort_matrix(p, q):
    '''Generates an orthogonal matrix of size  p x q '''
    U = torch.randn(p, q)
    U = torch.linalg.qr(U)[0]
    return U


def generate_sparse_template(n1, n2, p):
    '''Generates a parsed mask'''
    rows = torch.randint(0, n1, (int(n1 * n2 * p),))
    cols = torch.randint(0, n2, (int(n1 * n2 * p),))
    values = torch.ones(rows.shape[0])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n1, n2))


def generate_matrix(p, q, singular_values):
    '''Generates a matrix of rank k '''
    rank = len(singular_values)
    U = generate_ort_matrix(p, rank)
    D = torch.diag(singular_values)
    V = generate_ort_matrix(q, rank)
    return U @ D @ V.T, U, V


def perturb_orthog_mat(Q, noise=0.05):
    '''Adds noise to the matrix''' 
    perturbed_Q = Q + noise * torch.randn_like(Q)
    return perturbed_Q


def loss(U, V, Lambda, z, Z, Tau, g, mu_g, mu_o, mu_e, E, E_prime):
    '''Target loss function '''
    K = len(Tau)
    rank = Lambda.shape[0]
    loss1 = -0.5 * torch.norm(Z - z)**2
    loss2 = -torch.sum(0.5 * (z - torch.stack(
        [torch.sum(Tau_k.to_dense() * (U @ Lambda @ V.T)) for Tau_k in Tau]))**2)
    loss3 = -0.5 * mu_g * torch.sum(g**2 * torch.diag(Lambda)**2)
    loss4 = -0.5 * mu_o * (torch.norm(U.T @ U - torch.eye(rank), 'fro')
                           ** 2 + torch.norm(V.T @ V - torch.eye(rank), 'fro')**2)
    loss5 = -0.5 * mu_e * (torch.norm(U - E, 'fro') **
                           2 + torch.norm(V - E_prime, 'fro')**2)

    return loss1 + loss2 + loss3 + loss4 + loss5


def adam_gradient_descent(U_hat, D_hat, V_hat, U, V, D, z, Z, Tau, g, E, E_prime, max_iter=5000, mu_g=0.01, mu_o=0.005, mu_e=0.01, lr=0.01):
    optimizer = torch.optim.RMSprop(
        [U, V, D, z], lr=lr, maximize=True, alpha=0.98, eps=1e-8)

    for i in trange(max_iter, desc="Обучение", unit="шаг", dynamic_ncols=True):
        optimizer.zero_grad()
        current_loss = loss(U, V, D, z, Z, Tau, g, mu_g,
                            mu_o, mu_e, E, E_prime)
        current_loss.backward()
        optimizer.step()

        if i >= max_iter // 1000 and i % 500 == 0:
            E = U.detach().clone()
            E_prime = V.detach().clone()

    return U, V, D, z


def experiment_close(X0, U, V, templates, Z, singular_values, max_iter):
    p, q = X0.size()
    rank = singular_values.size(0)
    X_pertub = perturb_orthog_mat(X0, noise=0.05)
    U_pertub, D_pertub, V_pertub = torch.linalg.svd(X_pertub)

    U_pertub = U_pertub[:, :rank].requires_grad_()
    V_pertub = V_pertub[:, :rank].requires_grad_()
    D_pertub = torch.diag(D_pertub[:rank]).requires_grad_()
    D = torch.diag(singular_values)
    z = Z.clone().requires_grad_()

    g = torch.linspace(12, 1, rank)
    E = generate_ort_matrix(p, rank)
    E_prime = generate_ort_matrix(q, rank)

    lr = 0.0001
    U_new, V_new, D_new, z_new = adam_gradient_descent(
        U, D, V, U_pertub, V_pertub, D_pertub, z, Z, templates, g, E, E_prime, lr=lr, max_iter=max_iter)

    true_error = torch.norm(U @ D @ V.T - U_new.detach() @ D_new.detach()
                            @ V_new.detach().T, p='fro') / torch.norm(U @ D @ V.T, p='fro')
    return true_error.item()


def main():
    p, q, rank = 500, 300, 5
    condition_number = 1e1
    singular_values = torch.linspace(1, condition_number, rank)
    X0, U, V = generate_matrix(p, q, singular_values)
    prob = 0.01
    K_list = [(p + q) * rank + 300]
    max_iter_list = [5000]

    results = []
    for K in K_list:
        templates = [generate_sparse_template(p, q, prob) for _ in range(K)]
        Z = torch.tensor([(X0 * Tau_k.to_dense()).sum()
                         for Tau_k in templates])
        templates = templates

        for max_iter in max_iter_list:
            error = experiment_close(
                X0, U, V, templates, Z, singular_values, max_iter)
            results.append({"K": K, "max_iter": max_iter, "error": error})

    df = pd.DataFrame(results)
    df.pivot(index="K", columns="max_iter",
             values="error").to_html("results3.html")
    print(df.pivot(index="K", columns="max_iter", values="error"))
