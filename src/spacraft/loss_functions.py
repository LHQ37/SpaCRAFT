import torch
import torch.nn.functional as F


def adversarial_loss_g(reconstructed_source):
    """Generator loss for reconstructed samples."""
    return torch.log1p(torch.exp(-reconstructed_source)).mean()


def adversarial_loss_d(input_source, reconstructed_source):
    """Discriminator loss for input and reconstructed samples."""
    return (
        torch.log1p(torch.exp(-input_source))
        + torch.log1p(torch.exp(reconstructed_source))
    ).mean()


def mse_loss(inputs, reconstructed):
    """Mean-squared feature reconstruction loss."""
    return torch.mean((reconstructed - inputs) ** 2)


def contrastive_clustering_loss(z, graph_neigh, temperature=0.5):
    """Graph-neighborhood InfoNCE loss."""
    n_nodes = z.shape[0]
    z_norm = F.normalize(z, p=2, dim=1)
    similarity = torch.matmul(z_norm, z_norm.T) / temperature
    diagonal = torch.eye(n_nodes, device=z.device, dtype=torch.bool)
    positives = (graph_neigh > 0) & ~diagonal
    if positives.sum(dim=1).min() == 0:
        return torch.zeros((), device=z.device)
    exponentiated = torch.exp(similarity).masked_fill(diagonal, 0.0)
    positive_sum = (exponentiated * positives).sum(dim=1).clamp_min(1e-8)
    all_sum = exponentiated.sum(dim=1).clamp_min(1e-8)
    return -torch.log(positive_sum / all_sum).mean()
