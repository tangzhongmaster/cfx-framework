"""
Le D(.;y_t) retenu : flow de normalisation conditionnel (RealNVP), seule
option (Section 4.1, Figure 2 -- VAE / flow autoregressif / diffusion
etaient d'anciennes alternatives de benchmark, retirees du code, voir
config.py niveau 3 pour la justification du choix). Interface commune :
- encode(x, y_t) -> z0
- decode(z, y_t) -> x'
- log_prob(x, y_t) -> vraisemblance exacte
"""

import torch
import torch.nn as nn
import numpy as np

from device import DEVICE


class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden=32):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )
        self.dim = dim

    def forward(self, u, y_t, reverse=False):
        # device derive de u (le tensor reellement en jeu), pas de DEVICE
        # global -- robuste meme si ce layer est un jour appele sur un
        # tensor deliberement laisse sur CPU.
        y = torch.as_tensor(y_t, dtype=torch.float32, device=u.device).reshape(-1, 1)
        u_masked = u * self.mask
        st = self.net(torch.cat([u_masked, y], dim=-1))
        s, t = st[..., :self.dim], st[..., self.dim:]
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        if not reverse:
            u_out = u_masked + (1 - self.mask) * (u * torch.exp(s) + t)
            log_det = (s * (1 - self.mask)).sum(-1)
            return u_out, log_det
        else:
            u_out = u_masked + (1 - self.mask) * ((u - t) * torch.exp(-s))
            return u_out, None


class ConditionalFlow(nn.Module):
    """RealNVP conditionnel (eq:coupling, eq:flow-nll). Inversible exactement
    -- erreur de reconstruction = 0 par construction, pas par convergence."""

    def __init__(self, dim, n_layers=6, hidden=32):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1.0
            layers.append(CouplingLayer(dim, mask, hidden))
        self.layers = nn.ModuleList(layers)

    def encode(self, x, y_t):
        u = x.clone()
        if u.dim() == 1:
            u = u.unsqueeze(0)
        for layer in self.layers:
            u, _ = layer(u, y_t, reverse=False)
        return u.squeeze(0)

    def decode(self, z, y_t):
        u = z.clone()
        if u.dim() == 1:
            u = u.unsqueeze(0)
        for layer in reversed(self.layers):
            u, _ = layer(u, y_t, reverse=True)
        return u.squeeze(0)

    def log_prob(self, x, y_t):
        u = x.clone()
        if u.dim() == 1:
            u = u.unsqueeze(0)
        log_det_total = 0.0
        for layer in self.layers:
            u, log_det = layer(u, y_t, reverse=False)
            log_det_total = log_det_total + log_det
        base_log_prob = -0.5 * (u ** 2).sum(-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return (base_log_prob + log_det_total).squeeze(0)

    def nll_loss(self, x, y_t):
        return -self.log_prob(x, y_t).mean()


def build_decoder(config, X_train, dim):
    from config import Decoder
    if config.decoder == Decoder.FLOW:
        return ConditionalFlow(dim).to(DEVICE)
    raise ValueError(config.decoder)


def train_decoder(decoder, X_train, y_train, epochs=1000, lr=1e-3):
    """Entrainement par NLL exacte (flow, seul decodeur restant)."""
    decoder = decoder.to(DEVICE)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        opt.zero_grad()
        loss = decoder.nll_loss(Xt, yt)
        loss.backward()
        opt.step()
    return decoder
