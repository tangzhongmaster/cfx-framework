"""
Point UNIQUE de verite pour "sur quel device tourne le framework" (CPU ou
GPU). Importe par tous les modules qui creent des tensors ou instancient
des nn.Module (backbones.py, decoders.py, search.py, search_variants.py,
criteria.py, margin.py, run_experiment.py, protocol.py) -- jamais de
torch.device("cuda") ecrit en dur ailleurs, pour n'avoir qu'UN endroit a
changer.

Par defaut : CUDA si disponible, sinon CPU (torch.cuda.is_available()).
Sur Colab avec un runtime GPU, ca suffit a tout faire tourner sur GPU sans
rien configurer -- mais DEVICE est fige au moment du PREMIER import de ce
module, donc appeler set_device() APRES avoir deja construit des backbones/
decodeurs ne les deplace pas retroactivement (il faudrait les reconstruire,
voir move_module_to_device ci-dessous si besoin ponctuel).

Usage dans les autres modules :
    from device import DEVICE
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    model = QuantileNet(dim).to(DEVICE)
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(name: str):
    """Change DEVICE explicitement (ex: set_device("cpu") pour forcer le
    CPU meme si un GPU est disponible). N'affecte PAS les objets deja
    construits avec l'ancien device -- a appeler tout en haut d'un script,
    avant de construire quoi que ce soit (build_setup, build_backbone,
    build_decoder, ...)."""
    global DEVICE
    DEVICE = torch.device(name)


def is_cuda() -> bool:
    return DEVICE.type == "cuda"
