import torch
import torch.nn.functional as F


def generate_target_embedding(orig_embedding, angle_rad=0.35):
    """
    Generate a reachable fake identity by rotating the original
    embedding slightly in feature space.

    orig_embedding : torch tensor shape [1,512]
    angle_rad      : rotation angle (in radians)
                     0.2 → weak change
                     0.35 → safe cloaking
                     0.6 → strong change
    """

    # Normalize original embedding
    orig = F.normalize(orig_embedding, dim=1)

    # Create orthogonal direction (different identity direction)
    noise = torch.randn_like(orig)
    noise = noise - (noise * orig).sum(dim=1, keepdim=True) * orig
    noise = F.normalize(noise, dim=1)

    # Rotate embedding toward new identity
    angle = torch.tensor(angle_rad, device=orig.device)
    target = torch.cos(angle) * orig + torch.sin(angle) * noise

    # Normalize final target identity
    target = F.normalize(target, dim=1)

    return target
