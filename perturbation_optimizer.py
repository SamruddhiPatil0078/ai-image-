import torch
import torch.nn.functional as F

def identity_loss(adv_emb, orig_emb, target_emb):

    sim_orig = F.cosine_similarity(adv_emb, orig_emb)
    sim_target = F.cosine_similarity(adv_emb, target_emb)

    # Strong directional push
    return (sim_orig - 3.5 * sim_target).mean()


def optimize_perturbation(model, image_tensor, orig_emb, target_emb,
                          steps=12, eps=12/255, alpha=3/255, device="cpu"):

    adv = image_tensor.clone().detach().to(device).float()

    for step in range(steps):

        adv.requires_grad_(True)

        adv_norm = (adv - 0.5) / 0.5
        adv_emb = model(adv_norm)

        loss = identity_loss(adv_emb, orig_emb.detach(), target_emb.detach())

        loss.backward()

        grad = adv.grad

        # small gaussian smoothing effect (reduces visible lines)
        grad = torch.nn.functional.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)

        adv = adv - alpha * grad.sign()

        perturb = torch.clamp(adv - image_tensor, -eps, eps)
        adv = torch.clamp(image_tensor + perturb, 0, 1).detach()

        if step % 3 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    return adv
