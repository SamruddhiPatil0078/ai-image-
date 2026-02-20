import torch
import torch.nn.functional as F

def identity_loss(adv_emb, orig_emb, margin=0.35):

    sim_orig = F.cosine_similarity(adv_emb, orig_emb)

    # Only push if similarity is above margin
    loss = F.relu(sim_orig - margin)

    return loss.mean()
   

def optimize_perturbation(model, image_tensor, orig_emb, target_emb,
                          steps=30, eps=10/255, alpha=1/255, device="cpu"):

    adv = image_tensor.clone().detach().to(device).float()

    for step in range(steps):

        adv.requires_grad_(True)

        adv_norm = (adv - 0.5) / 0.5
        adv_emb = model(adv_norm)

        loss = identity_loss(adv_emb, orig_emb.detach(), target_emb.detach())

        loss.backward()

        grad = adv.grad

        # ---- LOW FREQUENCY FILTER (important) ----
        # remove sharp pixel noise (patches)
        grad = torch.nn.functional.avg_pool2d(grad, 5, 1, 2)

        # normalize gradient strength
        grad = grad / (grad.abs().mean() + 1e-8)

        # smooth perturbation instead of pixel noise
        adv = adv - alpha * grad.sign()

        perturb = torch.clamp(adv - image_tensor, -eps, eps)
        adv = torch.clamp(image_tensor + perturb, 0, 1).detach()

        if step % 3 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    return adv
