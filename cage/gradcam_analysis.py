"""GradCAM maps of WeakCNN on clean images and on denoiser outputs.

Every map is taken at WeakCNN's conv2 for the true label. WeakCNN's head is
linear, so the target logit also splits exactly into one term per position of
its 30x30 feature map, which measures evidence without GradCAM's spatial
averaging of gradients.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cage.evaluate import IndexedDataset, paired_gaussian_noise
from cage.gradcam import compute_gradcam

TOP_FRACTION = 0.2
# Correlations and overlaps compare maps at 32x32, coarse enough that
# per-pixel noise in the edit does not swamp them.
GRID = 32


def top_mask(maps: torch.Tensor, fraction: float = TOP_FRACTION) -> torch.Tensor:
    """1 on the top `fraction` positions of each map, 0 elsewhere."""
    flat = maps.flatten(1)
    k = max(1, int(fraction * flat.shape[1]))
    threshold = flat.topk(k, dim=1).values[:, -1:]
    return (flat >= threshold).float().view_as(maps)


def _flat_maps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # GradCAM is all zeros when no channel has a positive weight. A flat map has
    # no location, so comparisons involving one are NaN and left out of means.
    return (a.flatten(1).std(1) == 0) | (b.flatten(1).std(1) == 0)


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    flat = _flat_maps(a, b)
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    r = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
    return r.masked_fill(flat, float("nan"))


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ma, mb = top_mask(a), top_mask(b)
    overlap = (ma * mb).flatten(1).sum(1) / (ma + mb).clamp_max(1).flatten(1).sum(1)
    return overlap.masked_fill(_flat_maps(a, b), float("nan"))


def entropy(maps: torch.Tensor) -> torch.Tensor:
    """Entropy of each map as a distribution over positions, 1 for uniform."""
    p = maps.flatten(1).clamp_min(0)
    total = p.sum(1, keepdim=True)
    p = p / total.clamp_min(1e-12)
    h = -(p * p.clamp_min(1e-12).log()).sum(1) / torch.log(torch.tensor(float(p.shape[1])))
    return h.masked_fill(total[:, 0] == 0, float("nan"))


def _grid(maps: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(maps.unsqueeze(1), maps.shape[-1] // GRID).squeeze(1)


def _gradcam(cnn, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(compute_gradcam(cnn, x, y.tolist())).to(x.device)


def clean_reference(cnn, dataset, batch_size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """GradCAM and logit contributions of the clean images, once per critic."""
    reference = []
    for x, y, _ in DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=False):
        x, y = x.to(device), y.to(device).long().view(-1)
        with torch.no_grad():
            contributions = cnn.logit_contributions(x, y)
        reference.append((_gradcam(cnn, x, y), contributions))
    return reference


def gradcam_metrics(cnn, denoiser, dataset, reference, sigma: float, evaluation_seed: int, batch_size: int, device: torch.device) -> dict[str, float]:
    """Compare WeakCNN's GradCAM maps on clean images and on denoiser outputs.

    The edit is the output minus the clean image. The paste test puts the edit
    back on the clean image only inside the clean image's top-20% GradCAM
    region, or inside another image's region of the same shape as a control.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    loader = DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=False)
    for (x, y, indices), (cam_clean, contrib_clean) in zip(loader, reference):
        x, y = x.to(device), y.to(device).long().view(-1)
        noisy = paired_gaussian_noise(x.cpu(), indices, sigma, evaluation_seed).to(device)
        with torch.no_grad():
            output = denoiser(noisy)
            contrib_output = cnn.logit_contributions(output, y)
        cam_output = _gradcam(cnn, output, y)
        edit = output - x
        edit_size = _grid(edit.abs().mean(1))
        top_region = top_mask(cam_clean).unsqueeze(1)
        # Another image's mask keeps the region's size and shape but not its link to the content.
        shuffled = top_region.roll(1, dims=0)
        evidence_gain = contrib_output - contrib_clean
        values = {
            "gradcam_entropy_clean": entropy(_grid(cam_clean)),
            "gradcam_entropy_output": entropy(_grid(cam_output)),
            "gradcam_corr": pearson(_grid(cam_clean), _grid(cam_output)),
            "gradcam_iou": iou(_grid(cam_clean), _grid(cam_output)),
            "edit_corr_clean_gradcam": pearson(edit_size, _grid(cam_clean)),
            "edit_corr_output_gradcam": pearson(edit_size, _grid(cam_output)),
            "evidence_gain_top": (evidence_gain * top_mask(contrib_clean)).flatten(1).sum(1),
            "evidence_gain_total": evidence_gain.flatten(1).sum(1),
        }
        with torch.no_grad():
            for name, mask in (("none", torch.zeros_like(top_region)), ("top", top_region), ("shuffled", shuffled), ("all", torch.ones_like(top_region))):
                values[f"accuracy_paste_{name}"] = (cnn(x + mask * edit).argmax(1) == y).float()
        values["flat_clean_gradcam"] = (cam_clean.flatten(1).std(1) == 0).float()
        for name, value in values.items():
            valid = ~value.isnan()
            totals[name] = totals.get(name, 0.0) + value[valid].sum().item()
            counts[name] = counts.get(name, 0) + int(valid.sum().item())
    return {name: totals[name] / max(counts[name], 1) for name in totals}
