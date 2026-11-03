"""Analytic rectified-flow schedules with an optional re-noise jump."""

import math

import torch

from .presets import SHIFT


TIMESTEPS = 10000
MOSS_TRAINING_STEPS = 1000
PLUNGE_SIGMA_FLOOR = 0.75
SCHEDULE_MODES = ["cybermoss", "reference", "simple"]


def build_sampling_schedule(n_steps: int, *, schedule_mode="cybermoss", denoise=1.0,
                            shift=SHIFT, restart_frac=0.0, sigma_r=0.6, plunge=False):
    """Full trajectory or plain partial descent for latent refinement.

    At full denoise, reference follows OpenMOSS's nearest-training-grid
    scheduler; simple keeps the continuous shift grid used by Turbo/DMD.
    Partial denoising uses a continuous tail in both modes so snapping cannot
    collapse its requested step budget. Construct only the tail, even for
    tiny denoise.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("steps must be a positive integer")
    if schedule_mode not in SCHEDULE_MODES:
        raise ValueError(f"Unknown schedule mode: {schedule_mode}")
    if not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("denoise must be finite and between 0 and 1")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and greater than zero")
    if denoise == 0:
        return torch.empty(0, dtype=torch.float32)
    if schedule_mode == "cybermoss" and denoise == 1:
        return build_schedule(n_steps, shift, restart_frac, sigma_r, plunge)
    if schedule_mode == "reference" and denoise == 1:
        return _reference_schedule(n_steps, shift)
    expanded_steps = n_steps / denoise
    if not math.isfinite(expanded_steps):
        raise ValueError("denoise is too small")
    total = int(expanded_steps)
    t = torch.arange(n_steps, -1, -1, dtype=torch.float64) / float(total)
    if schedule_mode == "cybermoss":
        t_lo = shift / (shift + (TIMESTEPS - 1.0))
        t = t_lo + (1 - t_lo) * t
    sigmas = shift * t / (1 + (shift - 1) * t)
    sigmas[-1] = 0
    return sigmas.to(dtype=torch.float32)


def _reference_schedule(n_steps: int, shift: float) -> torch.Tensor:
    """OpenMOSS MOSSImageScheduler grid (training_shift=1), ending at zero.

    See OpenMOSS-Team/MOSS-Image-SFT/pipeline_moss_image.py on Hugging Face.
    Choose the lower training sigma on a tie, and remove consecutive repeats,
    matching the official scheduler without a second flow shift.
    """
    lower_time = shift / (MOSS_TRAINING_STEPS + shift - 1)
    times = torch.linspace(1.0, lower_time, n_steps, dtype=torch.float32)
    desired = shift * times / (1 + (shift - 1) * times)
    training = torch.linspace(1.0, 1.0 / MOSS_TRAINING_STEPS,
                              MOSS_TRAINING_STEPS, dtype=torch.float32).flip(0)
    upper = torch.searchsorted(training, desired).clamp(max=MOSS_TRAINING_STEPS - 1)
    candidates = torch.stack(((upper - 1).clamp(min=0), upper), dim=1)
    distances = (training[candidates] - desired[:, None]).abs()
    nearest = candidates.gather(1, distances.argmin(dim=1, keepdim=True)).flatten()
    return torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1)))


def sigma_from_t(t: float, shift: float) -> float:
    """Flow time shift."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def t_from_sigma(sigma: float, shift: float) -> float:
    return sigma / (shift - (shift - 1.0) * sigma)


def build_schedule(n_steps: int, shift: float = SHIFT, restart_frac: float = 0.0,
                   sigma_r: float = 0.6, plunge: bool = False) -> torch.Tensor:
    """Start at sigma 1, end at 0, and preserve the requested sampling budget.

    A restart reserves some evaluations for a linear descent from sigma_r.
    Its ascending re-noise jump does not evaluate the model. If the budget is
    too small for both segments, or sigma_r is zero, use the whole budget for
    the first descent. A one-step schedule is always [1, 0].

    Plunge stops the first descent at 0.75 before predicting the clean latent.
    It needs two evaluations; a restart after it needs at least three in total.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("shift must be finite and greater than zero")
    for name, value in (("restart_frac", restart_frac), ("sigma_r", sigma_r)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")

    min_structure_steps = 2 if plunge else 1
    n_r = 0
    if restart_frac > 0.0 and sigma_r > 0.0 and n_steps > min_structure_steps:
        n_r = min(max(1, round(n_steps * restart_frac)), n_steps - min_structure_steps)
    n_m = n_steps - n_r

    sigs = []
    if plunge and n_m >= 2:
        t_lo = t_from_sigma(PLUNGE_SIGMA_FLOOR, shift)
        for i in range(n_m):
            t = 1.0 - (1.0 - t_lo) * (i / (n_m - 1))
            sigs.append(sigma_from_t(t, shift))
    else:
        t_lo = shift / (shift + (TIMESTEPS - 1.0))
        for i in range(n_m):
            t = 1.0 - (1.0 - t_lo) * (i / n_m)
            sigs.append(sigma_from_t(t, shift))
    sigs.append(0.0)

    if n_r:
        sigs.extend(sigma_r * (1.0 - j / n_r) for j in range(n_r))
        sigs.append(0.0)

    return torch.tensor(sigs, dtype=torch.float32)
