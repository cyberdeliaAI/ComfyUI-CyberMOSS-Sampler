"""Euler/AB2 sampling with detail adjustment, restart and gated ancestral noise.

One guider keeps the workflow's patched model active for the full trajectory.
The engine is derived from ComfyUI-KreaPhoton via ComfyUI-CyberKrea-Sampler;
see README for attribution.
"""
import math

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import torch

from .guidance import CyberMOSSGuider
from .noise import contract_noise


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _detail_envelope(p: float, start: float, end: float, peak: float) -> float:
    if end <= start:
        return 0.0
    u = (p - start) / (end - start)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    peak = max(0.05, min(0.95, peak))
    w = u / peak if u < peak else (1.0 - u) / (1.0 - peak)
    return _smoothstep(w)


GATE_HI = 0.35


def _gated_eta(sigma_next: float, eta0: float, sigma_gate: float, gate_hi: float = GATE_HI) -> float:
    """No noise at/below the gate; ramp to eta0 at gate_hi.

    A gate at/above gate_hi uses a hard cutoff because the ramp has no width.
    """
    if sigma_next <= sigma_gate:
        return 0.0
    if gate_hi <= sigma_gate:
        return eta0
    u = (sigma_next - sigma_gate) / (gate_hi - sigma_gate)
    return eta0 * _smoothstep(u)


@torch.no_grad()
def cybermoss_sampler_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                            detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                            order=1, eta0=0.0, sigma_gate=0.10, restart_seed=0):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    gen = torch.Generator(device="cpu").manual_seed((int(restart_seed) + 0x5EED) & 0xffffffffffffffff)
    old_d = None
    old_dt = None

    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])

        # Schedules encode a restart as an ascending jump from zero.
        if s_next > s_cur:
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (1.0 - s_next) * x + s_next * eps
            old_d = None
            old_dt = None
            continue

        if s_cur <= 0.0:
            continue

        # Skip the detail adjustment on large jumps and the final readout.
        p = i / max(n - 1, 1)
        is_final = s_next <= 0.0
        is_plunge = (s_cur - s_next) > 0.25
        if is_final or is_plunge:
            a = 0.0
        else:
            a = detail_amount * _detail_envelope(p, detail_start, detail_end, detail_peak)
            a = max(-1.0, min(1.0, a))
        sigma_model = max(1e-4, s_cur - a * (s_cur - s_next))

        denoised = model(x, sigma_model * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})

        if is_final:
            x = denoised
            old_d = None
            old_dt = None
            continue

        eta = _gated_eta(s_next, eta0, sigma_gate)
        if eta > 0.0:
            # ComfyUI's rectified-flow ancestral update. A stochastic jump
            # invalidates the deterministic derivative history used by AB2.
            downstep_ratio = 1.0 + (s_next / s_cur - 1.0) * eta
            sigma_down = s_next * downstep_ratio
            alpha_next = 1.0 - s_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = max(0.0, s_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2) ** 0.5
            ratio = sigma_down / s_cur
            x = ratio * x + (1.0 - ratio) * denoised
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (alpha_next / alpha_down) * x + eps * renoise_coeff
            old_d = None
            old_dt = None
        else:
            d = (x - denoised) / s_cur
            dt = s_next - s_cur
            if order >= 2 and old_d is not None and abs(dt) <= 0.25:
                # Variable-step AB2; 1.5*d - 0.5*old_d only holds for equal dt.
                d_use = d + (dt / (2.0 * old_dt)) * (d - old_d)
            else:
                d_use = d
            x = x + d_use * dt
            old_d = d
            old_dt = dt

    return x


LATENT_CHANNELS = 128


def zero_conditioning(cond):
    """Copy positive conditioning with zeroed embeddings for CFG=1 sampling."""
    out = []
    for t, d in cond:
        d = d.copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t), d])
    return out


def _moss_layer_count(model):
    """Layer count of a MOSS-Image MODEL, or None when it cannot be inspected.

    Raises for a recognizable non-MOSS model, so a wrong model fails clearly
    instead of sampling with MOSS settings.
    """
    base = getattr(model, "model", None)
    if base is None:
        return None
    name = type(base).__name__
    try:
        import comfy.model_base
        moss_class = getattr(comfy.model_base, "MOSSImage", None)
    except ImportError:
        moss_class = None
    # OpenMOSS's custom loaders define their own MOSSImage BaseModel subclass.
    # It remains valid when ComfyUI also provides a native MOSSImage class.
    is_moss = (
        moss_class is not None and isinstance(base, moss_class)
    ) or any(cls.__name__ == "MOSSImage" for cls in type(base).__mro__)
    if not is_moss:
        raise ValueError(
            f"CyberMOSS requires a MOSS-Image model; got {name}. "
            "Use CyberKrea Sampler for Krea 2."
        )
    layers = getattr(getattr(base, "diffusion_model", None), "layers", None)
    try:
        return len(layers)
    except TypeError:
        return None


def _prepare_latent(model, latent_dict):
    samples = latent_dict["samples"]
    ratio = latent_dict.get("downscale_ratio_spacial")
    if ratio is None:
        return comfy.sample.fix_empty_latent_channels(model, samples)
    return comfy.sample.fix_empty_latent_channels(model, samples, ratio)


def run_sampling(model, positive, negative, latent_dict, sigmas, *, seed,
                 cfg=1.0, skip_layers=(), contraction=1.0,
                 detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                 order=1, eta0=0.0, sigma_gate=0.10):
    """Sample one MOSS-Image batch using the same patched model throughout."""
    layer_count = _moss_layer_count(model)
    latent = _prepare_latent(model, latent_dict)
    if latent.ndim != 4 or latent.shape[1] != LATENT_CHANNELS:
        raise ValueError(
            "CyberMOSS requires a MOSS-Image / Flux2 latent with shape "
            f"(batch, 128, height/16, width/16); got {tuple(latent.shape)}. "
            "Connect a MOSS-Image model and a compatible latent, such as CyberMOSS Empty Latent."
        )

    cfg = float(cfg)
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("cfg must be finite and zero or higher")
    use_cfg = not math.isclose(cfg, 1.0)
    skip_layers = tuple(skip_layers) if use_cfg else ()
    if layer_count is not None:
        outside = [layer for layer in skip_layers if layer >= layer_count]
        if outside:
            raise ValueError(
                f"skip_layers {outside} is outside this model's layers 0-{layer_count - 1}."
            )
    if negative is None:
        if use_cfg:
            raise ValueError(
                f"CFG {cfg:g} needs negative conditioning. MOSS-Image uses an empty negative "
                "prompt: connect a text encode node with empty text, or set cfg to 1."
            )
        negative = zero_conditioning(positive)

    noise = comfy.sample.prepare_noise(latent, seed, latent_dict.get("batch_index"))
    noise = contract_noise(noise, strength=contraction)

    sampler = comfy.samplers.KSAMPLER(cybermoss_sampler_loop, extra_options={
        "detail_amount": float(detail_amount), "detail_start": float(detail_start),
        "detail_end": float(detail_end), "detail_peak": float(detail_peak),
        "order": int(order), "eta0": float(eta0), "sigma_gate": float(sigma_gate),
        "restart_seed": int(seed),
    })

    guider = CyberMOSSGuider(model, cfg=cfg, skip_layers=skip_layers)
    guider.set_conds(positive, negative)

    try:
        import latent_preview
        sampling_steps = int(((sigmas[:-1] > 0) & (sigmas[1:] <= sigmas[:-1])).sum())
        preview_callback = latent_preview.prepare_callback(model, sampling_steps)
        completed = 0

        def callback(_step, denoised, x, _total):
            nonlocal completed
            preview_callback(completed, denoised, x, sampling_steps)
            completed += 1
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(noise, latent, sampler, sigmas,
                            denoise_mask=latent_dict.get("noise_mask"),
                            callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(),
                         dtype=comfy.model_management.intermediate_dtype())
    out = latent_dict.copy()
    out["samples"] = samples
    # The output is a MOSS latent now; drop hints that described an empty input latent.
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    return out
