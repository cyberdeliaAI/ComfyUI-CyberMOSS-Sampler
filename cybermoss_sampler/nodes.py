"""Compact CyberMOSS sampler node for MOSS-Image.

The sampling engine is derived from ComfyUI-KreaPhoton via CyberKrea Sampler.
This node deliberately exposes one MODEL path only, which keeps LoRA and other
model patches intact.
"""

import contextlib
import math

from .guidance import parse_skip_layers
from .presets import DEFAULT_PRESET, PRESETS, SHIFT
from .sampling import run_sampling
from .schedules import SCHEDULE_MODES, build_sampling_schedule


CATEGORY = "CyberMOSS"
PREVIEW_METHODS = ["default", "auto", "latent2rgb", "taesd", "none"]
PRESET_NAMES = list(PRESETS)
SAMPLER_NAMES = ["euler", "euler_2m"]

_ORDER_FROM_SAMPLER_NAME = {"euler": 1, "euler_2m": 2}
_DEFAULTS = PRESETS[DEFAULT_PRESET]
_FLOAT_RANGES = {
    "cfg": (0.0, 20.0),
    "restart_frac": (0.0, 0.60), "sigma_r": (0.0, 1.0),
    "detail": (0.0, 1.0), "eta0": (0.0, 2.0),
    "sigma_gate": (0.0, 1.0), "contraction": (0.0, 1.0),
    "denoise": (0.0, 1.0),
}

_PREVIEW_TOOLTIP = (
    "Live preview during sampling. default follows ComfyUI's global setting; "
    "auto uses ComfyUI's automatic selection. taesd requires taef2_decoder "
    "in models/vae_approx and falls back to latent2rgb."
)
_NEGATIVE_TOOLTIP = (
    "Negative conditioning, used when cfg is not 1. MOSS-Image's reference "
    "uses an empty negative prompt. Required for the full presets."
)
_VAE_TOOLTIP = (
    "Optional. Fully decodes only the first image for a thumbnail on this node. "
    "The LATENT output is unchanged."
)


def resolve_settings(preset, steps=None, sampler=None, cfg=None, skip_layers=None,
                     restart_frac=None, sigma_r=None, plunge=None, detail=None,
                     eta0=None, sigma_gate=None, contraction=None,
                     schedule_mode=None, denoise=1.0):
    """Fill omitted values from the preset; explicit controls always win."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown CyberMOSS preset: {preset}")
    overrides = {
        "steps": steps, "sampler": sampler, "cfg": cfg, "skip_layers": skip_layers,
        "restart_frac": restart_frac, "sigma_r": sigma_r, "plunge": plunge,
        "detail": detail, "eta0": eta0, "sigma_gate": sigma_gate,
        "contraction": contraction,
        "schedule_mode": schedule_mode, "denoise": denoise,
    }
    settings = {"denoise": 1.0, **PRESETS[preset],
                **{k: v for k, v in overrides.items() if v is not None}}
    steps = settings["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer between 1 and 100")
    if settings["sampler"] not in _ORDER_FROM_SAMPLER_NAME:
        raise ValueError(f"Unknown sampler: {settings['sampler']}")
    if settings["schedule_mode"] not in SCHEDULE_MODES:
        raise ValueError(f"Unknown schedule mode: {settings['schedule_mode']}")
    if not isinstance(settings["plunge"], bool):
        raise ValueError("plunge must be a boolean")
    for name, (minimum, maximum) in _FLOAT_RANGES.items():
        try:
            value = float(settings[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
        settings[name] = value
    settings["skip_layer_list"] = parse_skip_layers(settings["skip_layers"])
    settings["order"] = _ORDER_FROM_SAMPLER_NAME[settings["sampler"]]
    settings["shift"] = SHIFT
    return settings


@contextlib.contextmanager
def _live_preview(method):
    if method not in PREVIEW_METHODS:
        raise ValueError(f"Unknown preview method: {method}")
    if method == "default":
        yield
        return
    try:
        import latent_preview
        from comfy.cli_args import args
    except ImportError:
        yield
        return

    previous = args.preview_method
    args.preview_method = {
        "auto": latent_preview.LatentPreviewMethod.Auto,
        "latent2rgb": latent_preview.LatentPreviewMethod.Latent2RGB,
        "taesd": latent_preview.LatentPreviewMethod.TAESD,
    }.get(method, latent_preview.LatentPreviewMethod.NoPreviews)
    try:
        yield
    finally:
        args.preview_method = previous


def _result_with_preview(output, vae):
    if vae is None:
        return (output,)

    import nodes as comfy_nodes

    images = vae.decode(output["samples"][:1])
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    ui = comfy_nodes.PreviewImage().save_images(
        images, filename_prefix="CyberMOSS"
    )["ui"]
    return {"ui": ui, "result": (output,)}


class CyberMOSSSampler:
    """Preset-driven MOSS-Image sampler with a single patched MODEL path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "preset": (PRESET_NAMES, {
                    "default": DEFAULT_PRESET,
                    "cybermoss_presets": PRESETS,
                    "tooltip": "turbo: 8-step models at CFG 1 (SFT Turbo, DMD LoRA). "
                               "full: Base, SFT and fine-tunes at CFG 3. "
                               "full_reference follows the official MOSS scheduler; "
                               "turbo_reference uses the simple shift-5 schedule.",
                }),
                "steps": ("INT", {
                    "default": _DEFAULTS["steps"],
                    "min": 1,
                    "max": 100,
                    "tooltip": "Filled by the preset; may be adjusted manually.",
                }),
                "sampler": (SAMPLER_NAMES, {
                    "default": _DEFAULTS["sampler"],
                    "tooltip": "Euler or AB2 history. euler_2m needs consecutive deterministic "
                               "steps; use eta0=0 to enable it throughout each descent.",
                }),
                "cfg": ("FLOAT", {
                    "default": _DEFAULTS["cfg"],
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Constant CFG. 1 for Turbo/DMD models (negative not evaluated). "
                               "3 is the official value for full models; 3-6 for short prompts.",
                }),
                "skip_layers": ("STRING", {
                    "default": _DEFAULTS["skip_layers"],
                    "tooltip": "Layers skipped in the negative (uncond) pass, e.g. '22' or "
                               "'22, 24, 26'. Only active when cfg is not 1. Empty disables it.",
                }),
                "restart_frac": ("FLOAT", {
                    "default": _DEFAULTS["restart_frac"],
                    "min": 0.0,
                    "max": 0.60,
                    "step": 0.01,
                    "tooltip": "Fraction of the existing step budget used by the restart. "
                               "Only active with cybermoss schedule and denoise=1. "
                               "Skipped when the budget is too small or sigma_r=0.",
                }),
                "sigma_r": ("FLOAT", {
                    "default": _DEFAULTS["sigma_r"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Restart sigma. Filled by the preset; may be adjusted manually.",
                }),
                "plunge": ("BOOLEAN", {
                    "default": _DEFAULTS["plunge"],
                    "tooltip": "Direct end step from sigma 0.75. Meant for few-step models; "
                               "only active with cybermoss schedule and denoise=1.",
                }),
                "detail": ("FLOAT", {
                    "default": _DEFAULTS["detail"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Detail amount. Filled by the preset; may be adjusted manually.",
                }),
                "eta0": ("FLOAT", {
                    "default": _DEFAULTS["eta0"],
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Ancestral noise strength.",
                }),
                "sigma_gate": ("FLOAT", {
                    "default": _DEFAULTS["sigma_gate"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Eta is disabled at or below this sigma. Gates below 0.35 "
                               "ramp to full strength at 0.35; higher gates use a hard cutoff.",
                }),
                "contraction": ("FLOAT", {
                    "default": _DEFAULTS["contraction"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Scales only the initial noise: 1 leaves it unchanged, "
                               "0 removes it. Restart and ancestral noise are separate.",
                }),
                "schedule_mode": (SCHEDULE_MODES, {
                    "default": _DEFAULTS["schedule_mode"],
                    "tooltip": "reference uses the official MOSS training grid at denoise 1; "
                               "simple uses the continuous shift-5 grid for Turbo/DMD. "
                               "Both ignore restart/plunge and use continuous tails below denoise 1. "
                               "Other manual controls still apply.",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "1 generates from noise; below 1 refines an existing latent "
                               "without restart/plunge. 0 passes it through unchanged. "
                               "Preserved when switching presets.",
                }),
                "preview_method": (PREVIEW_METHODS, {
                    "default": "default",
                    "tooltip": _PREVIEW_TOOLTIP,
                }),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": _NEGATIVE_TOOLTIP}),
                "vae": ("VAE", {"tooltip": _VAE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent_image, seed, preset,
               steps=None, sampler=None, cfg=None, skip_layers=None,
               restart_frac=None, sigma_r=None, plunge=None, detail=None,
               eta0=None, sigma_gate=None, contraction=None,
               schedule_mode=None, denoise=1.0, preview_method="default",
               negative=None, vae=None):
        settings = resolve_settings(
            preset, steps, sampler, cfg, skip_layers, restart_frac, sigma_r, plunge,
            detail, eta0, sigma_gate, contraction, schedule_mode, denoise,
        )
        if settings["denoise"] == 0:
            return _result_with_preview(latent_image, vae)
        sigmas = build_sampling_schedule(
            settings["steps"],
            schedule_mode=settings["schedule_mode"],
            denoise=settings["denoise"],
            shift=settings["shift"],
            restart_frac=settings["restart_frac"],
            sigma_r=settings["sigma_r"],
            plunge=settings["plunge"],
        )
        with _live_preview(preview_method):
            output = run_sampling(
                model,
                positive,
                negative,
                latent_image,
                sigmas,
                seed=seed,
                cfg=settings["cfg"],
                skip_layers=settings["skip_layer_list"],
                contraction=settings["contraction"],
                detail_amount=settings["detail"],
                order=settings["order"],
                eta0=settings["eta0"],
                sigma_gate=settings["sigma_gate"],
            )
        return _result_with_preview(output, vae)


NODE_CLASS_MAPPINGS = {
    "CyberMOSSSampler": CyberMOSSSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberMOSSSampler": "CyberMOSS Sampler",
}
