"""Shared sampler defaults, also sent to the frontend through node metadata.

MOSS-Image uses a fixed flow shift of 5 for both the 50-step models and the
8-step Turbo variant. Full reference follows the official MOSS scheduler;
Turbo reference preserves the simple schedule used by Turbo/DMD workflows.
The turbo and full presets are CyberMOSS starting points for manual tuning,
not guarantees for every model or prompt.
"""

SHIFT = 5.0
DEFAULT_PRESET = "turbo"

_REFERENCE_NEUTRAL = {
    "schedule_mode": "reference",
    "sampler": "euler",
    "restart_frac": 0.0,
    "sigma_r": 0.65,
    "plunge": False,
    "detail": 0.0,
    "eta0": 0.0,
    "sigma_gate": 0.10,
    "contraction": 1.0,
}

PRESETS = {
    # Few-step models (MOSS-Image SFT Turbo, DMD LoRA): CFG 1. The restart,
    # plunge and ancestral-noise values come from CyberKrea's Turbo presets.
    "turbo": {
        "schedule_mode": "cybermoss", "steps": 8, "sampler": "euler",
        "cfg": 1.0, "skip_layers": "",
        "restart_frac": 0.25, "sigma_r": 0.65, "plunge": True,
        "detail": 0.50, "eta0": 1.0, "sigma_gate": 0.10, "contraction": 0.70,
    },
    # Euler, simple-style shift-5 grid, 8 steps, CFG 1.
    "turbo_reference": {**_REFERENCE_NEUTRAL, "schedule_mode": "simple",
                        "steps": 8, "cfg": 1.0, "skip_layers": ""},
    # Full models (Base, SFT, fine-tunes): real CFG, no plunge or restart,
    # deterministic steps, uncond pass skips layer 22.
    "full": {
        "schedule_mode": "cybermoss", "steps": 30, "sampler": "euler",
        "cfg": 3.0, "skip_layers": "22",
        "restart_frac": 0.0, "sigma_r": 0.65, "plunge": False,
        "detail": 0.30, "eta0": 0.0, "sigma_gate": 0.10, "contraction": 1.0,
    },
    # Official MOSS-Image defaults: Euler, 50 steps, CFG 3, shift 5, layer 22.
    "full_reference": {**_REFERENCE_NEUTRAL, "steps": 50, "cfg": 3.0, "skip_layers": "22"},
}
