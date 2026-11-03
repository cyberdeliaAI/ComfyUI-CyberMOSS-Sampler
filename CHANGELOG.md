# Changelog

## 0.1.1

- Accept the official OpenMOSS custom loader's MOSSImage class even when
  ComfyUI's independent native MOSSImage class is also available. Subclasses
  are supported; non-MOSS models and invalid skip layers are still rejected.
- Correct `reference` at full denoise to match the official full-model
  MOSSImageScheduler: shift 5, 1,000-step training-grid snapping, consecutive
  duplicate removal and a final zero. The 50-step last positive sigma is 0.024.
- Add `simple` for the previous continuous reference grid and use it in
  `turbo_reference`, preserving the Turbo/DMD baseline. Existing saved manual
  values are retained; reselect the preset to load its new schedule mode.
- Preserve continuous partial-denoise tails and their requested step budgets.
- Add official-scheduler numeric fixtures and regression tests for native and
  custom loader coexistence, subclasses, preset wiring and schedule budgets.

## 0.1.0

- First MOSS-Image version, based on CyberKrea Sampler 0.4.0.
- 128-channel Flux2 latent support: the sampler validates
  `(batch, 128, height/16, width/16)` and passes ComfyUI's downscale hint, so
  the standard Empty Latent Image is converted correctly.
- Fixed MOSS-Image flow shift of 5 for both schedule modes. The `reference`
  mode is the plain shift-5 grid (Euler with the simple scheduler).
- Constant `cfg` control with a negative-branch layer skip (`skip_layers`,
  default `22` for full models), applied through ComfyUI's patch-replace hook
  for the uncond pass only. An existing Skip Layer Guidance cond-batch function
  is left in charge.
- Presets: `turbo` and `turbo_reference` for SFT Turbo / DMD (8 steps, CFG 1),
  `full` and `full_reference` for Base, SFT and fine-tunes (CFG 3, layer 22).
- CFG above 1 without a negative input raises a clear error instead of sampling
  against zeroed conditioning.
- A non-MOSS model and out-of-range skip layers are rejected with a clear error.
- CyberMOSS Empty Latent with 1, 1.5, 2 and 3 MP tiers in 11 aspect ratios, all
  multiples of 16.
- Steps range 1-100. TAESD preview uses `taef2_decoder`.
