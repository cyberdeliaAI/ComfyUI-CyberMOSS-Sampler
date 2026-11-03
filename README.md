# ComfyUI-CyberMOSS-Sampler

<p align="center">
  <img src="assets/icon.png" alt="CyberMOSS Sampler icon" width="200">
</p>

[![Tests](https://github.com/cyberdeliaAI/ComfyUI-CyberMOSS-Sampler/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberdeliaAI/ComfyUI-CyberMOSS-Sampler/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A compact MOSS-Image sampler and resolution-aware empty latent node for
workflows that already patch the model with LoRAs.

One model connection keeps the model supplied by your LoRA chain active
throughout sampling. Choose a preset for your MOSS-Image variant, then adjust
any control independently. CyberMOSS uses its own node identifiers and category
so it can coexist with CyberKrea Sampler and other sampler packs.

CyberMOSS is the MOSS-Image sibling of
[ComfyUI-CyberKrea-Sampler](https://github.com/cyberdeliaAI/ComfyUI-CyberKrea-Sampler).
The sampling engine is derived from
[ComfyUI-KreaPhoton](https://github.com/Kostik2702/ComfyUI-KreaPhoton)
by Kostiantyn Hrytsuk.

## Node

`CyberMOSS Sampler` is under the `CyberMOSS` category.

### Connections

| Option | Description |
|---|---|
| `model` | The final MOSS-Image model from your model chain. Connect it after LoRAs. |
| `positive` | Positive conditioning from your prompt. |
| `latent_image` | A 128-channel MOSS-Image / Flux2 latent at 1/16 of the image size, such as the output from CyberMOSS Empty Latent. ComfyUI's own Empty Latent Image also works; it is converted before sampling. |
| `seed` | Controls the initial noise and all seeded sampling-noise streams for reproducible runs. |
| `negative` | Negative conditioning, used when `cfg` is not 1. Required for the full presets. MOSS-Image's reference uses an empty negative prompt: a text encode node with empty text. Ignored at CFG 1. |
| `vae` | Optional. Fully decodes the first image in the batch and displays a thumbnail on the node. This costs a full VAE decode for that image; the complete LATENT output is unchanged. |

### Sampler controls

Selecting a preset immediately fills in all visible control values. You can
then change any individual value without losing the rest of that preset.

| Option | Description |
|---|---|
| `preset` | Loads `turbo`, `turbo_reference`, `full` or `full_reference` values, including CFG, layer skipping and the schedule mode. Preview and denoise are preserved. |
| `steps` | Number of denoising evaluations in the sampler loop, from 1 to 100. With CFG above 1 each evaluation runs the model twice. |
| `sampler` | `euler` is the first-order method used by all presets. `euler_2m` uses variable-step Adams–Bashforth history on consecutive deterministic steps; set `eta0=0` to use it. Restarts and stochastic steps reset its history. |
| `cfg` | Constant classifier-free guidance. `1` for Turbo and DMD models: the negative is not evaluated. `3` is the official value for full models; 3 to 6 can help short prompts. |
| `skip_layers` | Layers skipped in the negative (uncond) pass, such as `22` or `22, 24, 26`. Only active when `cfg` is not 1. Empty disables it. See [Layer skipping](#layer-skipping). |
| `restart_frac` | Fraction of the existing step budget reserved for the restart pass. Active only with `schedule_mode=cybermoss` and `denoise=1`. Skipped if the budget cannot fit both descents. |
| `sigma_r` | Noise level to jump back to at the restart. Higher values make the restart stronger. `0` disables restart and keeps the entire step budget for the first descent. |
| `plunge` | Takes a direct end step from sigma 0.75 before the restart refines the image. Meant for few-step models. Active only with `schedule_mode=cybermoss` and `denoise=1`. |
| `detail` | Strength of the detail sigma adjustment. Higher values emphasize fine detail; excessive values can look harsh. |
| `eta0` | Maximum gated ancestral-noise strength during the middle of sampling. `0` disables this extra noise injection. |
| `sigma_gate` | Eta noise is disabled at or below this sigma. Below `0.35`, strength ramps up to `eta0` at sigma `0.35`. A gate of `0.35` or higher uses a hard cutoff. |
| `contraction` | Multiplies the initial noise: `1.0` leaves it unchanged, `0.70` is the turbo value, and `0` removes it. Restart and ancestral noise are separate. Very low values are experimental. |
| `schedule_mode` | `cybermoss` uses the CyberMOSS schedule with plunge and restart. At full denoise, `reference` follows the official MOSS training-grid scheduler; `simple` uses the continuous shift-5 grid for Turbo/DMD workflows. Both ignore plunge and restart. Below full denoise, `reference` and `simple` use a continuous schedule tail. Other manual controls still apply. |
| `denoise` | `1.0` generates from noise. Values below 1 refine an existing or upscaled latent using the final requested steps of a longer, plain schedule, without plunge/restart. `0` returns the latent unchanged (an optional VAE thumbnail can still be decoded). |
| `preview_method` | `default` follows ComfyUI's global preview setting. Explicit overrides are `auto`, `latent2rgb`, `taesd`, and `none`. TAESD requires `taef2_decoder` in `models/vae_approx`. The previous global setting is restored after sampling. |

### Presets

MOSS-Image uses a fixed flow shift of 5 for both the full models and the Turbo
variant. The presets differ in CFG, step count and sampling extras.

| Preset | For | Steps | CFG | Skip layers | Schedule | Extras |
|---|---|---:|---:|---|---|---|
| `turbo` | SFT Turbo, DMD LoRA | 8 | 1 | none | cybermoss | restart 0.25 at 0.65, plunge, detail 0.50, eta 1.0, contraction 0.70 |
| `turbo_reference` | SFT Turbo, DMD LoRA | 8 | 1 | none | simple | neutral |
| `full` | Base, SFT, fine-tunes such as Kromoss | 30 | 3 | 22 | cybermoss | detail 0.30, otherwise neutral |
| `full_reference` | Base, SFT, fine-tunes such as Kromoss | 50 | 3 | 22 | reference | neutral |

`full_reference` follows the official full-model settings: Euler, 50 steps,
CFG 3, shift 5 and layer 22 skipped in the negative pass. At `denoise=1`,
its sigmas match
[OpenMOSS's MOSSImageScheduler](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/pipeline_moss_image.py):
the shifted inference grid is snapped to the nearest point on the 1,000-step
training grid, consecutive duplicates are removed, and zero is appended.
At 50 steps the last positive sigma is 0.024.

`turbo_reference` keeps the continuous simple-style shift-5 schedule at 8 steps
and CFG 1. It is a Turbo/DMD baseline, not the full-model reference scheduler;
follow your adapter's own sampling instructions when they differ.
`turbo` and `full` are starting points
for tuning, not guarantees. The `turbo` extras come from CyberKrea's Turbo
presets and have not been validated on MOSS-Image checkpoints.

Few-step adapters that ship their own sigmas, such as the 4-step rapid-moss
node, should be used with their own sampler and sigma outputs.

### Layer skipping

MOSS-Image's reference pipeline skips layer 22 in the negative branch of CFG
(`uncond_skip_layers=[22]`, zero-based). The positive pass runs the full model.
The user guide also mentions layers 22, 24 and 26 as options that can improve
quality.

CyberMOSS applies this through ComfyUI's standard `dit` / `single_block`
patch-replace hook, only for the uncond pass. LoRAs and other model patches stay
active in both passes. It is the same mechanism as ComfyUI's
`SkipLayerGuidanceDiTSimple` node with `single_layers` set to `22`.

Use one or the other. If the model already carries a Skip Layer Guidance
cond-batch function, CyberMOSS leaves it in charge, does not apply
`skip_layers`, and logs a warning.

### Refinement and latent upscaling

For a second pass, connect an already generated, VAE-encoded or enlarged latent
with the same MOSS-Image model and matching positive conditioning. Start with
`preset=turbo_reference`, `steps=3` and `denoise=0.30` for a Turbo model, or
`preset=full_reference` with around 15 steps for a full model. These are
starting points for testing, not a guarantee of image quality. Lower denoise
generally preserves more of the source; higher denoise allows more changes.
An empty latent has no image to refine.

For `0 < denoise < 1`, the schedule uses the final `steps` transitions from a
plain schedule with `int(steps / denoise)` total steps. This happens in all
schedule modes; plunge and restart are bypassed, while detail, eta and contraction
remain manually controllable. In `reference` mode refinement deliberately uses
the same continuous tail as `simple`, since snapping a long expanded schedule
to the training grid would merge steps and reduce the requested budget.
Denoise is a schedule fraction, not a direct sigma value.

Preset changes preserve denoise. Set it back to `1.0` when returning to full
text-to-image generation.

At full denoise, one step always uses the full `[1, 0]` descent. A restart needs at least two
steps without plunge, or three with plunge. Smaller budgets fall back to a
single descent while preserving the requested evaluation count.

### API

The frontend obtains preset and resolution tables from Python's node metadata;
there are no separate JavaScript copies.

ComfyUI API prompts must include the visible required controls. The supplied
values take precedence: changing only the `preset` label does not overwrite
them. Headless clients can obtain the templates from
`GET /object_info/CyberMOSSSampler`, at
`CyberMOSSSampler.input.required.preset[1].cybermoss_presets`, and merge their
overrides before submitting the prompt. Direct Python callers can use
`resolve_settings("full", steps=40)` to fill omitted values from a preset.

## CyberMOSS Empty Latent

`CyberMOSS Empty Latent` creates the 128-channel latent expected by MOSS-Image
(Flux2 VAE, 1/16 of the image size) and outputs the selected `width` and
`height` as integers. Choose a size tier first; the resolution dropdown then
shows only the matching dimensions.

| Option | Description |
|---|---|
| `size` | Selects the S, M, L, or XL megapixel tier, filters the resolution dropdown, and preserves the selected aspect ratio. |
| `resolution` | Selects the concrete width, height, and aspect ratio within the chosen tier. |
| `batch_size` | Number of empty latents generated in one batch. Higher values require more VRAM. |

| Aspect | S (~1.0 MP) | M (~1.5 MP) | L (~2.0 MP) | XL (~3.0 MP) |
|---|---|---|---|---|
| 1:1 | 1024x1024 | 1248x1248 | 1456x1456 | 1776x1776 |
| 4:3 | 1152x864 | 1472x1104 | 1664x1248 | 2048x1536 |
| 3:4 | 864x1152 | 1104x1472 | 1248x1664 | 1536x2048 |
| 3:2 | 1248x832 | 1536x1024 | 1776x1184 | 2160x1440 |
| 2:3 | 832x1248 | 1024x1536 | 1184x1776 | 1440x2160 |
| 16:9 | 1360x768 | 1680x944 | 1936x1088 | 2304x1296 |
| 9:16 | 768x1360 | 944x1680 | 1088x1936 | 1296x2304 |
| 21:9 | 1568x672 | 1904x816 | 2240x960 | 2688x1152 |
| 9:21 | 672x1568 | 816x1904 | 960x2240 | 1152x2688 |
| 5:4 | 1120x896 | 1440x1152 | 1600x1280 | 2000x1600 |
| 4:5 | 896x1120 | 1152x1440 | 1280x1600 | 1600x2000 |

Every dimension is a multiple of 16, as MOSS-Image requires. S, M and L follow
the official 1, 1.5 and 2 MP recommendation. XL goes beyond it, inside the 4 MP
range the user guide describes as extrapolation. The 16:9 and 9:16 buckets
approximate their ratio; all others are exact. These buckets have not been
validated through generation tests.

## Wiring

```text
base model -> LoRA(s) ------------------------> CyberMOSS Sampler (model)
prompt text encode ---------------------------> CyberMOSS Sampler (positive)
empty-text encode (full presets) -------------> CyberMOSS Sampler (negative)
CyberMOSS Empty Latent -----------------------> CyberMOSS Sampler (latent_image)
```

## Installation

Copy the complete `ComfyUI-CyberMOSS-Sampler` folder to:

```text
ComfyUI/custom_nodes/ComfyUI-CyberMOSS-Sampler
```

Restart ComfyUI. No extra Python packages are required beyond ComfyUI's own
PyTorch/Comfy modules. A ComfyUI version with MOSS-Image support is required.
The model check recognizes both ComfyUI's native MOSSImage class and the
independent MOSSImage class supplied by the official OpenMOSS custom loaders,
including their subclasses.

After updating, also refresh the browser so it loads the matching frontend
extensions and node metadata. See [CHANGELOG.md](CHANGELOG.md) for changes.

When upgrading from 0.1.0, `reference` now selects the official MOSS grid at
full denoise. To retain the old continuous reference grid, choose `simple`.
For saved Turbo workflows, select `turbo_reference` again to load its updated
`simple` setting; loading a workflow preserves its saved manual controls.

## Development checks

With Python 3.11+, PyTorch, NumPy and Node.js 22+ installed:

```sh
python -B -m unittest discover -s tests -v
```

Tests cover schedule budgets, tensor-level sampling, seed reproducibility,
guidance and layer skipping, previews, validation and frontend widget behavior.
ComfyUI boundaries use test doubles; image quality and third-party patch
compatibility still need generation tests with a real MOSS-Image checkpoint.

## License

Released under the MIT license. The original Kostiantyn Hrytsuk copyright
notice is retained alongside Cyberdelia's notice. See [LICENSE](LICENSE).
