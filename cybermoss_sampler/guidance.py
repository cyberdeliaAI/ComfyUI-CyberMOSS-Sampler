"""Constant CFG with optional layer skipping in the negative (uncond) pass.

MOSS-Image's reference pipeline skips layer 22 in the negative branch
(uncond_skip_layers=[22]). This guider reproduces that through ComfyUI's
standard "dit" / "single_block" patch-replace hook, which MOSSImage honors.
"""

import logging
import math
import re

import comfy.model_patcher
import comfy.samplers


def parse_skip_layers(value) -> tuple:
    """'22' or '22, 24, 26' -> (22,) / (22, 24, 26). Empty disables skipping."""
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        layers = [int(v) for v in value]
    else:
        text = str(value).strip()
        if text and not re.fullmatch(r"[\d\s,;]*", text):
            raise ValueError("skip_layers must be layer numbers separated by commas, e.g. '22' or '22, 24'")
        layers = [int(v) for v in re.findall(r"\d+", text)]
    if any(layer < 0 for layer in layers):
        raise ValueError("skip_layers must be zero or positive layer numbers")
    return tuple(dict.fromkeys(layers))


def _skip_block(args, extra_args):
    return args


def with_uncond_layer_skip(model_options, layers):
    """Return model options whose uncond pass skips the given MOSS layers.

    The cond pass keeps the original options, including every LoRA and model
    patch. Only the uncond pass gets the block replacement.
    """
    def calc_cond_batch(args):
        cond, uncond = args["conds"]
        x, sigma, model = args["input"], args["sigma"], args["model"]
        options = args["model_options"]
        cond_out, _ = comfy.samplers.calc_cond_batch(model, [cond, None], x, sigma, options)
        if uncond is None:
            return [cond_out, None]
        skip_options = {**options, "transformer_options": options.get("transformer_options", {})}
        for layer in layers:
            skip_options = comfy.model_patcher.set_model_options_patch_replace(
                skip_options, _skip_block, "dit", "single_block", layer)
        _, uncond_out = comfy.samplers.calc_cond_batch(model, [None, uncond], x, sigma, skip_options)
        return [cond_out, uncond_out]

    options = dict(model_options)
    options["sampler_calc_cond_batch_function"] = calc_cond_batch
    return options


class CyberMOSSGuider(comfy.samplers.CFGGuider):
    """Constant CFG, preserving model options and patches."""

    def __init__(self, model_patcher, cfg: float = 1.0, skip_layers=()):
        super().__init__(model_patcher)
        self.set_cfg(float(cfg))
        self.skip_layers = tuple(skip_layers)
        self._warned = False

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        negative = self.conds.get("negative", None)
        if self.skip_layers and negative is not None and not math.isclose(self.cfg, 1.0):
            if "sampler_calc_cond_batch_function" in model_options:
                if not self._warned:
                    logging.warning("CyberMOSS: the model already has a cond-batch function "
                                    "(e.g. a Skip Layer Guidance node); skip_layers is not applied.")
                    self._warned = True
            else:
                model_options = with_uncond_layer_skip(model_options, self.skip_layers)
        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            negative, self.conds.get("positive", None),
            self.cfg, model_options=model_options, seed=seed,
        )
