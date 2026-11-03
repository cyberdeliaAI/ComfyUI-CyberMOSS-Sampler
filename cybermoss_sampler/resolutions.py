"""MOSS-Image resolution selector and 128-channel empty latent node."""

# The Flux2 VAE packs 2x2 patches of its /8 latent into 128 channels: /16 overall.
# MOSS-Image recommends 1, 1.5 or 2 MP with dimensions in multiples of 16.
RESOLUTION_BUCKETS = {
    "S (~1.0 MP)": {
        "1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152),
        "3:2": (1248, 832), "2:3": (832, 1248),
        "16:9": (1360, 768), "9:16": (768, 1360),
        "21:9": (1568, 672), "9:21": (672, 1568),
        "5:4": (1120, 896), "4:5": (896, 1120),
    },
    "M (~1.5 MP)": {
        "1:1": (1248, 1248), "4:3": (1472, 1104), "3:4": (1104, 1472),
        "3:2": (1536, 1024), "2:3": (1024, 1536),
        "16:9": (1680, 944), "9:16": (944, 1680),
        "21:9": (1904, 816), "9:21": (816, 1904),
        "5:4": (1440, 1152), "4:5": (1152, 1440),
    },
    "L (~2.0 MP)": {
        "1:1": (1456, 1456), "4:3": (1664, 1248), "3:4": (1248, 1664),
        "3:2": (1776, 1184), "2:3": (1184, 1776),
        "16:9": (1936, 1088), "9:16": (1088, 1936),
        "21:9": (2240, 960), "9:21": (960, 2240),
        "5:4": (1600, 1280), "4:5": (1280, 1600),
    },
    # Beyond the official 1-2 MP recommendation, inside its 4 MP extrapolation range.
    "XL (~3.0 MP)": {
        "1:1": (1776, 1776), "4:3": (2048, 1536), "3:4": (1536, 2048),
        "3:2": (2160, 1440), "2:3": (1440, 2160),
        "16:9": (2304, 1296), "9:16": (1296, 2304),
        "21:9": (2688, 1152), "9:21": (1152, 2688),
        "5:4": (2000, 1600), "4:5": (1600, 2000),
    },
}
DEFAULT_RESOLUTION_SIZE = "M (~1.5 MP)"
DEFAULT_RESOLUTION_ASPECT = "2:3"


CATEGORY = "CyberMOSS"


def _build_resolution_options():
    options = {}
    dimensions = {}
    for size, aspects in RESOLUTION_BUCKETS.items():
        size_options = []
        for aspect, (width, height) in aspects.items():
            label = f"{width}x{height} ({aspect})"
            size_options.append(label)
            dimensions[(size, label)] = (width, height)
        options[size] = size_options
    return options, dimensions


RESOLUTION_OPTIONS, _DIMENSIONS = _build_resolution_options()
ALL_RESOLUTIONS = list(dict.fromkeys(
    label for labels in RESOLUTION_OPTIONS.values() for label in labels
))

_default_width, _default_height = RESOLUTION_BUCKETS[DEFAULT_RESOLUTION_SIZE][
    DEFAULT_RESOLUTION_ASPECT
]
DEFAULT_RESOLUTION = (
    f"{_default_width}x{_default_height} "
    f"({DEFAULT_RESOLUTION_ASPECT})"
)


def resolve_dimensions(size, resolution):
    """Return dimensions, preserving the aspect after a stale UI tier change."""
    try:
        return _DIMENSIONS[(size, resolution)]
    except KeyError:
        pass

    # ComfyUI can briefly restore the size widget before it refreshes the
    # dependent resolution widget. If the stale value is one of our valid
    # resolutions, select the same aspect in the active tier instead.
    if resolution in ALL_RESOLUTIONS and size in RESOLUTION_OPTIONS:
        aspect = resolution.rpartition("(")[2].removesuffix(")")
        suffix = f"({aspect})"
        for candidate in RESOLUTION_OPTIONS[size]:
            if candidate.endswith(suffix):
                return _DIMENSIONS[(size, candidate)]

    raise ValueError(
        f"Resolution {resolution!r} does not belong to size {size!r}"
    )


class CyberMOSSEmptyLatent:
    """Create a MOSS-Image / Flux2-compatible 128-channel empty latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size": (list(RESOLUTION_OPTIONS.keys()), {
                    "default": DEFAULT_RESOLUTION_SIZE,
                    "tooltip": "MOSS-Image resolution tier; filters the resolution list. "
                               "1-2 MP is the official range; XL extrapolates.",
                    "cybermoss_resolutions": RESOLUTION_OPTIONS,
                }),
                "resolution": (ALL_RESOLUTIONS, {
                    "default": DEFAULT_RESOLUTION,
                    "tooltip": "Concrete MOSS-Image width, height and aspect ratio.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, size, resolution, batch_size):
        import torch

        width, height = resolve_dimensions(size, resolution)
        latent = torch.zeros([int(batch_size), 128, height // 16, width // 16])
        return ({"samples": latent}, width, height)


NODE_CLASS_MAPPINGS = {
    "CyberMOSSEmptyLatent": CyberMOSSEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberMOSSEmptyLatent": "CyberMOSS Empty Latent",
}
