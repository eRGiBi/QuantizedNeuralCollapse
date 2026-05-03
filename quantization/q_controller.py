import torch

PRECISION_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

def apply_quantization_config(model, config):
    precision = config.get("precision")

    # Standard PyTorch Casting
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    if precision in dtype_map:
        return model.to(device=config["device"], dtype=dtype_map[precision])

    return model.to(config["device"])
