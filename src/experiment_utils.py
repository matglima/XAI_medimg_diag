import json
import os
import platform
import random
from argparse import Namespace

import numpy as np
import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - lightning should normally exist here
    pl = None


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    if pl is not None:
        pl.seed_everything(seed, workers=True)


def _to_serializable(value):
    if isinstance(value, Namespace):
        return {k: _to_serializable(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(output_path, payload):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(payload), handle, indent=2, sort_keys=True)


def collect_system_info():
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }

    try:
        cuda_available = torch.cuda.is_available()
        info["cuda_available"] = cuda_available
        if cuda_available:
            info["cuda_device_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["cuda_probe_error"] = str(exc)

    return info


def save_run_manifest(output_dir, filename, args, extra=None):
    payload = {
        "args": _to_serializable(args),
        "system": collect_system_info(),
    }
    if extra:
        payload["extra"] = _to_serializable(extra)

    output_path = os.path.join(output_dir, filename)
    write_json(output_path, payload)
    return output_path
