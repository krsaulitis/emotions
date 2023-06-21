import os
import torch
import re
from pathlib import Path
from typing import Tuple, Union
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler


def save_state(
        obj: Union[torch.nn.Module, LRScheduler],
        target_dir: str,
        state_name: str, ):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    state_name = f"{state_name}.pt"
    state_path = target_dir_path / state_name

    print(f"ℹ️ Saving state to: {state_path}")
    torch.save(
        obj=obj.state_dict(),
        f=state_path,
    )


def load_state(
        obj: Union[torch.nn.Module, LRScheduler],
        target_dir: str,
        state_name: str,
        extension: str = 'pt',
) -> Union[torch.nn.Module, LRScheduler, None]:
    target_dir_path = Path(target_dir)

    state_name = f"{state_name}.{extension}"
    state_path = target_dir_path / state_name

    if not os.path.isfile(state_path):
        return None

    print(f"ℹ️ Loading state from: {state_path}")
    obj.load_state_dict(torch.load(state_path))
    return obj


def load_model_w_epoch(
        model: torch.nn.Module,
        target_dir: str,
) -> Tuple[Union[Module, LRScheduler, None], int]:
    epoch = 0
    if not os.path.isdir(target_dir):
        return None, epoch

    for filename in os.listdir(target_dir):
        if filename.startswith('model-e'):
            epoch = get_model_epoch(filename)

    model = load_state(model, target_dir, f"model-e{epoch}")
    return (model if model and epoch else None), epoch


def load_scheduler(scheduler: LRScheduler, target_dir: str) -> Union[LRScheduler, None]:
    if not os.path.isdir(target_dir):
        return None
    return load_state(scheduler, target_dir, "scheduler")


def check_file_in_dir(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return True

    return False


def get_model_epoch(filename):
    x = re.search(r'_e(\d+)', filename)
    return x.group(1) if x else None


def get_device(force_cpu):
    device = "cpu"
    if not force_cpu and torch.cuda.is_available():
        device = "cuda"
    elif not force_cpu and torch.backends.mps.is_available():
        device = "mps"
    return device
