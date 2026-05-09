
import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np
import torch


def enable_full_determinism(seed: int, warn_only: bool = False):
    """Linguistic collapse paper.

    Reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    - https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism for tensorflow
    """

    # Enable PyTorch deterministic mode. This potentially requires either the environment
    # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # The environment variable required to enable deterministic mode on Ascend NPUs.
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    os.environ["HCCL_DETERMINISTIC"] = "1"

    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_seed(seed: int, deterministic: bool = False):
    """Provide reproducible behavior.
    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if deterministic:
        torch.use_deterministic_algorithms(True)

    # if is_torch_mlu_available():
    #     torch.mlu.manual_seed_all(seed)
    # if is_torch_musa_available():
    #     torch.musa.manual_seed_all(seed)
    # if is_torch_npu_available():
    #     torch.npu.manual_seed_all(seed)
    # if is_torch_hpu_available():
    #     torch.hpu.manual_seed_all(seed)
    # if is_torch_xpu_available():
    #     torch.xpu.manual_seed_all(seed)
    # if is_tf_available():
    #     import tensorflow as tf

        # tf.random.set_seed(seed)
        # if deterministic:
        #     tf.config.experimental.enable_op_determinism()
