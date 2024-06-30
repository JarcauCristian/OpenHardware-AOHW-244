import os
import platform
import socket

import torch


def is_windows_os() -> bool:
    return platform.system() == 'Windows'


def is_linux_os() -> bool:
    return platform.system() == 'Linux'


def get_machine_name() -> str:
    if is_linux_os():
        return socket.gethostname()
    else:
        return os.environ['COMPUTERNAME']


def can_compile_torch_model(use_cuda: bool = True, compile: bool = True) -> bool:
    if not use_cuda or not compile or is_windows_os():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 7
