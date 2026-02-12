# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import logging
import sys
import os

_FORMAT = "%(levelname)s %(asctime)s,%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

# 避免多进程重复初始化的标记
_LOGGER_INITIALIZED = False
_root_logger = logging.getLogger("nano-vllm-ascend")
_default_handler = None


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])
        return msg


def _setup_logger(force: bool = False):
    """
    初始化日志，适配多进程场景
    :param force: 强制重新初始化（子进程调用）
    """
    global _LOGGER_INITIALIZED, _default_handler

    # 主进程已初始化且非强制，直接返回
    if _LOGGER_INITIALIZED and not force:
        return

    # 清空原有handler（子进程可能继承了主进程的handler，需要重置）
    if _default_handler is not None:
        _root_logger.removeHandler(_default_handler)
        _default_handler = None

    _root_logger.setLevel(logging.DEBUG)
    _default_handler = logging.StreamHandler(sys.stdout)
    # 关键：强制立即刷新，避免缓冲
    _default_handler.flush = lambda: sys.stdout.flush()
    _default_handler.setLevel(logging.INFO)

    # 设置formatter
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    _root_logger.addHandler(_default_handler)
    # 关闭传播，避免重复输出
    _root_logger.propagate = False

    # 标记初始化完成
    _LOGGER_INITIALIZED = True


def init_logger(name: str) -> logging.Logger:
    """
    创建/获取logger实例，确保多进程下正确初始化
    """
    _setup_logger(force=os.getpid() != 0)  # 子进程PID非0，强制重新初始化
    logger = logging.getLogger(f"nano-vllm-ascend.{name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    return logger


_setup_logger()
