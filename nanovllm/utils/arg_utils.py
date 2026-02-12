# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class EngineArgs:
    """Arguments for nano-vLLM-Ascend engine."""
    model: str
    dtype: str = "auto"
    seed: int = 0
    tensor_parallel_size: int = 1
    block_size: int = 256
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 16
    enforce_eager: bool = True
    hccl_port: int = 3456

    def __post_init__(self):
        self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Shared CLI arguments for nano-vLLM-Ascend engine."""
        # Model arguments
        parser.add_argument('--model', type=str, default='facebook/opt-125m',
                            help='name or path of the huggingface model to use')
        # TODO(woosuk): Support FP32.
        parser.add_argument('--dtype', type=str, default=EngineArgs.dtype,
                            choices=['auto', 'half', 'bfloat16', 'float'],
                            help='data type for model weights and activations. '
                                 'The "auto" option will use FP16 precision '
                                 'for FP32 and FP16 models, and BF16 precision '
                                 'for BF16 models.')
        parser.add_argument('--tensor-parallel-size', '-tp', type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # KV cache arguments
        parser.add_argument('--block-size', type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed', type=int, default=EngineArgs.seed,
                            help='random seed')
        parser.add_argument('--gpu-memory-utilization', type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                                 'the model executor')
        parser.add_argument('--max-num-batched-tokens', type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                                 'iteration')
        parser.add_argument('--max-num-seqs', type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats', action='store_true',
                            help='disable logging statistics')
        parser.add_argument("--enforce-eager", action="store_true",
                            help="Enforce eager mode (default is False/Graph mode)")
        parser.add_argument('--hccl-port', type=int, default=EngineArgs.hccl_port,
                            help='hccl port')

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous nano-vLLM-Ascend engine."""
    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--disable-log-requests', action='store_true',
                            help='disable logging requests')
        return parser
