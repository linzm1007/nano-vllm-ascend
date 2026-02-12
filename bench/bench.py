import argparse
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.arg_utils import EngineArgs
from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)





def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    parser = argparse.ArgumentParser(
        description="Nano vLLM Ascend OpenAI-Compatible RESTful API server."
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args = EngineArgs.from_cli_args(args)
    logger.info(f"args: {args}")
    llm = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=4096, max_num_seqs=args.max_num_seqs,
              tensor_parallel_size=args.tensor_parallel_size, hccl_port=args.hccl_port)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _
                       in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    logger.info(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
