import os
from nanovllm import LLM, SamplingParams
from nanovllm.utils.logger import logger
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/xx/xx/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)

    prompts = [
        "你是谁",
        "讲一个笑话，200字",
        "写一个作文，题材任意，300字",
        "中国是",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        logger.info("\n")
        logger.info(f"Prompt: {prompt!r}")
        logger.info(f"Completion: len: {len(output['token_ids'])} text:{output['text']!r}")


if __name__ == "__main__":
    main()
