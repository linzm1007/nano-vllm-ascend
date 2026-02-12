# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Nano-vLLM project

"""Minimal multimodal inference example using Qwen3-VL."""

import os
from PIL import Image
from transformers import AutoProcessor
from nanovllm import LLM, SamplingParams

DEFAULT_IMAGE_URL = "/data_mount/linzm/nano-vllm-ascend/example/asset/fanren.jpg"
DEFAULT_PROMPT = "描述图片内容"


def load_local_image(path: str) -> Image.Image:
    """读取本地图片"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"本地图片不存在，请检查路径：{path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    model_path = os.path.expanduser("/data_mount/models/Qwen3-VL-2B-Instruct")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        hccl_port=28801,
        gpu_memory_utilization=0.4,
        kvcache_block_size=128
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    image: Image.Image = load_local_image(DEFAULT_IMAGE_URL)
    chat_prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    request = {"text": chat_prompt, "images": [image]}
    print(f"request: {request}")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    outputs = llm.generate_multimodal(
        [request],
        sampling_params,
        processor,
        use_tqdm=False,
    )

    print("Prompt:", DEFAULT_PROMPT)
    print("Image Path:", DEFAULT_IMAGE_URL)
    print("Completion:", outputs[0]["text"])


if __name__ == "__main__":
    main()
