from .llama import LlamaForCausalLM
from .mini_cpm4 import MiniCPMForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration

model_dict = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen3ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
    "MiniCPMForCausalLM": MiniCPMForCausalLM,
}
