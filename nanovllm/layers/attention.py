import torch
import torchair as tng
import torch_npu
from torch import nn
from nanovllm.utils.context import get_context


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scaling, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = 0
        self.scale = scaling if scaling is not None else (1.0 / (head_dim ** 0.5))
        self.register_buffer("mask", ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device="npu")
        ), persistent=False)
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def _store_kvcache(self, k, v, context):
        block_size = self.block_size
        if context.is_prefill:
            torch_npu._npu_reshape_and_cache(
                k, v,
                self.k_cache.view(self.k_cache.shape[0], block_size, self.num_kv_heads, self.head_dim),
                self.v_cache.view(self.v_cache.shape[0], block_size, self.num_kv_heads, self.head_dim),
                context.slot_mapping.int()
            )
        else:
            cast_key = k.view(k.shape[0], 1, -1).contiguous()
            cast_value = v.view(v.shape[0], 1, -1).contiguous()
            torch_npu.scatter_update_(self.k_cache, context.slot_mapping, cast_key, -2)
            torch_npu.scatter_update_(self.v_cache, context.slot_mapping, cast_value, -2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        self.block_size = context.block_size
        batch_size = q.shape[0]
        if k.ndim == 2:
            k = k.view(-1, int(self.num_kv_heads), int(self.head_dim))
            v = v.view(-1, int(self.num_kv_heads), int(self.head_dim))
            q = q.view(-1, int(self.num_heads), int(self.head_dim))
        self._store_kvcache(k, v, context)
        if context.is_prefill:
            actual_qlen = context.cu_seqlens_q[1:].to(torch.int32).tolist()
            actual_kvlen = context.cu_seqlens_k[1:].to(torch.int32).tolist()
            return torch_npu.npu_fused_infer_attention_score_v2(
                q, k, v,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                atten_mask=self.mask,
                actual_seq_qlen=actual_qlen,
                actual_seq_kvlen=actual_kvlen,
                sparse_mode=3, inner_precise=1
            )[0].reshape(batch_size, -1)
        else:
            if context.is_enforce_eager:
                # 单算子
                q_input = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
                attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                    q_input, self.k_cache, self.v_cache,
                    num_query_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BNSD",
                    softmax_scale=self.scale,
                    block_table=context.block_tables,
                    block_size=self.block_size,
                    actual_seq_kvlen=context.context_lens,
                    sparse_mode=0,
                    inner_precise=1,
                )[0]
                return attn_output.transpose(1, 2).reshape(batch_size, -1)
            else:
                # 图编译
                q_input = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
                attn_output, _ = tng.ops.npu_fused_infer_attention_score(
                    q_input,
                    self.k_cache,
                    self.v_cache,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BNSD",
                    scale=self.scale,
                    actual_seq_lengths_kv=context.context_lens.to(torch.int64),
                    block_table=context.block_tables,
                    block_size=self.block_size,
                    inner_precise=1
                )
                return attn_output.transpose(1, 2).reshape(batch_size, -1)
