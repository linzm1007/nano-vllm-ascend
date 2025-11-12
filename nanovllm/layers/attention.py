import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context
from nanovllm.utils.logger import logger

def store_kvcache(
        key: torch.Tensor,  # [N, G, D]
        value: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, block_size, G, D]
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # [N]
        kvcache_block_size: int = 256,
):
    N, G, D = key.shape
    num_blocks, block_size, cache_G, cache_D = k_cache.shape
    assert G == cache_G and D == cache_D, f"KV维度不匹配：输入(G={G},D={D})，缓存(G={cache_G},D={cache_D})"
    assert slot_mapping.shape == (N,), f"slot_mapping维度错误：期望(N,)，实际{slot_mapping.shape}"

    # 1. 筛选有效slot（排除slot=-1的情况）
    valid_mask = (slot_mapping != -1)  # [N,] 布尔张量

    # 2. 向量化计算block索引
    all_block_idx = slot_mapping // kvcache_block_size  # [N,]
    all_block_inner_idx = slot_mapping % kvcache_block_size  # [N,]

    # 3. 筛选在缓存范围内的元素
    in_range_mask = (all_block_idx < num_blocks) & (all_block_inner_idx < block_size)  # [N,]
    final_mask = valid_mask & in_range_mask  # [N,] 最终有效掩码

    # 4. 生成静态范围索引，再用掩码筛选（替代torch.nonzero，避免动态形状）
    # 生成0~N-1的索引张量（静态形状[N,]）
    indices = torch.arange(N, device=final_mask.device, dtype=torch.long)
    # 用final_mask筛选有效索引（结果可能为空，但形状由掩码动态决定，但生成方式是静态的）
    valid_i = indices[final_mask]  # [M,]，M为有效元素数量（可能为0）

    # 5. 提取有效block索引和数据
    valid_block_idx = all_block_idx[valid_i]  # [M,]
    valid_block_inner_idx = all_block_inner_idx[valid_i]  # [M,]
    valid_key = key[valid_i]  # [M, G, D]
    valid_value = value[valid_i]  # [M, G, D]

    # 6. 批量更新缓存
    k_cache[valid_block_idx, valid_block_inner_idx] = valid_key.to(k_cache.dtype, non_blocking=True)
    v_cache[valid_block_idx, valid_block_inner_idx] = valid_value.to(v_cache.dtype, non_blocking=True)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    seq_len, G, D = hidden_states.shape
    assert G * n_rep <= 32, f"Q头数过多（{G * n_rep}），可能导致溢出"  # 防护性校验
    return hidden_states.unsqueeze(2).expand(seq_len, G, n_rep, D).reshape(seq_len, G * n_rep, D)


class Attention(nn.Module):
    def __init__(
            self,
            num_heads: int,  # 从模型权重反推的真实Q头数
            head_dim: int,  # 从模型权重反推的真实头维度
            scale,
            num_kv_heads: int,  # 从模型权重反推的真实KV头数
            kvcache_block_size: int = 256,
    ):
        super().__init__()
        # 关键：用模型权重的真实维度，而非配置文件标注
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kvcache_block_size = kvcache_block_size

        # 强制维度对齐（基于真实权重）
        assert self.num_heads % self.num_kv_heads == 0, f"Q头数{num_heads}必须是KV头数{num_kv_heads}的整数倍"
        self.num_rep = self.num_heads // self.num_kv_heads
        # 仅内部计算scale，删除外部传入（避免重复缩放）
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.bfloat16)

        # 初始化KV缓存（4维）
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def _init_cache(self, num_blocks: int, device: torch.device, dtype: torch.dtype):
        self.k_cache = torch.zeros(
            (num_blocks, self.kvcache_block_size, self.num_kv_heads, self.head_dim),
            device=device,
            dtype=dtype,
            requires_grad=False
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    def _check_numeric_range(self, x: torch.Tensor, name: str):
        """检查数值范围，避免溢出（bfloat16敏感）"""
        max_val = x.max().item()
        min_val = x.min().item()
        if max_val > 1e4 or min_val < -1e4:
            logger.warning(f"警告：{name}数值溢出（max={max_val:.2f}, min={min_val:.2f}）")
        return x

    def _prefill_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q, cu_seqlens_k,
                      device: torch.device):
        batch_size = len(cu_seqlens_q) - 1
        outputs = []

        for i in range(batch_size):
            start_q, end_q = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            start_k, end_k = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
            q_seq = q[start_q:end_q]  # [Lq, H, D]
            k_seq = k[start_k:end_k]  # [Lk, G, D]
            v_seq = v[start_k:end_k]  # [Lk, G, D]

            # 数值范围检查（避免QKV本身乱码）
            self._check_numeric_range(q_seq, f"prefill_q_seq_{i}")
            self._check_numeric_range(k_seq, f"prefill_k_seq_{i}")

            # GQA扩展
            k_seq = repeat_kv(k_seq, self.num_rep)  # [Lk, H, D]
            v_seq = repeat_kv(v_seq, self.num_rep)  # [Lk, H, D]

            # 注意力计算（避免重复转置，提升效率）
            q_trans = q_seq.transpose(0, 1)  # [H, Lq, D]
            k_trans = k_seq.transpose(0, 1).transpose(-2, -1)  # [H, D, Lk]

            attn = torch.matmul(q_trans, k_trans) * self.scale  # [H, Lq, Lk]
            self._check_numeric_range(attn, f"prefill_attn_{i}")  # 检查注意力分数

            # 因果掩码（适配Lq≠Lk场景）
            Lq, Lk = q_seq.shape[0], k_seq.shape[0]
            mask = torch.triu(torch.full((Lq, Lk), float('-inf'), device=device, dtype=attn.dtype), diagonal=1)
            attn = attn + mask.unsqueeze(0)

            # Softmax（限制数值范围，避免溢出）
            attn = F.softmax(attn.clamp(-100, 100), dim=-1)  # 裁剪极端值
            out_seq = torch.matmul(attn, v_seq.transpose(0, 1))  # [H, Lq, D]
            out_seq = out_seq.transpose(0, 1)  # [Lq, H, D]

            outputs.append(out_seq)

        return torch.cat(outputs, dim=0)

    def _decode_attn(self, q: torch.Tensor, context_lens, block_tables, device: torch.device):
        batch_size = q.shape[0]
        q = q.reshape(batch_size, self.num_heads, self.head_dim)  # [B, H, D]
        self._check_numeric_range(q, "decode_q")
        outputs = []

        for i in range(batch_size):
            # 验证缓存索引（关键：避免读取其他序列的KV）
            block_ids = block_tables[i][block_tables[i] != -1]
            history_len = context_lens[i].item()
            if len(block_ids) == 0 and history_len > 0:
                logger.warning(f"警告：序列{i}无缓存块，但历史长度={history_len}")
                history_len = 0

            # 读取历史KV
            k_history_list = []
            v_history_list = []
            for block_idx in block_ids:
                if block_idx >= self.k_cache.shape[0]:
                    logger.warning(f"警告：序列{i}的块索引{block_idx}超出缓存范围")
                    continue
                k_block = self.k_cache[block_idx]  # [block_size, G, D]
                v_block = self.v_cache[block_idx]
                k_history_list.append(k_block)
                v_history_list.append(v_block)

            if not k_history_list:
                k_history = torch.zeros((0, self.num_kv_heads, self.head_dim), device=device, dtype=q.dtype)
                v_history = torch.zeros_like(k_history)
            else:
                k_history = torch.cat(k_history_list, dim=0)[:history_len]  # [Lk, G, D]
                v_history = torch.cat(v_history_list, dim=0)[:history_len]
            self._check_numeric_range(k_history, f"decode_k_history_{i}")

            # GQA扩展
            k_history = repeat_kv(k_history, self.num_rep)  # [Lk, H, D]
            v_history = repeat_kv(v_history, self.num_rep)  # [Lk, H, D]

            # 注意力计算
            q_i = q[i].unsqueeze(0).unsqueeze(-2)  # [1, H, 1, D]
            k_trans = k_history.transpose(0, 1).transpose(-2, -1)  # [H, D, Lk]
            attn = torch.matmul(q_i, k_trans) * self.scale  # [1, H, 1, Lk]
            attn = F.softmax(attn.clamp(-100, 100), dim=-1)

            # 加权求和
            out_i = torch.matmul(attn, v_history.transpose(0, 1))  # [1, H, 1, D]
            out_i = out_i.squeeze(-2)  # [1, H, D]
            outputs.append(out_i)

        return torch.cat(outputs, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        device = q.device
        dtype = q.dtype

        # 初始化缓存
        if self.k_cache.numel() == 0 and hasattr(context, "num_kvcache_blocks"):
            self._init_cache(context.num_kvcache_blocks, device, dtype)

        # 写入缓存（增加索引校验）
        if self.k_cache.numel() > 0 and self.v_cache.numel() > 0:
            assert context.slot_mapping.numel() == q.shape[
                0], f"slot_mapping长度（{context.slot_mapping.numel()}）与Q长度（{q.shape[0]}）不匹配"
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping, self.kvcache_block_size)

        # 核心维度校验（确保QKV与权重匹配）
        assert q.shape == (k.shape[0], self.num_heads,
                           self.head_dim), f"Q形状错误：期望({k.shape[0]}, {self.num_heads}, {self.head_dim})，实际{q.shape}"
        assert k.shape == (k.shape[0], self.num_kv_heads,
                           self.head_dim), f"K形状错误：期望({k.shape[0]}, {self.num_kv_heads}, {self.head_dim})，实际{k.shape}"
        assert v.shape == k.shape, f"V形状错误：期望{k.shape}，实际{v.shape}"

        # 分阶段计算
        if context.is_prefill:
            assert context.cu_seqlens_q is not None and context.cu_seqlens_k is not None
            o = self._prefill_attn(q, k, v, context.cu_seqlens_q, context.cu_seqlens_k, device)
        else:
            assert context.context_lens is not None and context.block_tables is not None
            o = self._decode_attn(q, context.context_lens, context.block_tables, device)

        o = o.reshape(o.shape[0], -1)  # [seq_len/B, H, D] → [seq_len/B, H×D]

        expected_shape = (q.shape[0], self.num_heads * self.head_dim)
        assert o.shape == expected_shape, f"输出形状错误：期望{expected_shape}，实际{o.shape}"
        self._check_numeric_range(o, "attention_output")

        return o
