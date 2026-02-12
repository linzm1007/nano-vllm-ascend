import unittest
from unittest.mock import MagicMock
from collections import deque
import torch

from nanovllm.engine.block_manager import BlockManager


class TestMultiCardBlockManager(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.num_blocks = 100
        # 模拟两个 Rank 的 Manager
        self.manager_rank0 = BlockManager(self.num_blocks, self.block_size)
        self.manager_rank1 = BlockManager(self.num_blocks, self.block_size)

    def create_real_mock_seq(self, tokens):
        seq = MagicMock()
        seq.token_ids = tokens
        seq.num_tokens = len(tokens)
        seq.__len__ = lambda *args: seq.num_tokens
        seq.block = lambda i: seq.token_ids[i * self.block_size: (i + 1) * self.block_size]
        seq.num_blocks = (len(tokens) + self.block_size - 1) // self.block_size
        seq.block_table = []
        seq.num_cached_tokens = 0
        return seq

    def test_rank_consistency_on_prefill(self):
        """测试不同 Rank 对同一个 Prompt 的分配是否完全一致"""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2个满 block
        seq_rank0 = self.create_real_mock_seq(tokens)
        seq_rank1 = self.create_real_mock_seq(tokens)

        # 两个 Rank 分别执行 allocate
        self.manager_rank0.allocate(seq_rank0)
        self.manager_rank1.allocate(seq_rank1)

        # 检查 Block ID 是否一致
        self.assertEqual(seq_rank0.block_table, seq_rank1.block_table)
        # 检查 Hash 映射是否一致
        self.assertEqual(self.manager_rank0.hash_to_block_id, self.manager_rank1.hash_to_block_id)

    def test_rank_consistency_on_decode_append(self):
        """测试 Decode 阶段步进时，各 Rank 分配新 Block 的时机和 ID 是否一致"""
        tokens = [1, 2, 3, 4]  # 刚好满一个 block
        seq0 = self.create_real_mock_seq(tokens)
        seq1 = self.create_real_mock_seq(tokens)

        self.manager_rank0.allocate(seq0)
        self.manager_rank1.allocate(seq1)

        # 模拟生成了一个新 token (len 4 -> 5)，触发新 block 分配
        for s in [seq0, seq1]:
            s.token_ids.append(99)
            s.num_tokens = 5
            # 注意：may_append 内部用 len(seq)+1 判断，如果已经 append 了，要保证逻辑闭环
            # 这里的 seq0 已经是 5 了，如果 may_append 逻辑是 len+1 = 6，则不触发
            # 按照你给的代码逻辑，我们手动控制调用时机：

        # 恢复到即将生成第5个的状态
        seq0.num_tokens = 4
        seq1.num_tokens = 4

        self.manager_rank0.may_append(seq0)
        self.manager_rank1.may_append(seq1)

        self.assertEqual(len(seq0.block_table), 2)
        self.assertEqual(seq0.block_table, seq1.block_table)
        self.assertEqual(seq0.block_table[-1], 1)  # 应该是空闲队列的下一个

    def test_out_of_memory_protection(self):
        """测试当物理 Block 耗尽时，系统是否能正确处理而非返回非法 ID"""
        small_manager = BlockManager(num_blocks=1, block_size=4)
        seq = self.create_real_mock_seq([1, 2, 3, 4, 5])  # 需要2个 block

        # 此时 can_allocate 应该返回 False
        self.assertFalse(small_manager.can_allocate(seq))

        # 如果强制 allocate，应该抛出 IndexError 或我们自定义的错误，而不是分配出越界的 ID
        with self.assertRaises(IndexError):
            small_manager.allocate(seq)
