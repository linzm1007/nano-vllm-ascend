import unittest
from unittest.mock import MagicMock
from collections import deque

from nanovllm.engine.block_manager import BlockManager


class TestBlockManager(unittest.TestCase):
    def setUp(self):
        self.block_size = 4
        self.num_blocks = 10
        self.manager = BlockManager(self.num_blocks, self.block_size)

    def create_mock_seq(self, tokens, block_table=None):
        seq = MagicMock()
        seq.token_ids = tokens
        seq.num_tokens = len(tokens)
        seq.__len__.return_value = len(tokens)
        # 模拟 block(i) 方法
        seq.block = lambda i: tokens[i * self.block_size: (i + 1) * self.block_size]
        seq.num_blocks = (len(tokens) + self.block_size - 1) // self.block_size
        seq.block_table = block_table if block_table is not None else []
        seq.num_cached_tokens = 0
        return seq

    def test_basic_allocate_deallocate(self):
        """测试基础分配与回收"""
        seq = self.create_mock_seq([1, 2, 3, 4, 5])  # 需要2个block
        self.assertTrue(self.manager.can_allocate(seq))

        self.manager.allocate(seq)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(self.manager.free_block_ids), 8)
        self.assertEqual(len(self.manager.used_block_ids), 2)

        self.manager.deallocate(seq)
        self.assertEqual(len(seq.block_table), 0)
        self.assertEqual(len(self.manager.free_block_ids), 10)
        self.assertEqual(self.manager.blocks[0].ref_count, 0)

    def test_prefix_caching_hit(self):
        """测试 Prefix Caching 命中逻辑"""
        # 1. 第一个序列分配，填满第一个 block
        tokens1 = [10, 20, 30, 40, 50]
        seq1 = self.create_mock_seq(tokens1)
        self.manager.allocate(seq1)

        # 2. 第二个序列有相同的 prefix (前4个token)
        tokens2 = [10, 20, 30, 40, 99]
        seq2 = self.create_mock_seq(tokens2)

        self.manager.allocate(seq2)

        # seq1 和 seq2 的第一个 block 应该是一样的 ID (因为 hash 命中)
        self.assertEqual(seq1.block_table[0], seq2.block_table[0])
        self.assertEqual(self.manager.blocks[seq1.block_table[0]].ref_count, 2)
        # seq2 的第二个 block 应该是新的
        self.assertNotEqual(seq1.block_table[1], seq2.block_table[1])
        # 验证 cached tokens 数量 (第一个 block 命中，4个 token)
        self.assertEqual(seq2.num_cached_tokens, 4)

    def test_can_append_logic(self):
        """测试 Decode 阶段是否需要申请新 Block 的逻辑"""
        # 情况1：当前 block 未满 (例如 block_size=4, 现有3个 token)
        seq = self.create_mock_seq([1, 2, 3])
        # 模拟已经分配了一个 block
        seq.block_table = [0]
        # len(seq) 即将变为 4，不需要新 block
        self.assertTrue(self.manager.can_append(seq))

        # 情况2：当前 block 已满 (现有4个 token)
        seq = self.create_mock_seq([1, 2, 3, 4])
        # len(seq) 即将变为 5 (5%4 == 1)，需要新 block
        # 如果 free 队列为空，则不能 append
        self.manager.free_block_ids.clear()
        self.assertFalse(self.manager.can_append(seq))

        # 恢复一个 block
        self.manager.free_block_ids.append(9)
        self.assertTrue(self.manager.can_append(seq))

    def test_may_append_and_hash_update(self):
        """测试 may_append 过程中 hash 的更新"""
        # 1. 分配一个未满 block 的序列 (3个 tokens, block_size=4)
        tokens = [1, 2, 3]
        seq = self.create_mock_seq(tokens)
        self.manager.allocate(seq)

        last_block_id = seq.block_table[-1]
        self.assertEqual(self.manager.blocks[last_block_id].hash, -1)

        # 2. 模拟增加一个 token 变为 4 个，刚好填满 block
        seq.token_ids.append(4)
        seq.num_tokens = 4
        seq.__len__.return_value = 3
        seq.block = lambda i: [1, 2, 3, 4]  # 更新 mock 的 block 返回值

        self.manager.may_append(seq)

        # 此时 block 填满了，hash 应该被计算并更新
        self.assertNotEqual(self.manager.blocks[last_block_id].hash, -1)
        self.assertIn(self.manager.blocks[last_block_id].hash, self.manager.hash_to_block_id)

    def test_ref_count_safety(self):
        """测试共享 Block 时的引用计数安全回收"""
        tokens = [1, 2, 3, 4]  # 刚好一个 block
        seq1 = self.create_mock_seq(tokens)
        seq2 = self.create_mock_seq(tokens)

        self.manager.allocate(seq1)
        self.manager.allocate(seq2)

        block_id = seq1.block_table[0]
        self.assertEqual(self.manager.blocks[block_id].ref_count, 2)

        # 释放 seq1，block 不应该回到 free 队列
        self.manager.deallocate(seq1)
        self.assertNotIn(block_id, self.manager.free_block_ids)
        self.assertEqual(self.manager.blocks[block_id].ref_count, 1)

        # 释放 seq2，block 应该回到 free 队列
        self.manager.deallocate(seq2)
        self.assertIn(block_id, self.manager.free_block_ids)
        self.assertEqual(self.manager.blocks[block_id].ref_count, 0)


if __name__ == '__main__':
    unittest.main()
