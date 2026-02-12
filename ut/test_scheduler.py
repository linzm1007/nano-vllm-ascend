import unittest
from unittest.mock import MagicMock

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus, FinishReason


class TestScheduler(unittest.TestCase):
    def setUp(self):
        # 1. 模拟 Config
        self.config = MagicMock()
        self.config.max_num_seqs = 2
        self.config.max_num_batched_tokens = 100
        self.config.eos = 2
        self.config.num_kvcache_blocks = 100
        self.config.kvcache_block_size = 16
        self.config.max_model_len = 512

        # 2. 实例化 Scheduler
        # 注意：代码中 BlockManager 会在 __init__ 中被实例化，
        # 如果你想完全隔离，可以 mock BlockManager 类。
        self.scheduler = Scheduler(self.config)

    def create_mock_seq(self, seq_id, length, status=1):  # 1 = WAITING
        seq = MagicMock()
        seq.request_id = f"req_{seq_id}"
        seq.__len__.return_value = length
        seq.num_cached_tokens = 0
        seq.num_prompt_tokens = length
        seq.num_completion_tokens = 0
        seq.max_tokens = 50
        seq.status = status
        seq.ignore_eos = False
        seq.is_finished.return_value = False
        seq.finish_reason = None
        seq.block_table = []
        return seq

    def test_add_and_is_finished(self):
        self.assertTrue(self.scheduler.is_finished())
        seq = self.create_mock_seq(1, 10)
        self.scheduler.add(seq)
        self.assertFalse(self.scheduler.is_finished())
        self.assertEqual(len(self.scheduler.waiting), 1)

    def test_schedule_prefill_success(self):
        """测试正常的 Prefill 调度"""
        seq = self.create_mock_seq(1, 10)
        self.scheduler.add(seq)

        # 模拟 BlockManager 允许分配
        self.scheduler.block_manager.can_allocate = MagicMock(return_value=True)
        self.scheduler.block_manager.allocate = MagicMock()

        scheduled_seqs, is_prefill = self.scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(len(scheduled_seqs), 1)
        self.assertEqual(scheduled_seqs[0].status, SequenceStatus.RUNNING)  # RUNNING
        self.assertEqual(len(self.scheduler.running), 1)
        self.assertEqual(len(self.scheduler.waiting), 0)

    def test_schedule_prefill_batch_limit(self):
        """测试超过 max_num_batched_tokens 时的限制"""
        self.scheduler.max_num_batched_tokens = 15
        seq1 = self.create_mock_seq(1, 10)
        seq2 = self.create_mock_seq(2, 10)
        self.scheduler.add(seq1)
        self.scheduler.add(seq2)

        self.scheduler.block_manager.can_allocate = MagicMock(return_value=True)

        scheduled_seqs, _ = self.scheduler.schedule()

        # 只能调度第一个，因为 10+10 > 15
        self.assertEqual(len(scheduled_seqs), 1)
        self.assertEqual(scheduled_seqs[0].request_id, "req_1")

    def test_schedule_decode_and_preempt(self):
        """测试进入 Decode 阶段以及空间不足触发的抢占"""
        # 1. 先让一个序列进入 running
        seq1 = self.create_mock_seq(1, 10, status=3)  # RUNNING
        self.scheduler.running.append(seq1)

        # 2. 模拟 BlockManager 不允许 append (没有空余 Block)
        self.scheduler.block_manager.can_append = MagicMock(return_value=False)
        self.scheduler.block_manager.deallocate = MagicMock()

        # 调度
        scheduled_seqs, is_prefill = self.scheduler.schedule()

        # 此时应该触发抢占：seq1 从 running 移回到 waiting 头部
        self.assertFalse(is_prefill)
        self.assertEqual(len(scheduled_seqs), 0)
        self.assertEqual(len(self.scheduler.waiting), 1)
        self.assertEqual(self.scheduler.waiting[0].status, SequenceStatus.WAITING)
        self.assertEqual(self.scheduler.waiting[0].finish_reason, FinishReason.PREEMPTED)

    def test_postprocess_finished(self):
        """测试生成 EOS 后序列结束并释放资源"""
        seq = self.create_mock_seq(1, 10, status=3)
        self.scheduler.running.append(seq)

        self.scheduler.block_manager.deallocate = MagicMock()

        # 模拟生成了 EOS token
        token_ids = [self.config.eos]
        self.scheduler.postprocess([seq], token_ids)

        self.assertEqual(seq.status, SequenceStatus.FINISHED)  # FINISHED
        self.scheduler.block_manager.deallocate.assert_called_with(seq)
        self.assertEqual(len(self.scheduler.running), 0)


if __name__ == '__main__':
    unittest.main()
