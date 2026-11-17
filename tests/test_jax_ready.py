import unittest
from unittest.mock import Mock, patch

from expo_ft.utils.jax_ready import block_until_ready_serial


class ReadyTests(unittest.TestCase):
    def test_waits_all_leaves_in_order_without_batch_fanout(self):
        events = []
        a = Mock(block_until_ready=lambda: events.append('a'))
        b = Mock(block_until_ready=lambda: events.append('b'))
        tree = {'state': [a, b], 'metadata': None}
        with patch('jax.block_until_ready', side_effect=AssertionError('batched wait used')):
            self.assertIs(block_until_ready_serial(tree), tree)
        self.assertEqual(events, ['a', 'b'])

    def test_wait_failure_propagates(self):
        failed = RuntimeError('GPU failure')
        leaf = Mock(block_until_ready=Mock(side_effect=failed))
        with self.assertRaises(RuntimeError) as caught:
            block_until_ready_serial([leaf])
        self.assertIs(caught.exception, failed)
