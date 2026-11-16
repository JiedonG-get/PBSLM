import unittest
from evaluation.continuous_overload import execute


class ContinuousOverloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_admission_explicit_retry_and_ordered_drain(self):
        result=await execute()
        self.assertEqual(result['queued_while_worker_blocked'],4)
        self.assertEqual(result['pending_including_active_worker'],5)
        self.assertEqual(len(result['accepted_before_release']),5)
        self.assertEqual(len(result['explicit_rejections']),2)
        self.assertEqual(len(result['producer_retries']),2)
        self.assertEqual(len(result['processed_revisions']),7)
        self.assertTrue(result['no_silent_loss_of_accepted_inputs'])
        self.assertTrue(result['no_output_from_passive_inputs'])


if __name__=='__main__':unittest.main()
