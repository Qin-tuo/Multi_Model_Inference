"""Check pinned exact matching and all-rotation semantics without cloud judges."""
import copy
from pathlib import Path
import ssl
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.mmbench_scoring import evaluate_group


class ScorerTests(unittest.TestCase):
    def test_all_rotations_and_tls_state(self):
        before=ssl._create_default_https_context
        rows=[{'A':'蓝色','B':'红色','GT':'A','prediction':'A'},
              {'A':'红色','B':'蓝色','GT':'B','prediction':'蓝色'}]
        original=copy.deepcopy(rows);result=evaluate_group(rows)
        self.assertTrue(result['circular_correct']);self.assertEqual(result['parsed'],['A','B'])
        self.assertEqual(rows,original);self.assertIs(ssl._create_default_https_context,before)

    def test_one_wrong_or_unparsed_rotation_fails_group(self):
        for answer in ('A','不知道'):
            result=evaluate_group([{'A':'蓝色','B':'红色','GT':'A','prediction':'A'},
                {'A':'红色','B':'蓝色','GT':'B','prediction':answer}])
            self.assertFalse(result['circular_correct'])


if __name__=='__main__':unittest.main(verbosity=2)
