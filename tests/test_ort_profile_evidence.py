"""Synthetic trace-parser tests; these are not GPU execution evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.ort_profile_report import inspect_profile


class ProfileEvidenceTests(unittest.TestCase):
    def inspect(self, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(events))
            return inspect_profile(path, "detection")

    def test_provider_name_without_model_execution_does_not_pass(self):
        self.assertEqual(self.inspect([{"cat": "Session", "args": {"provider": "TensorrtExecutionProvider"}}])["status"], "FAIL")

    def test_partial_gpu_or_weighted_cpu_path_does_not_pass(self):
        node = {"cat": "Node", "args": {"provider": "TensorrtExecutionProvider", "op_name": "TRTKernel",
            "input_type_shape": [{"float": [1, 3, 640, 640]}], "output_type_shape": [{"float": [1, 84, 8400]}]}}
        self.assertEqual(self.inspect([node] * 5)["status"], "PASS")
        cpu = {"cat": "Node", "args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}}
        self.assertEqual(self.inspect([node] * 5 + [cpu])["status"], "FAIL")
        node["args"]["output_type_shape"] = [{"float": [1, 64, 320, 320]}]
        self.assertEqual(self.inspect([node] * 5)["status"], "FAIL")


if __name__ == "__main__": unittest.main()
