#!/usr/bin/env python3
"""The experimental hardware runner must reject truncated correctness gates."""
import copy
import unittest

from run_local_assembly_gpu import checked_decode


class ExactGate(unittest.TestCase):
    def setUp(self):
        self.data = dict(tokens=32, context=1024, tokens_per_launch=32,
                         matched=32, compared=32, layers_within_gate=27, layers=27,
                         produced=[1] * 32, ms=[3.595] * 32, median_ms=3.595)
        self.output = "tokens: 32/32 match HF greedy\nconsecutive layers within the gate: 27 of 27"

    def test_complete(self):
        checked_decode(self.data, self.output)

    def test_incomplete_or_invalid(self):
        for patch in (dict(tokens=31), dict(matched=31, compared=31),
                      dict(layers=26, layers_within_gate=26), dict(tokens_per_launch=1),
                      dict(context=512), dict(median_ms=float("nan")),
                      dict(produced=[1] * 31), dict(ms=[3.595] * 31)):
            with self.subTest(patch=patch), self.assertRaises(RuntimeError):
                checked_decode({**copy.deepcopy(self.data), **patch}, self.output)

    def test_stdout_and_abort(self):
        for output in ("", self.output.replace("32/32", "31/31"),
                       self.output.replace("27 of 27", "26 of 26"),
                       self.output + "\nabort: local assembly payload mismatch"):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                checked_decode(self.data, output)


if __name__ == "__main__":
    unittest.main()
