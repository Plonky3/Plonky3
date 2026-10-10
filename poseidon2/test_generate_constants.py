#!/usr/bin/env python3
"""The Python reference permutation reproduces every embedded default test vector.

Run with: python3 -m unittest poseidon2/test_generate_constants.py -v
"""

import importlib.util
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "generate_constants", Path(__file__).with_name("generate_constants.py")
)
generate_constants = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generate_constants)


class EmbeddedTestVectorTests(unittest.TestCase):
    def test_every_embedded_vector_has_a_default_diagonal(self):
        for field_name, width in generate_constants.DEFAULT_POSEIDON2_TEST_VECTORS:
            with self.subTest(field=field_name, width=width):
                diagonal = generate_constants.DEFAULT_INTERNAL_DIAGONALS[(field_name, width)]
                self.assertEqual(len(diagonal), width)

    def test_reference_permutation_reproduces_every_embedded_vector(self):
        for field_name, width in sorted(generate_constants.DEFAULT_POSEIDON2_TEST_VECTORS):
            with self.subTest(field=field_name, width=width):
                self.assertTrue(generate_constants.check_embedded_test_vector(field_name, width))


if __name__ == "__main__":
    unittest.main()
