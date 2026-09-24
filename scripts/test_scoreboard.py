"""Tests for the scoreboard runner's parsing and workload plan."""

from __future__ import annotations

import unittest

from scoreboard import (
    SECURITY_BITS,
    THREAD_COUNTS,
    WORKLOADS,
    available_objectives,
    gate_failures,
    hashes_proved,
    markdown,
    peak_rss_bytes,
    prover_command,
    throughput,
    time_flag,
    workload_rows,
)


def run(**overrides: object) -> dict:
    """One report, with the fields a gate reads."""
    report = {
        "objective": "blake-3-compressions",
        "log_trace_length": 10,
        "threads": 8,
        "proof_bytes": 597_212,
        "security_bits": 97.118384,
        "prove_seconds": 0.06,
    }
    report.update(overrides)
    return report


BASELINE = {
    "runs": [
        {
            "objective": "blake-3-compressions",
            "log_trace_length": 10,
            "proof_bytes": 597_212,
            "security_bits": 97.118384,
        }
    ]
}


class GateTest(unittest.TestCase):
    def test_an_unchanged_run_passes(self) -> None:
        self.assertEqual(gate_failures([run()], BASELINE, 1.5), [])

    def test_a_changed_proof_size_fails(self) -> None:
        failures = gate_failures([run(proof_bytes=597_213)], BASELINE, 1.5)
        self.assertEqual(len(failures), 1)
        self.assertIn("proof is 597213 bytes", failures[0])

    def test_lost_security_fails(self) -> None:
        failures = gate_failures([run(security_bits=95.0)], BASELINE, 1.5)
        self.assertEqual(len(failures), 1)
        self.assertIn("security is 95.00 bits", failures[0])

    def test_the_gate_holds_a_machine_of_any_core_count(self) -> None:
        # The baseline is keyed on the workload, not the thread count, so a machine
        # with different cores is still held to the same proof.
        self.assertEqual(gate_failures([run(threads=1)], BASELINE, 1.5), [])
        self.assertEqual(
            len(gate_failures([run(threads=96, proof_bytes=1)], BASELINE, 1.5)), 1
        )

    def test_a_proof_that_moves_with_the_thread_count_fails_on_its_own(self) -> None:
        # No baseline row is needed: a proof that depends on how many threads produced
        # it is wrong whatever the committed size says.
        failures = gate_failures(
            [run(threads=1), run(threads=8, proof_bytes=597_999)], {"runs": []}, 1.5
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("on 8 thread(s) but", failures[0])

    def test_a_timing_budget_is_gated_with_slack(self) -> None:
        baseline = {
            "runs": [dict(BASELINE["runs"][0], prove_seconds_budget=0.06)]
        }
        self.assertEqual(gate_failures([run(prove_seconds=0.08)], baseline, 1.5), [])
        failures = gate_failures([run(prove_seconds=0.5)], baseline, 1.5)
        self.assertEqual(len(failures), 1)
        self.assertIn("budget is 0.060 s", failures[0])


class ThroughputTest(unittest.TestCase):
    def test_keccak_counts_permutations_not_rows(self) -> None:
        # 25 rows go into one permutation, so the trace proves far fewer hashes than rows.
        self.assertEqual(hashes_proved("keccak-f-permutations", 14), 16384 // 25)

    def test_a_compression_is_one_row(self) -> None:
        self.assertEqual(hashes_proved("blake-3-compressions", 10), 1024)

    def test_throughput_is_hashes_per_proving_second(self) -> None:
        rate = throughput(run(prove_seconds=0.5))
        self.assertAlmostEqual(rate, 2048.0)


class PeakMemoryTest(unittest.TestCase):
    def test_reads_the_macos_line_as_bytes(self) -> None:
        self.assertEqual(
            peak_rss_bytes("        1234567  maximum resident set size"), 1234567
        )

    def test_reads_the_gnu_line_as_kilobytes(self) -> None:
        self.assertEqual(
            peak_rss_bytes("\tMaximum resident set size (kbytes): 4096"), 4096 * 1024
        )

    def test_an_unfamiliar_time_costs_only_the_memory_column(self) -> None:
        self.assertIsNone(peak_rss_bytes("real 1.2\nuser 0.9\nsys 0.1"))


class TimeFlagTest(unittest.TestCase):
    def test_macos_and_gnu_take_different_flags(self) -> None:
        self.assertEqual(time_flag("Darwin"), "-l")
        self.assertEqual(time_flag("Linux"), "-v")


class ObjectivesTest(unittest.TestCase):
    def test_reads_the_objectives_a_build_offers(self) -> None:
        self.assertEqual(
            available_objectives("[possible values: a-compressions, b-permutations]"),
            {"a-compressions", "b-permutations"},
        )

    def test_a_build_without_the_list_reports_none(self) -> None:
        self.assertEqual(available_objectives("no values here"), set())

    def test_an_objective_a_build_lacks_is_skipped(self) -> None:
        present = {WORKLOADS[0][0]}
        rows = list(workload_rows(present))
        self.assertTrue(rows)
        self.assertEqual({objective for objective, _, _ in rows}, present)


class PlanTest(unittest.TestCase):
    def test_every_workload_runs_at_every_thread_count(self) -> None:
        rows = list(workload_rows())
        expected = sum(len(heights) for _, heights in WORKLOADS) * len(THREAD_COUNTS)
        self.assertEqual(len(rows), expected)
        self.assertIn(1, {threads for _, _, threads in rows})
        self.assertIn(None, {threads for _, _, threads in rows})

    def test_every_run_asks_for_the_frozen_security_and_json(self) -> None:
        command = prover_command("blake-3-compressions", 10, binary="/tmp/prover")
        self.assertEqual(command[0], "/tmp/prover")
        self.assertIn("--format", command)
        self.assertEqual(command[command.index("--format") + 1], "json")
        self.assertEqual(
            command[command.index("--security-bits") + 1], str(SECURITY_BITS)
        )


class MarkdownTest(unittest.TestCase):
    def test_a_row_carries_its_security_and_its_memory(self) -> None:
        table = markdown(
            [
                {
                    "objective": "blake-3-compressions",
                    "log_trace_length": 10,
                    "width": 11536,
                    "threads": 8,
                    "witness_seconds": 0.002,
                    "prove_seconds": 0.06,
                    "verify_seconds": 0.042,
                    "serialize_seconds": 0.0004,
                    "proof_bytes": 597_000,
                    "security_bits": 97.1,
                    "peak_rss_bytes": 512 * (1 << 20),
                }
            ]
        )
        self.assertIn("blake-3-compressions", table)
        self.assertIn("512 MiB", table)
        self.assertIn("97.1", table)

    def test_a_run_without_memory_still_tabulates(self) -> None:
        table = markdown(
            [
                {
                    "objective": "sha-256-compressions",
                    "log_trace_length": 10,
                    "width": 23712,
                    "threads": 1,
                    "witness_seconds": 0.02,
                    "prove_seconds": 0.5,
                    "verify_seconds": 0.1,
                    "serialize_seconds": 0.001,
                    "proof_bytes": 856_000,
                    "security_bits": 97.1,
                    "peak_rss_bytes": None,
                }
            ]
        )
        self.assertIn("n/a", table)


if __name__ == "__main__":
    unittest.main()
