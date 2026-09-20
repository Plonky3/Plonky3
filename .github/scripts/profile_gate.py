#!/usr/bin/env python3
"""Refuse a release whose measured profiles break, or escape, their declared budgets.

Reads the budget table and, when given one, a measurements file produced by the benchmark
harness. The comparison fails closed: an unlisted profile, a missing measurement and a missing
budget key are all failures, not silent passes.

Usage:
    profile_gate.py --budgets .github/profile-budgets.toml
    profile_gate.py --budgets ... --measurements measurements.json
    profile_gate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import unittest
from pathlib import Path

CEILINGS = ("proof_bytes", "verify_millis")
FLOORS = ("throughput_per_second",)
KEYS = CEILINGS + FLOORS
UNSET = "unset"


class GateError(Exception):
    """A budget table or a measurement set the gate refuses."""


def load_budgets(text: str) -> dict[str, dict[str, object]]:
    """Parse and validate the budget table, returning it keyed by profile name."""
    table = tomllib.loads(text)
    profiles = table.get("profile")
    if not profiles:
        raise GateError("the budget table lists no profiles")

    out: dict[str, dict[str, object]] = {}
    for entry in profiles:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise GateError("a profile has no name")
        if name in out:
            raise GateError(f"profile {name} is listed twice")
        if not entry.get("description"):
            raise GateError(f"profile {name} has no description")
        budgets: dict[str, object] = {}
        for key in KEYS:
            if key not in entry:
                raise GateError(f"profile {name} declares no {key}")
            value = entry[key]
            if value == UNSET:
                budgets[key] = None
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise GateError(f"profile {name} has a {key} that is neither positive nor unset")
            budgets[key] = value
        unknown = set(entry) - set(KEYS) - {"name", "description"}
        if unknown:
            raise GateError(f"profile {name} declares unknown keys {sorted(unknown)}")
        out[name] = budgets
    return out


def compare(budgets: dict[str, dict[str, object]], measured: dict[str, dict[str, float]]) -> list[str]:
    """Return every reason the measurements fail the budgets, in a stable order."""
    failures: list[str] = []
    for name in sorted(set(measured) - set(budgets)):
        failures.append(f"{name}: measured but carries no budget entry")
    for name in sorted(set(budgets) - set(measured)):
        failures.append(f"{name}: has a budget entry but was not measured")

    for name in sorted(set(budgets) & set(measured)):
        values = measured[name]
        for key in KEYS:
            if key not in values:
                failures.append(f"{name}: no measurement for {key}")
        limit_source = budgets[name]
        for key in CEILINGS:
            limit, value = limit_source[key], values.get(key)
            if limit is not None and value is not None and value > limit:
                failures.append(f"{name}: {key} is {value}, above the budget of {limit}")
        for key in FLOORS:
            limit, value = limit_source[key], values.get(key)
            if limit is not None and value is not None and value < limit:
                failures.append(f"{name}: {key} is {value}, below the budget of {limit}")
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.budgets is None:
        parser.error("--budgets is required unless --self-test is given")

    try:
        budgets = load_budgets(args.budgets.read_text())
    except GateError as error:
        print(f"::error::{error}")
        return 1

    unset = [n for n, b in budgets.items() if all(v is None for v in b.values())]
    print(f"{len(budgets)} profile(s) declared, {len(unset)} with every budget still unset")

    if args.measurements is None:
        return 0

    failures = compare(budgets, json.loads(args.measurements.read_text()))
    for failure in failures:
        print(f"::error::{failure}")
    return 1 if failures else 0


class GateTests(unittest.TestCase):
    SHIPPED = Path(__file__).resolve().parent.parent / "profile-budgets.toml"

    def budgets(self):
        return load_budgets(self.SHIPPED.read_text())

    def test_the_shipped_table_parses(self):
        self.assertTrue(self.budgets())

    def test_a_profile_missing_a_budget_key_is_refused(self):
        table = '[[profile]]\nname = "a"\ndescription = "d"\nproof_bytes = "unset"\n'
        with self.assertRaises(GateError):
            load_budgets(table)

    def test_a_negative_budget_is_refused(self):
        table = (
            '[[profile]]\nname = "a"\ndescription = "d"\n'
            'proof_bytes = -1\nverify_millis = "unset"\nthroughput_per_second = "unset"\n'
        )
        with self.assertRaises(GateError):
            load_budgets(table)

    def test_a_measured_profile_with_no_entry_fails(self):
        failures = compare(self.budgets(), {"an/unlisted-profile": {}})
        self.assertIn("carries no budget entry", "\n".join(failures))

    def test_a_declared_profile_that_was_not_measured_fails(self):
        self.assertTrue(compare(self.budgets(), {}))

    def test_a_ceiling_and_a_floor_both_bite(self):
        budgets = {"p": {"proof_bytes": 100, "verify_millis": None, "throughput_per_second": 10}}
        full = {"proof_bytes": 100, "verify_millis": 1, "throughput_per_second": 10}
        self.assertEqual(compare(budgets, {"p": full}), [])
        self.assertTrue(compare(budgets, {"p": {**full, "proof_bytes": 101}}))
        self.assertTrue(compare(budgets, {"p": {**full, "throughput_per_second": 9}}))

    def test_an_unset_budget_never_fails_on_the_value(self):
        budgets = {"p": {key: None for key in KEYS}}
        huge = {key: 10**9 for key in KEYS}
        self.assertEqual(compare(budgets, {"p": huge}), [])

    def test_every_declared_profile_names_a_shipped_entry_point(self):
        # A renamed or deleted example must not leave a budget entry pointing at nothing.
        root = self.SHIPPED.resolve().parent.parent
        for name in self.budgets():
            crate, target = name.split("/", 1)
            self.assertTrue(
                (root / crate / "examples" / f"{target}.rs").is_file(),
                f"{name} names no shipped example",
            )

    def test_a_missing_measurement_key_fails_even_when_unset(self):
        budgets = {"p": {key: None for key in KEYS}}
        self.assertTrue(compare(budgets, {"p": {}}))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
