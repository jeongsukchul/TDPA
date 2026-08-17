from __future__ import annotations

from dataclasses import asdict

from tdpa.evaluation.oracle_gate import evaluate_manifest, make_manifest


def test_manifest_and_rollout_evaluation_are_deterministic() -> None:
    first_manifest = make_manifest("push", [7], 1)
    second_manifest = make_manifest("push", [7], 1)
    assert [asdict(row) for row in first_manifest] == [asdict(row) for row in second_manifest]
    first = evaluate_manifest(first_manifest, ["no_adaptation", "oracle"])
    second = evaluate_manifest(second_manifest, ["no_adaptation", "oracle"])
    assert first == second


def test_methods_consume_identical_episode_metadata() -> None:
    manifest = make_manifest("lift", [3], 1)
    result = evaluate_manifest(manifest, ["no_adaptation", "oracle"])
    keys = ("task", "split", "seed", "index", "mass", "friction", "behavior_policy_id")
    nominal = [row for row in result["episodes"] if row["method"] == "no_adaptation"]
    oracle = [row for row in result["episodes"] if row["method"] == "oracle"]
    assert [{key: row[key] for key in keys} for row in nominal] == [
        {key: row[key] for key in keys} for row in oracle
    ]


def test_policy_shift_changes_action_trace_at_identical_physics() -> None:
    manifest = make_manifest("push", [11], 1)
    identity = next(row for row in manifest if row.split == "id")
    shifted = next(row for row in manifest if row.split == "policy_shift")
    assert (identity.mass, identity.friction) == (shifted.mass, shifted.friction)
    result = evaluate_manifest([identity, shifted], ["no_adaptation"])["episodes"]
    assert result[0]["action_trace_hash"] != result[1]["action_trace_hash"]
