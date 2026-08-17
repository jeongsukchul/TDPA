from __future__ import annotations

from tdpa.evaluation.oracle_gate import evaluate_manifest, gate_decision, make_manifest


def test_synthetic_oracle_first_gate_passes_predeclared_checks() -> None:
    manifest = [row for task in ("push", "lift") for row in make_manifest(task, [0, 1, 2], 2)]
    result = evaluate_manifest(manifest, ["no_adaptation", "oracle"])
    decision = gate_decision(result)
    assert decision["passed"], decision

