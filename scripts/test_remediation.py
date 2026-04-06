#!/usr/bin/env python3
"""Tests for the research agent's remediation logic.

Tests gather_evidence, validate_fix, and the remediation prompt construction
without calling any real LLM. Uses mock data to verify that:
1. Evidence is gathered from experiment data directories
2. Hedge-only fixes are rejected
3. Substantive fixes (with real numbers) are accepted
4. Status is set correctly (only "fix" => "resolved")
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Add parent dir so we can import the agent
sys.path.insert(0, str(Path(__file__).parent))

# We need to patch the globals before importing
import research_agent as agent


def test_validate_fix_rejects_hedges():
    """Fixes that only add 'not yet quantified' etc. should be rejected."""
    cases = [
        # (old, new, should_be_valid)
        ("strong rank correlation", "strong rank correlation — coefficients not yet quantified", False),
        ("N=15 nodes", "N=15 nodes; truncation error not yet quantified", False),
        ("costs only 0.002", "costs only 0.002 — pending further analysis", False),
        ("costs only 0.002", "costs only 0.002 (future work needed to confirm)", False),
    ]
    passed = 0
    for old, new, expected_valid in cases:
        is_valid, reason = agent.validate_fix(old, new)
        if is_valid == expected_valid:
            passed += 1
            print(f"  PASS: validate_fix({old!r:.40s}...) => valid={is_valid}")
        else:
            print(f"  FAIL: validate_fix({old!r:.40s}...) => valid={is_valid}, expected={expected_valid}, reason={reason}")
    return passed, len(cases)


def test_validate_fix_accepts_real_data():
    """Fixes that insert actual numbers/coefficients should be accepted."""
    cases = [
        ("strong rank correlation",
         "Spearman rho = 0.9934 across all 32,767 subsets; Kendall tau = 0.9821",
         True),
        ("N=15 nodes",
         "N=15 nodes (convergence: N=25 gives delta < 0.0003 for all tested subsets; error ≈ 3e-4)",
         True),
        ("removing digit 20 costs only 0.002",
         "removing digit 20 costs 0.00187 ± 0.0003 (convergence at N=25,35 confirms this exceeds truncation error)",
         True),
    ]
    passed = 0
    for old, new, expected_valid in cases:
        is_valid, reason = agent.validate_fix(old, new)
        if is_valid == expected_valid:
            passed += 1
            print(f"  PASS: validate_fix accepts real data: valid={is_valid}")
        else:
            print(f"  FAIL: validate_fix({new!r:.50s}...) => valid={is_valid}, expected={expected_valid}, reason={reason}")
    return passed, len(cases)


def test_validate_fix_edge_cases():
    """Edge cases: empty strings, identical text."""
    cases = [
        ("", "something", False),  # empty old_text
        ("something", "", False),  # empty new_text
    ]
    passed = 0
    for old, new, expected_valid in cases:
        is_valid, reason = agent.validate_fix(old, new)
        if is_valid == expected_valid:
            passed += 1
            print(f"  PASS: edge case ({reason})")
        else:
            print(f"  FAIL: edge case => valid={is_valid}, expected={expected_valid}")
    return passed, len(cases)


def test_gather_evidence():
    """gather_evidence should find CSVs, metadata, and logs from experiment dirs."""
    # Create a temporary file structure mimicking a finding + experiment
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Mock WEBSITE_ROOT and REPO_ROOT
        website = tmpdir / "bigcompute.science"
        repo = tmpdir / "idontknow"
        findings_dir = website / "src" / "content" / "findings"
        findings_dir.mkdir(parents=True)
        exp_dir = repo / "scripts" / "experiments" / "hausdorff-spectrum" / "results"
        exp_dir.mkdir(parents=True)

        # Write a finding with related_experiment
        (findings_dir / "2026-03-29-test-finding.md").write_text(
            "---\ntitle: Test\nslug: test-finding\nrelated_experiment: /experiments/hausdorff-spectrum/\n---\nBody\n"
        )

        # Write experiment data
        (exp_dir / "metadata_n5.json").write_text('{"n": 5, "subsets": 31, "time_s": 0.5}')
        (exp_dir / "spectrum_n5.csv").write_text(
            "subset_mask,subset_digits,cardinality,max_digit_in_subset,dimension\n"
            "1,{1},1,1,0.000000000000000\n"
            "2,{2},1,2,0.000000000000000\n"
            "31,{1,2,3,4,5},5,5,0.836829443681209\n"
        )
        (exp_dir / "run_n5.log").write_text("Starting computation\nDone in 0.5s\n")

        # Patch globals
        with patch.object(agent, 'REPO_ROOT', repo), \
             patch.object(agent, 'WEBSITE_ROOT', website):
            evidence = agent.gather_evidence("test-finding")

        passed = 0
        total = 4
        if "metadata_n5.json" in evidence:
            passed += 1
            print("  PASS: metadata found in evidence")
        else:
            print("  FAIL: metadata not found in evidence")

        if "spectrum_n5.csv" in evidence:
            passed += 1
            print("  PASS: CSV found in evidence")
        else:
            print("  FAIL: CSV not found in evidence")

        if "0.836829" in evidence:
            passed += 1
            print("  PASS: actual dimension value in evidence")
        else:
            print("  FAIL: dimension value not in evidence")

        if "run_n5.log" in evidence:
            passed += 1
            print("  PASS: log found in evidence")
        else:
            print("  FAIL: log not found in evidence")

        return passed, total


def test_remediation_status_logic():
    """Only action='fix' that passes validation should produce status='resolved'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Set up minimal file structure
        website = tmpdir / "bigcompute.science"
        repo = tmpdir / "idontknow"
        findings_dir = website / "src" / "content" / "findings"
        findings_dir.mkdir(parents=True)
        verifications_dir = repo / "docs" / "verifications"
        verifications_dir.mkdir(parents=True)
        exp_results = repo / "scripts" / "experiments" / "hausdorff-spectrum" / "results"
        exp_results.mkdir(parents=True)

        # Write finding
        finding_file = findings_dir / "2026-03-29-test-slug.md"
        finding_file.write_text(
            "---\ntitle: Test\nslug: test-slug\n"
            "related_experiment: /experiments/hausdorff-spectrum/\n---\n"
            "strong rank correlation\n"
            "N=15 nodes\n"
        )

        # Write a review with DISPUTED claims
        review_file = verifications_dir / "test-slug_o3pro-review.json"
        review_file.write_text(json.dumps({
            "reviewer": {"model": "o3-pro"},
            "claim_reviews": [
                {"verdict": "DISPUTED", "claim": "strong rank correlation",
                 "concerns": "Supply correlation coefficients"},
                {"verdict": "NEEDS_CLARIFICATION", "claim": "N=15 nodes delivers accuracy",
                 "concerns": "Quantify truncation error"},
            ]
        }))

        # Mock LLM response: one real fix, one hedge fix (should be demoted)
        mock_llm_response = json.dumps({"remediations": [
            {
                "claim": "strong rank correlation",
                "action": "fix",
                "description": "Added Spearman correlation",
                "old_text": "strong rank correlation",
                "new_text": "Spearman rho = 0.9934 across all 32,767 subsets (rank correlation)"
            },
            {
                "claim": "N=15 nodes delivers accuracy",
                "action": "fix",
                "description": "Added hedge about truncation error",
                "old_text": "N=15 nodes",
                "new_text": "N=15 nodes; truncation error not yet quantified"
            },
        ]})

        with patch.object(agent, 'REPO_ROOT', repo), \
             patch.object(agent, 'WEBSITE_ROOT', website), \
             patch.object(agent, 'call_llm', return_value=mock_llm_response), \
             patch('subprocess.run') as mock_subprocess:
            # Mock git log for commit hash
            mock_subprocess.return_value = MagicMock(stdout="abc1234\n", returncode=0)
            agent.remediate(["test-slug"])

        # Check the finding was updated with the real fix but not the hedge
        content = finding_file.read_text()
        passed = 0
        total = 4

        if "Spearman rho = 0.9934" in content:
            passed += 1
            print("  PASS: real fix applied to finding")
        else:
            print("  FAIL: real fix not applied")

        if "truncation error not yet quantified" not in content:
            passed += 1
            print("  PASS: hedge fix was NOT applied")
        else:
            print("  FAIL: hedge fix was applied (should have been rejected)")

        # Check remediation JSONs
        rem_dir = verifications_dir / "remediations" / "test-slug"
        rem_files = list(rem_dir.glob("*.json"))
        if len(rem_files) >= 2:
            passed += 1
            print(f"  PASS: {len(rem_files)} remediation JSON(s) written")
        else:
            print(f"  FAIL: expected >=2 remediation JSONs, got {len(rem_files)}")

        # Check that the hedge fix got demoted to "acknowledged"
        statuses = {}
        for rf in rem_files:
            data = json.loads(rf.read_text())
            statuses[data.get("issue_id", "")] = data.get("status", "")

        has_acknowledged = "acknowledged" in statuses.values()
        has_resolved = "resolved" in statuses.values()
        if has_acknowledged and has_resolved:
            passed += 1
            print("  PASS: hedge demoted to 'acknowledged', real fix is 'resolved'")
        else:
            print(f"  FAIL: statuses = {statuses}")

        return passed, total


def test_severity_from_original_verdict():
    """Severity should come from the original review verdict, not a stale variable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        website = tmpdir / "bigcompute.science"
        repo = tmpdir / "idontknow"
        findings_dir = website / "src" / "content" / "findings"
        findings_dir.mkdir(parents=True)
        verifications_dir = repo / "docs" / "verifications"
        verifications_dir.mkdir(parents=True)
        (repo / "scripts" / "experiments").mkdir(parents=True)

        finding_file = findings_dir / "2026-01-01-sev-test.md"
        finding_file.write_text("---\nslug: sev-test\n---\nold text A\nold text B\n")

        # One DISPUTED, one NEEDS_CLARIFICATION
        review_file = verifications_dir / "sev-test_review.json"
        review_file.write_text(json.dumps({
            "reviewer": {"model": "test"},
            "claim_reviews": [
                {"verdict": "DISPUTED", "claim": "old text A claim",
                 "concerns": "fix this"},
                {"verdict": "NEEDS_CLARIFICATION", "claim": "old text B claim",
                 "concerns": "clarify this"},
            ]
        }))

        mock_response = json.dumps({"remediations": [
            {"claim": "old text A claim", "action": "fix", "description": "fixed A",
             "old_text": "old text A", "new_text": "new text A with rho = 0.99 data"},
            {"claim": "old text B claim", "action": "fix", "description": "fixed B",
             "old_text": "old text B", "new_text": "new text B with tau = 0.88 values"},
        ]})

        with patch.object(agent, 'REPO_ROOT', repo), \
             patch.object(agent, 'WEBSITE_ROOT', website), \
             patch.object(agent, 'call_llm', return_value=mock_response), \
             patch('subprocess.run') as mock_sub:
            mock_sub.return_value = MagicMock(stdout="def5678\n", returncode=0)
            agent.remediate(["sev-test"])

        rem_dir = verifications_dir / "remediations" / "sev-test"
        passed = 0
        total = 2

        for rf in rem_dir.glob("*.json"):
            data = json.loads(rf.read_text())
            if "old-text-a" in data["issue_id"]:
                if data["severity"] == "important":
                    passed += 1
                    print("  PASS: DISPUTED claim => severity 'important'")
                else:
                    print(f"  FAIL: DISPUTED claim got severity '{data['severity']}'")
            elif "old-text-b" in data["issue_id"]:
                if data["severity"] == "minor":
                    passed += 1
                    print("  PASS: NEEDS_CLARIFICATION claim => severity 'minor'")
                else:
                    print(f"  FAIL: NEEDS_CLARIFICATION claim got severity '{data['severity']}'")

        return passed, total


def main():
    total_passed = 0
    total_tests = 0

    suites = [
        ("validate_fix rejects hedges", test_validate_fix_rejects_hedges),
        ("validate_fix accepts real data", test_validate_fix_accepts_real_data),
        ("validate_fix edge cases", test_validate_fix_edge_cases),
        ("gather_evidence", test_gather_evidence),
        ("remediation status logic", test_remediation_status_logic),
        ("severity from original verdict", test_severity_from_original_verdict),
    ]

    for name, test_fn in suites:
        print(f"\n--- {name} ---")
        try:
            p, t = test_fn()
            total_passed += p
            total_tests += t
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            total_tests += 1

    print(f"\n{'='*40}")
    print(f"Results: {total_passed}/{total_tests} passed")
    if total_passed == total_tests:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {total_tests - total_passed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
