#!/usr/bin/env python3
"""
Validate review JSON files against the schema.

Usage:
    python3 scripts/reviews/validate.py docs/verifications/some-review.json
    python3 scripts/reviews/validate.py --all
"""

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_VERIFICATIONS = REPO_ROOT / "docs" / "verifications"

VALID_VERDICTS = {"VERIFIED", "NEEDS_CLARIFICATION", "DISPUTED", "UNVERIFIABLE"}
VALID_OVERALL = {"ACCEPT", "ACCEPT_WITH_REVISION", "REVISE_AND_RESUBMIT", "REJECT", "UNKNOWN"}
VALID_CERT = {"gold", "silver", "bronze", "uncertified", ""}

# Required fields for v2+ schema
REQUIRED_FIELDS = {"finding_slug"}
REVIEW_FIELDS = {"finding_slug", "overall_verdict"}


def validate_file(path):
    """Validate a single review JSON. Returns list of (severity, message)."""
    issues = []
    basename = Path(path).name

    if basename == "manifest.json":
        return []

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [("ERROR", f"Invalid JSON: {e}")]
    except UnicodeDecodeError as e:
        return [("ERROR", f"Encoding error: {e}")]

    if not isinstance(data, dict):
        return [("ERROR", "Root must be a JSON object")]

    # Check finding_slug
    if "finding_slug" not in data:
        issues.append(("ERROR", "Missing required field: finding_slug"))

    # Skip validation for revision-response files
    if data.get("entry_type") == "revision-response":
        return issues

    # Check overall_verdict if present
    verdict = data.get("overall_verdict", "")
    if verdict and verdict not in VALID_OVERALL:
        issues.append(("WARN", f"Unknown overall_verdict: '{verdict}' (expected one of {VALID_OVERALL})"))

    # Check certification_recommendation
    cert = data.get("certification_recommendation", "")
    if cert and cert.lower() not in VALID_CERT:
        issues.append(("WARN", f"Unknown certification_recommendation: '{cert}'"))

    # Check claim_reviews
    claims = data.get("claim_reviews", [])
    if claims:
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                issues.append(("ERROR", f"claim_reviews[{i}] is not an object"))
                continue
            v = claim.get("verdict", "")
            if v and v not in VALID_VERDICTS:
                issues.append(("WARN", f"claim_reviews[{i}].verdict unknown: '{v}'"))
            if not claim.get("claim"):
                issues.append(("WARN", f"claim_reviews[{i}] has no 'claim' text"))

    # Check reviewer
    reviewer = data.get("reviewer", {})
    if reviewer and not reviewer.get("model") and not reviewer.get("human_name") and not data.get("verified_by"):
        issues.append(("WARN", "No reviewer.model, reviewer.human_name, or verified_by"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate review JSON files")
    parser.add_argument("files", nargs="*", help="Specific files to validate")
    parser.add_argument("--all", action="store_true", help="Validate all files in docs/verifications/")
    parser.add_argument("--verifications-dir", default=str(DEFAULT_VERIFICATIONS))
    args = parser.parse_args()

    if args.all:
        files = sorted(glob.glob(str(Path(args.verifications_dir) / "*.json")))
    elif args.files:
        files = args.files
    else:
        parser.print_help()
        sys.exit(1)

    total = 0
    errors = 0
    warnings = 0

    for f in files:
        basename = Path(f).name
        issues = validate_file(f)
        total += 1

        if issues:
            for severity, msg in issues:
                prefix = "ERROR" if severity == "ERROR" else "WARN "
                print(f"  [{prefix}] {basename}: {msg}")
                if severity == "ERROR":
                    errors += 1
                else:
                    warnings += 1
        # No output for clean files (quiet by default)

    print(f"\nValidated {total} files: {errors} errors, {warnings} warnings")
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
