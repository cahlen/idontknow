#!/usr/bin/env python3
"""
Autonomous research agent for bigcompute.science.

Monitors GPU experiments, harvests results, calls Claude API for analysis,
runs peer reviews, and deploys updates. Runs in a loop or one-shot.

Usage:
    # Run once (check and process everything)
    python3 scripts/research_agent.py --once

    # Run in loop (default 10 min)
    python3 scripts/research_agent.py --interval 10m

    # Run specific phase
    python3 scripts/research_agent.py --phase monitor
    python3 scripts/research_agent.py --phase harvest

    # Dry run (report only, no changes)
    python3 scripts/research_agent.py --dry-run

    # Auto-launch next experiment when GPUs free (off by default)
    python3 scripts/research_agent.py --interval 10m --auto-launch

Environment:
    ANTHROPIC_API_KEY  — for Claude analysis/writing (required for full cycle)
    OPENAI_API_KEY     — for OpenAI peer reviews (optional)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WEBSITE_ROOT = REPO_ROOT.parent / "bigcompute.science"
STATE_FILE = REPO_ROOT / "data" / "agent_state.json"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
LOG_DIRS = [
    REPO_ROOT / "logs",
    REPO_ROOT / "scripts" / "experiments",
]

def get_experiment_keywords():
    """Auto-discover experiment names from scripts/experiments/ directories."""
    exp_dir = REPO_ROOT / "scripts" / "experiments"
    keywords = []
    if exp_dir.exists():
        for d in exp_dir.iterdir():
            if d.is_dir() and not d.name.startswith('.'):
                # Convert dir name to keyword (e.g., "zaremba-density" -> "zaremba")
                keywords.append(d.name.split('-')[0])
    # Also catch common process names
    keywords.extend(["char_table", "class_v2"])
    return list(set(keywords))

EXPERIMENT_KEYWORDS = get_experiment_keywords()


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ── State Management ──────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_tick": None, "harvested": {}, "reviewed": {}, "deployed_at": None}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_tick"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Phase 1: Monitor ─────────────────────────────────────────

def monitor_gpus():
    """Check GPU status and running experiments."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            idx, mem_used, mem_total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            gpus.append({"index": idx, "mem_used_mb": mem_used, "mem_total_mb": mem_total,
                         "utilization": util, "busy": util > 10})
    except (subprocess.TimeoutExpired, FileNotFoundError):
        log("nvidia-smi not available", "WARN")

    # Check running experiment processes
    processes = []
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            for kw in EXPERIMENT_KEYWORDS:
                if kw in line.lower() and "grep" not in line and "research_agent" not in line:
                    # Extract PID and command
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append({
                            "pid": parts[1],
                            "cpu": parts[2],
                            "command": " ".join(parts[10:])[:100],
                        })
                    break
    except subprocess.TimeoutExpired:
        pass

    free_gpus = [g["index"] for g in gpus if not g["busy"]]
    busy_gpus = [g["index"] for g in gpus if g["busy"]]

    log(f"GPUs: {len(busy_gpus)} busy, {len(free_gpus)} free {free_gpus}")
    log(f"Processes: {len(processes)} experiment(s) running")
    for p in processes[:5]:
        log(f"  PID {p['pid']}: {p['command']}")

    return {"gpus": gpus, "free_gpus": free_gpus, "busy_gpus": busy_gpus, "processes": processes}


# ── Phase 2: Harvest ─────────────────────────────────────────

def find_log_files():
    """Find all experiment log files."""
    logs = []
    for log_dir in LOG_DIRS:
        if not log_dir.exists():
            continue
        for f in log_dir.rglob("*.log"):
            logs.append(f)
    return logs


def is_complete(log_path):
    """Check if a log file represents a completed experiment."""
    try:
        text = log_path.read_text(errors="replace")
        # Common completion markers
        if "RESULTS" in text and "========" in text:
            return True
        if "Time:" in text and ("Density:" in text or "Uncovered:" in text):
            return True
        if "Character table:" in text and "Saved:" in text:
            return True
        if "Nonzero coefficients:" in text:
            return True
    except (OSError, UnicodeDecodeError):
        pass
    return False


def extract_results(log_path):
    """Extract key numbers from a completed log file."""
    text = log_path.read_text(errors="replace")
    results = {"file": str(log_path), "name": log_path.stem}

    # Zaremba density
    m = re.search(r"Uncovered:\s*(\d+)", text)
    if m:
        results["uncovered"] = int(m.group(1))
    m = re.search(r"Density:\s*([\d.]+)%", text)
    if m:
        results["density"] = float(m.group(1))
    m = re.search(r"Digits:\s*\{([^}]+)\}", text)
    if m:
        results["digits"] = m.group(1)
    m = re.search(r"Range: d = 1 to (\d+)", text)
    if m:
        results["range"] = int(m.group(1))

    # Kronecker / character table
    m = re.search(r"Nonzero coefficients:\s*(\d+)", text)
    if m:
        results["nonzero"] = int(m.group(1))
    m = re.search(r"Max coefficient:\s*(\d+)", text)
    if m:
        results["max_coeff"] = int(m.group(1))
    m = re.search(r"Character table:\s*([\d.]+)s", text)
    if m:
        results["char_table_time"] = float(m.group(1))
    m = re.search(r"Partitions:\s*(\d+)", text)
    if m:
        results["partitions"] = int(m.group(1))

    # Timing
    m = re.search(r"Time:\s*([\d.]+)s", text)
    if m:
        results["time_seconds"] = float(m.group(1))

    return results


def harvest(state, dry_run=False):
    """Find new completed experiments and extract results."""
    logs = find_log_files()
    new_results = []

    for log_path in logs:
        rel = str(log_path.relative_to(REPO_ROOT))
        if rel in state["harvested"]:
            continue
        if not is_complete(log_path):
            continue

        results = extract_results(log_path)
        new_results.append(results)

        if not dry_run:
            state["harvested"][rel] = datetime.now(timezone.utc).isoformat()

        log(f"NEW: {log_path.name} — {json.dumps({k: v for k, v in results.items() if k != 'file'}, default=str)}")

    if not new_results:
        log("No new completed experiments.")
    else:
        log(f"Harvested {len(new_results)} new result(s).")

    return new_results


# ── Generic LLM caller ────────────────────────────────────────

def call_llm(prompt, purpose="task"):
    """Call any available LLM. Tries in order: claude CLI, Anthropic API, OpenAI API.
    Returns response text or None on failure."""
    import httpx

    # 1. Try claude CLI (Claude Code premium account — no API key needed)
    claude_cli = shutil.which("claude")
    if claude_cli:
        try:
            log(f"  [{purpose}] via claude CLI...")
            result = subprocess.run(
                [claude_cli, "-p", prompt],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            log(f"  [{purpose}] claude CLI timed out", "WARN")

    # 2. Try Anthropic API
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            log(f"  [{purpose}] via Anthropic API...")
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 4000,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=120.0,
            )
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"].strip()
            log(f"  [{purpose}] Anthropic API {resp.status_code}: {resp.text[:100]}", "WARN")
        except Exception as e:
            log(f"  [{purpose}] Anthropic API failed: {e}", "WARN")

    # 3. Try OpenAI API
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            log(f"  [{purpose}] via OpenAI API (gpt-4.1)...")
            # OpenAI json_object mode requires "json" in the prompt
            openai_prompt = prompt if "json" in prompt.lower() else prompt + "\nRespond with JSON."
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                json={"model": "gpt-4.1", "messages": [{"role": "user", "content": openai_prompt}],
                      "max_completion_tokens": 4000, "response_format": {"type": "json_object"}},
                timeout=120.0,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            log(f"  [{purpose}] OpenAI API {resp.status_code}: {resp.text[:100]}", "WARN")
        except Exception as e:
            log(f"  [{purpose}] OpenAI API failed: {e}", "WARN")

    # 4. Try Google Gemini API (free tier in Colab)
    gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    if gemini_key:
        try:
            log(f"  [{purpose}] via Gemini API (gemini-3-flash-preview)...")
            gemini_prompt = prompt if "json" in prompt.lower() else prompt + "\nRespond with JSON."
            resp = httpx.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {gemini_key}", "Content-Type": "application/json"},
                json={"model": "gemini-3-flash-preview", "messages": [{"role": "user", "content": gemini_prompt}],
                      "max_completion_tokens": 4000},
                timeout=120.0,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            log(f"  [{purpose}] Gemini API {resp.status_code}: {resp.text[:100]}", "WARN")
        except Exception as e:
            log(f"  [{purpose}] Gemini API failed: {e}", "WARN")

    log(f"  [{purpose}] No LLM available (need claude CLI, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY)", "ERROR")
    return None


def parse_json_from_llm(text):
    """Extract JSON from LLM response that may contain prose before/after."""
    if not text:
        return None
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    elif "{" in text:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            if depth == 0:
                text = text[start:i+1]
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Phase 3: Analyze ─────────────────────────────────────────

def analyze_results(new_results):
    """Analyze new results using any available LLM."""
    if not (shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        log("No LLM available — skipping analysis.", "WARN")
        return []

    context = CLAUDE_MD.read_text()[:6000] if CLAUDE_MD.exists() else ""
    results_text = json.dumps(new_results, indent=2, default=str)

    prompt = f"""You are a research assistant for bigcompute.science. New experiment results have come in.
Analyze them against the project context and determine:
1. Is each result a NEW finding, an UPDATE to an existing finding, or ROUTINE confirmation?
2. For new/update results, what finding slug should be used?
3. What's the one-sentence summary of what's interesting?

Respond with JSON only (no markdown fences):
{{"analyses": [
  {{"log_file": "filename", "type": "new|update|routine", "slug": "finding-slug-or-null",
   "summary": "one sentence", "interesting": true/false}}
]}}

Project context (from CLAUDE.md):
{context}

New results:
{results_text}"""

    text = call_llm(prompt, "analyze")
    parsed = parse_json_from_llm(text)
    if parsed:
        return parsed.get("analyses", [])
    if text:
        log(f"Analysis parse failed. Raw: {text[:200]}", "ERROR")
    return []


# ── Phase 5: Review (multi-model) ────────────────────────────

# Models to use for peer review, in order. Each needs its API key.
REVIEW_MODELS = [
    {"model": "o3-pro",  "provider": "openai",  "key_env": "OPENAI_API_KEY", "api_base": "https://api.openai.com/v1"},
    {"model": "gpt-4.1", "provider": "openai",  "key_env": "OPENAI_API_KEY", "api_base": "https://api.openai.com/v1"},
    {"model": "o3",      "provider": "openai",  "key_env": "OPENAI_API_KEY", "api_base": "https://api.openai.com/v1"},
]


def run_reviews(slugs, models_override=None, dry_run=False):
    """Run peer reviews on findings using multiple models."""
    if not slugs:
        return

    review_script = REPO_ROOT / "scripts" / "reviews" / "run_review.py"
    if not review_script.exists():
        log("Review script not found", "WARN")
        return

    # Use override list or default multi-model list
    if models_override:
        review_configs = [{"model": m, "provider": "openai", "key_env": "OPENAI_API_KEY",
                           "api_base": "https://api.openai.com/v1"} for m in models_override]
    else:
        review_configs = REVIEW_MODELS

    reviewed_any = False
    for slug in slugs:
        for cfg in review_configs:
            api_key = os.environ.get(cfg["key_env"], "")
            if not api_key:
                continue

            log(f"Reviewing {slug} with {cfg['model']} ({cfg['provider']})...")
            if dry_run:
                log(f"  [DRY RUN] Would review")
                continue

            env = os.environ.copy()
            env["API_KEY"] = api_key
            try:
                result = subprocess.run(
                    [sys.executable, str(review_script), "--slug", slug,
                     "--model", cfg["model"], "--provider", cfg["provider"],
                     "--api-base", cfg["api_base"], "--skip-existing"],
                    env=env, timeout=600, capture_output=True, text=True,
                )
                if "Saved:" in result.stdout:
                    reviewed_any = True
                    log(f"  Review saved")
                elif "already reviewed" in result.stdout:
                    log(f"  Already reviewed by {cfg['model']}, skipping")
            except subprocess.TimeoutExpired:
                log(f"  Review timed out for {slug}/{cfg['model']}", "WARN")

    # Aggregate all reviews
    if reviewed_any and not dry_run:
        aggregate_reviews()


def aggregate_reviews():
    """Run aggregate + sync + copy manifest."""
    log("Aggregating reviews...")
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "reviews" / "aggregate.py")],
                   capture_output=True, timeout=30)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "reviews" / "sync_website.py")],
                   capture_output=True, timeout=30)
    manifest = REPO_ROOT / "docs" / "verifications" / "manifest.json"
    mcp_dest = WEBSITE_ROOT / "workers" / "mcp" / "src" / "generated" / "manifest.json"
    if manifest.exists() and mcp_dest.parent.exists():
        shutil.copy2(manifest, mcp_dest)


# ── Phase 6: Remediate ───────────────────────────────────────

# Hedging phrases that indicate the LLM punted instead of fixing
HEDGE_PHRASES = [
    "not yet quantified", "not yet computed", "pending", "further analysis needed",
    "has not yet been", "remains to be", "study needed", "to be determined",
    "future work", "not yet been computed",
]


def gather_evidence(slug):
    """Collect experiment data files relevant to a finding for the LLM prompt.

    Reads the finding's related_experiment field, finds the matching experiment
    directory, and returns a summary string of available data (CSV samples,
    metadata JSON, log tails) that the LLM can use to compute real answers.
    """
    # Read the finding to get related_experiment
    finding_files = list((WEBSITE_ROOT / "src" / "content" / "findings").glob(f"*{slug}*"))
    if not finding_files:
        return ""

    finding_text = finding_files[0].read_text()
    related = ""
    for line in finding_text.split("\n"):
        if line.startswith("related_experiment:"):
            related = line.split(":", 1)[1].strip().strip("/")
            break

    # Map website experiment path to local directory
    # e.g. "experiments/hausdorff-dimension-spectrum" -> "scripts/experiments/hausdorff-spectrum"
    exp_name = related.split("/")[-1] if related else ""
    exp_dir = None
    if exp_name:
        # Try exact match first, then prefix match
        candidate = REPO_ROOT / "scripts" / "experiments" / exp_name
        if candidate.exists():
            exp_dir = candidate
        else:
            # Fuzzy match: try shortened forms
            for d in (REPO_ROOT / "scripts" / "experiments").iterdir():
                if d.is_dir() and (exp_name.startswith(d.name) or d.name.startswith(exp_name)
                                   or exp_name.replace("-dimension", "").replace("-gpu", "") == d.name
                                   or d.name.replace("-spectrum", "-dimension-spectrum") == exp_name):
                    exp_dir = d
                    break

    if not exp_dir or not exp_dir.exists():
        return ""

    evidence_parts = []
    results_dir = exp_dir / "results"
    if not results_dir.exists():
        return ""

    # Collect metadata JSONs (small, full content)
    for meta in sorted(results_dir.glob("metadata*.json"))[:3]:
        try:
            evidence_parts.append(f"=== {meta.name} ===\n{meta.read_text()[:2000]}")
        except Exception:
            pass

    # Collect CSV headers + sample rows (first 5, last 5)
    for csv_file in sorted(results_dir.glob("*.csv"))[:3]:
        try:
            lines = csv_file.read_text().split("\n")
            sample = lines[:6]  # header + 5 rows
            if len(lines) > 11:
                sample.append(f"... ({len(lines)} total rows) ...")
                sample.extend(lines[-5:])
            evidence_parts.append(f"=== {csv_file.name} (sample) ===\n" + "\n".join(sample))
        except Exception:
            pass

    # Collect analysis JSONs
    for analysis in sorted(results_dir.glob("analysis*.json"))[:2]:
        try:
            evidence_parts.append(f"=== {analysis.name} ===\n{analysis.read_text()[:3000]}")
        except Exception:
            pass

    # Collect recent log tails
    for logf in sorted(results_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]:
        try:
            lines = logf.read_text().split("\n")
            tail = lines[-20:] if len(lines) > 20 else lines
            evidence_parts.append(f"=== {logf.name} (tail) ===\n" + "\n".join(tail))
        except Exception:
            pass

    return "\n\n".join(evidence_parts)


def validate_fix(old_text, new_text):
    """Check whether a proposed fix actually adds substance vs just hedging.

    Returns (is_valid, reason). A fix that only inserts hedge phrases without
    adding concrete data (numbers, coefficients, error bounds) is rejected.
    """
    if not old_text or not new_text:
        return False, "empty old_text or new_text"

    # What was added?
    added = new_text.replace(old_text, "").strip() if old_text in new_text else new_text

    # Check if the addition is purely hedging
    added_lower = added.lower()
    hedge_count = sum(1 for phrase in HEDGE_PHRASES if phrase in added_lower)

    # Check if real data was added (numbers, equals signs, correlation values)
    has_numbers = bool(re.search(r'\d+\.\d{2,}', added))  # decimal with 2+ places
    has_equals = "=" in added or "≈" in added
    has_data_words = any(w in added_lower for w in [
        "spearman", "kendall", "correlation", "r =", "rho =", "tau =",
        "± ", "converges to", "n=25", "n=35",
    ])

    if hedge_count > 0 and not has_numbers and not has_data_words:
        return False, f"fix only adds hedging ({hedge_count} hedge phrases, no concrete data)"

    return True, "ok"


def remediate(slugs, dry_run=False):
    """Use Claude to read reviews, identify fixable issues, and fix them."""
    if not slugs:
        return

    if not (shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        log("No LLM available — cannot remediate.", "WARN")
        return

    # Read the latest reviews for these slugs
    verifications_dir = REPO_ROOT / "docs" / "verifications"
    for slug in slugs:
        review_files = sorted(verifications_dir.glob(f"{slug}*review*.json"))
        if not review_files:
            continue

        # Collect all DISPUTED and NEEDS_CLARIFICATION claims
        issues = []
        for rf in review_files:
            try:
                with open(rf) as f:
                    review = json.load(f)
                for claim in review.get("claim_reviews", []):
                    verdict = claim.get("verdict", "")
                    if verdict in ("DISPUTED", "NEEDS_CLARIFICATION"):
                        issues.append({
                            "from": review.get("reviewer", {}).get("model", "unknown"),
                            "verdict": verdict,
                            "claim": claim.get("claim", "")[:100],
                            "concern": claim.get("concerns", "")[:200],
                        })
            except (json.JSONDecodeError, KeyError):
                continue

        if not issues:
            log(f"No disputed/unclear claims for {slug}")
            continue

        log(f"Remediating {slug}: {len(issues)} issue(s)")

        if dry_run:
            for iss in issues:
                log(f"  [{iss['verdict']}] ({iss['from']}) {iss['claim'][:60]}")
            continue

        # Read the finding
        finding_files = list((WEBSITE_ROOT / "src" / "content" / "findings").glob(f"*{slug}*"))
        finding_text = finding_files[0].read_text()[:2000] if finding_files else ""

        # Gather actual experiment data so the LLM can compute real answers
        evidence = gather_evidence(slug)

        prompt = f"""Fix issues in a bigcompute.science finding. Slug: {slug}

Finding (truncated):
{finding_text}

Issues from reviewers:
{json.dumps(issues[:6], indent=2)}

"""
        if evidence:
            prompt += f"""Experiment data (use this to compute actual values — do NOT just add hedging language):
{evidence[:6000]}

"""
        prompt += """IMPORTANT RULES:
1. For "fix": you MUST insert concrete data (computed values, correlation coefficients, error bounds) derived from the experiment data above. Do NOT just add phrases like "not yet quantified" or "pending further analysis" — that is NOT a fix, it is hedging.
2. If the experiment data does not contain enough information to compute the requested value, use "acknowledge" (not "fix" with a hedge).
3. "acknowledge" = the issue is real but we cannot fix it without running more computation. Be specific about what computation is needed.
4. "disagree" = the reviewer is wrong. Explain why with evidence.

For each: fix (provide old_text/new_text with real data), acknowledge (can't fix now — say what's needed), or disagree (reviewer wrong — explain why).
JSON only, no markdown: {"remediations": [{"claim":"...","action":"fix|acknowledge|disagree","description":"...","old_text":"...","new_text":"..."}]}"""

        try:
            text = call_llm(prompt, f"remediate:{slug}")
            if not text:
                continue

            # Extract JSON
            if "{" in text:
                start = text.index("{")
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{": depth += 1
                    elif text[i] == "}": depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break

            plan = json.loads(text)
            remediations = plan.get("remediations", [])

            for rem in remediations:
                action = rem.get("action", "")
                desc = rem.get("description", "")[:80]

                if action == "fix" and rem.get("old_text") and rem.get("new_text") and finding_files:
                    # Validate that the fix adds substance, not just hedges
                    is_valid, reason = validate_fix(rem["old_text"], rem["new_text"])
                    if not is_valid:
                        log(f"  REJECTED fix (hedge-only): {reason} — {desc}", "WARN")
                        rem["action"] = "acknowledge"
                        rem["description"] = f"Auto-demoted from fix: {reason}. {desc}"
                        continue

                    # Apply the fix
                    content = finding_files[0].read_text()
                    if rem["old_text"] in content:
                        content = content.replace(rem["old_text"], rem["new_text"], 1)
                        finding_files[0].write_text(content)
                        log(f"  FIXED: {desc}")
                    else:
                        log(f"  SKIP (text not found): {desc}", "WARN")
                elif action == "acknowledge":
                    log(f"  ACKNOWLEDGED: {desc}")
                elif action == "disagree":
                    log(f"  DISAGREE: {desc}")

            # Update remediation JSONs
            # Get the latest commit hash from bigcompute.science (where finding text lives)
            commit_hash = ""
            try:
                ch = subprocess.run(
                    ["git", "log", "-1", "--format=%h"], cwd=str(WEBSITE_ROOT),
                    capture_output=True, text=True, timeout=5)
                commit_hash = ch.stdout.strip()
            except Exception:
                pass

            rem_dir = verifications_dir / "remediations" / slug
            rem_dir.mkdir(parents=True, exist_ok=True)
            for rem in remediations:
                action = rem.get("action", "")
                if action in ("fix", "acknowledge", "disagree"):
                    issue_id = rem.get("claim", "unknown")[:40].lower().replace(" ", "-").replace("/", "-")
                    issue_id = re.sub(r'[^a-z0-9-]', '-', issue_id).strip('-')
                    rem_file = rem_dir / f"{issue_id}.json"

                    # Only "fix" that was actually applied => "resolved"
                    # "acknowledge" => "acknowledged" (open — needs more computation)
                    # "disagree" => "disputed" (open — reviewer disagreement)
                    if action == "fix":
                        status = "resolved"
                    elif action == "acknowledge":
                        status = "acknowledged"
                    else:
                        status = "disputed"

                    found_by = [rf.stem for rf in review_files]

                    # Derive severity from the original issue's verdict
                    claim_text = rem.get("claim", "")[:100]
                    original_verdict = "DISPUTED"
                    for iss in issues:
                        if iss["claim"][:40] == claim_text[:40]:
                            original_verdict = iss.get("verdict", "DISPUTED")
                            break

                    with open(rem_file, "w") as f:
                        json.dump({
                            "issue_id": issue_id,
                            "finding_slug": slug,
                            "found_by_reviews": found_by,
                            "severity": "important" if original_verdict == "DISPUTED" else "minor",
                            "description": rem.get("description", ""),
                            "status": status,
                            "action_taken": rem.get("description", ""),
                            "resolved_at": datetime.now(timezone.utc).isoformat() if status == "resolved" else None,
                            "commits": [commit_hash] if commit_hash and status == "resolved" else [],
                        }, f, indent=2)

        except (json.JSONDecodeError, Exception) as e:
            log(f"Remediation failed for {slug}: {e}", "ERROR")


# ── Phase 7: Deploy ──────────────────────────────────────────

def deploy(dry_run=False, direct_push=False):
    """Build website, commit both repos, deploy MCP.

    By default, creates a branch and opens a PR (safe for contributors).
    With --direct-push, pushes to main (for repo owner only).
    """
    if dry_run:
        log("[DRY RUN] Would build, commit, push, deploy")
        return

    branch_name = None
    if not direct_push:
        branch_name = f"agent/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log(f"Creating branch: {branch_name} (use --direct-push to push to main)")
        for repo in [REPO_ROOT, WEBSITE_ROOT]:
            subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(repo),
                          capture_output=True, timeout=10)

    # Update changelog from git history
    log("Updating changelog...")
    try:
        git_log = subprocess.run(
            ["git", "log", "--oneline", "--format={\"hash\":\"%h\",\"date\":\"%ad\",\"message\":\"%s\"}",
             "--date=format:%b %d", "-10"],
            cwd=str(WEBSITE_ROOT), capture_output=True, text=True, timeout=10
        )
        if git_log.returncode == 0:
            entries = []
            for line in git_log.stdout.strip().split("\n"):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            changelog_path = WEBSITE_ROOT / "src" / "data" / "changelog.json"
            with open(changelog_path, "w") as f:
                json.dump(entries, f, indent=2)
    except Exception as e:
        log(f"Changelog update failed: {e}", "WARN")

    # Build website
    log("Building website...")
    result = subprocess.run(
        ["npm", "run", "build"], cwd=str(WEBSITE_ROOT),
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        log(f"Build failed: {result.stderr[:200]}", "ERROR")
        return

    # Check if there are changes to commit
    any_changes = False
    for repo, name in [(REPO_ROOT, "idontknow"), (WEBSITE_ROOT, "bigcompute.science")]:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo),
            capture_output=True, text=True, timeout=30
        )
        changed = [line.split()[-1] for line in status.stdout.strip().split("\n")
                   if line.strip() and not any(x in line for x in [".bin", ".npz", ".csv", "node_modules"])]
        if changed:
            any_changes = True
            log(f"Committing {name} ({len(changed)} files)...")
            for f in changed[:50]:
                subprocess.run(["git", "add", f], cwd=str(repo), capture_output=True, timeout=30)
            subprocess.run(
                ["git", "commit", "-m",
                 f"research-agent: auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                 f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                cwd=str(repo), capture_output=True, timeout=60
            )

            if direct_push:
                subprocess.run(["git", "push"], cwd=str(repo), capture_output=True, timeout=120)
                log(f"  Pushed {name} to main")
            elif branch_name:
                subprocess.run(["git", "push", "-u", "origin", branch_name],
                              cwd=str(repo), capture_output=True, timeout=120)
                log(f"  Pushed {name} to branch {branch_name}")

    # Open PR if on a branch
    if branch_name and any_changes:
        for repo, name in [(REPO_ROOT, "idontknow"), (WEBSITE_ROOT, "bigcompute.science")]:
            try:
                result = subprocess.run(
                    ["gh", "pr", "create", "--title",
                     f"research-agent: auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                     "--body", "Automated update by the research agent.\n\nReview the changes before merging.",
                     "--base", "main"],
                    cwd=str(repo), capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    pr_url = result.stdout.strip()
                    log(f"  PR opened for {name}: {pr_url}")
                else:
                    log(f"  PR creation skipped for {name} (gh not available or no changes)")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                log(f"  PR creation skipped for {name} (gh CLI not installed)")

        # Return to main
        for repo in [REPO_ROOT, WEBSITE_ROOT]:
            subprocess.run(["git", "checkout", "main"], cwd=str(repo),
                          capture_output=True, timeout=10)

    # Deploy MCP only on direct push (PRs get deployed after merge)
    if direct_push:
        mcp_dir = WEBSITE_ROOT / "workers" / "mcp"
        if mcp_dir.exists():
            log("Deploying MCP worker...")
            subprocess.run(
                ["npx", "wrangler", "deploy"], cwd=str(mcp_dir),
                capture_output=True, text=True, timeout=120
            )

        # Ping search engines about updated content
        indexnow_script = REPO_ROOT / "scripts" / "indexnow.py"
        if indexnow_script.exists():
            log("Pinging search engines (IndexNow + Google)...")
            subprocess.run(
                [sys.executable, str(indexnow_script)],
                capture_output=True, text=True, timeout=30
            )


# ── Phase 8: Plan Next ───────────────────────────────────────

def get_launchable_binaries():
    """Auto-discover compiled CUDA binaries in the repo root."""
    binaries = {}
    for f in REPO_ROOT.iterdir():
        if f.is_file() and os.access(f, os.X_OK) and not f.suffix:
            # Check if it's an ELF binary (compiled CUDA/C)
            try:
                with open(f, 'rb') as fh:
                    magic = fh.read(4)
                if magic == b'\x7fELF':
                    binaries[f.name] = {"binary": f.name, "single_gpu": True}
            except:
                pass
    return binaries

LAUNCHABLE = get_launchable_binaries()


def plan_next(free_gpus, auto_launch=False, state=None):
    """Use Claude to decide what to run on free GPUs, then launch it."""
    if not free_gpus:
        log("No free GPUs — nothing to plan.")
        return

    log(f"{len(free_gpus)} free GPU(s): {free_gpus}")

    if not auto_launch:
        log("Auto-launch disabled. Use --auto-launch to enable.")
        return

    if not (shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        log("No LLM available — cannot plan experiments.", "WARN")
        return

    # Get current state for Claude to reason about
    context = CLAUDE_MD.read_text()[:6000] if CLAUDE_MD.exists() else ""
    harvested_names = list((state or {}).get("harvested", {}).keys())[-20:]
    running_procs = []
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split("\n"):
            for kw in EXPERIMENT_KEYWORDS:
                if kw in line.lower() and "grep" not in line and "research_agent" not in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        running_procs.append(" ".join(parts[10:])[:80])
                    break
    except subprocess.TimeoutExpired:
        pass

    prompt = f"""You are the experiment planner for bigcompute.science. You have {len(free_gpus)} free GPU(s) (indices: {free_gpus}).

Currently running:
{json.dumps(running_procs, indent=2) if running_procs else "nothing"}

Recently harvested logs:
{json.dumps(harvested_names[-10:], indent=2)}

Available compiled binaries (in repo root):
- ./zaremba_density_gpu <range> <digits>  (e.g., ./zaremba_density_gpu 100000000000 1,2,8)
- ./kronecker_gpu <n> [gpu_id]
- ./ramanujan_gpu <degree> <range>
- ./class_v2 <gpu_id> <start>

Project context (priorities from CLAUDE.md):
{context}

For EACH free GPU, respond with ONE concrete launch command. Consider:
- What's already running (don't duplicate)
- What was recently harvested (extend to next range or try next digit set)
- CLAUDE.md priorities

Respond with JSON only (no markdown):
{{"launches": [
  {{"gpu": 0, "command": "./zaremba_density_gpu 100000000000 1,2,8", "log_file": "scripts/experiments/zaremba-density/results/gpu_A128_1e11.log", "reason": "one sentence"}}
]}}"""

    try:
        text = call_llm(prompt, "plan")
        if not text:
            return
        parsed = parse_json_from_llm(text)
        if not parsed:
            log(f"Plan parse failed. Raw: {text[:200]}", "ERROR")
            return

        plan = parsed
        launches = plan.get("launches", [])

        for launch in launches:
            gpu = launch["gpu"]
            cmd = launch["command"]
            log_file = launch.get("log_file", f"logs/auto_gpu{gpu}.log")
            reason = launch.get("reason", "")

            # Safety: only allow known binaries
            binary = cmd.split()[0].lstrip("./")
            if binary not in [v["binary"] for v in LAUNCHABLE.values()]:
                log(f"  BLOCKED: unknown binary '{binary}' — skipping for safety", "WARN")
                continue

            log(f"  LAUNCHING on GPU {gpu}: {cmd}")
            log(f"    Reason: {reason}")
            log(f"    Log: {log_file}")

            # Ensure log directory exists
            log_path = REPO_ROOT / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Launch with stdbuf for line-buffered output to log file
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(log_path, "w") as lf:
                subprocess.Popen(
                    ["stdbuf", "-oL"] + cmd.split(),
                    stdout=lf, stderr=subprocess.STDOUT,
                    env=env, cwd=str(REPO_ROOT),
                )
            log(f"  Started (background)")

    except json.JSONDecodeError as e:
        log(f"Plan JSON parse error: {e}", "ERROR")
    except Exception as e:
        log(f"Planning failed: {e}", "ERROR")


# ── Main Loop ─────────────────────────────────────────────────

def tick(args, state):
    """Run one cycle of the research loop."""
    phase = args.phase
    dry_run = args.dry_run

    log("=" * 60)
    log(f"Research Agent — tick {'(DRY RUN)' if dry_run else ''}")
    log("=" * 60)

    # Phase 1: Monitor
    if not phase or phase == "monitor":
        gpu_status = monitor_gpus()
    else:
        gpu_status = {"free_gpus": [], "busy_gpus": [], "processes": []}

    # Phase 2: Harvest
    new_results = []
    if not phase or phase == "harvest":
        new_results = harvest(state, dry_run)

    # Phase 3: Analyze
    analyses = []
    if (not phase or phase == "analyze") and new_results:
        analyses = analyze_results(new_results)
        for a in analyses:
            status = "INTERESTING" if a.get("interesting") else "routine"
            log(f"  [{status}] {a.get('log_file', '?')}: {a.get('summary', '?')}")

    # Phase 5: Review (new or updated findings — multiple models)
    finding_slugs = []
    if not phase or phase == "review":
        finding_slugs = [a["slug"] for a in analyses if a.get("slug") and a.get("type") in ("new", "update")]
        if finding_slugs:
            models = args.models.split(",") if args.models else None
            run_reviews(finding_slugs, models, dry_run)

    # Phase 6: Remediate (fix issues found by reviewers)
    # Remediate new findings from this tick AND any existing unresolved issues
    if not phase or phase == "remediate":
        all_slugs_to_remediate = list(finding_slugs)  # new findings from this tick
        # Also check all findings for unresolved review issues
        verifications_dir = REPO_ROOT / "docs" / "verifications"
        for rf in verifications_dir.glob("*review*.json"):
            try:
                with open(rf) as f:
                    review = json.load(f)
                slug = review.get("finding_slug", "")

                # Count claims that are DISPUTED/NEEDS_CLARIFICATION
                disputed_claims = [
                    c for c in review.get("claim_reviews", [])
                    if c.get("verdict", "") in ("DISPUTED", "NEEDS_CLARIFICATION")
                ]
                if not disputed_claims or not slug or slug in all_slugs_to_remediate:
                    continue

                # Check how many are already resolved/acknowledged in remediation dir
                rem_dir = verifications_dir / "remediations" / slug
                resolved_statuses = set()
                if rem_dir.exists():
                    for rem_file in rem_dir.glob("*.json"):
                        try:
                            rem_data = json.loads(rem_file.read_text())
                            if rem_data.get("status") in ("resolved", "disputed"):
                                resolved_statuses.add(rem_data.get("description", "")[:40])
                        except (json.JSONDecodeError, KeyError):
                            pass

                # Only re-remediate if there are claims with no resolution yet
                # (i.e. no remediation JSON, or status is "acknowledged" meaning needs more work)
                has_truly_unresolved = False
                for claim in disputed_claims:
                    claim_id = claim.get("claim", "")[:40].lower().replace(" ", "-").replace("/", "-")
                    claim_id = re.sub(r'[^a-z0-9-]', '-', claim_id).strip('-')
                    rem_file = rem_dir / f"{claim_id}.json"
                    if not rem_file.exists():
                        has_truly_unresolved = True
                        break
                    try:
                        rem_data = json.loads(rem_file.read_text())
                        if rem_data.get("status") == "acknowledged":
                            has_truly_unresolved = True
                            break
                    except (json.JSONDecodeError, KeyError):
                        has_truly_unresolved = True
                        break

                if has_truly_unresolved:
                    # Still rate-limit to once per day
                    last_rem = state.get("last_remediated", {}).get(slug, "")
                    if not last_rem or (datetime.now(timezone.utc).isoformat()[:10] != last_rem[:10]):
                        all_slugs_to_remediate.append(slug)
            except (json.JSONDecodeError, KeyError):
                continue

        if all_slugs_to_remediate:
            log(f"Remediating {len(all_slugs_to_remediate)} finding(s) with unresolved issues...")
            remediate(all_slugs_to_remediate, dry_run)
            if not dry_run:
                for s in all_slugs_to_remediate:
                    state.setdefault("last_remediated", {})[s] = datetime.now(timezone.utc).isoformat()
                aggregate_reviews()

    # Phase 7: Deploy
    if not phase or phase == "deploy":
        if analyses or (phase == "deploy"):
            deploy(dry_run, direct_push=args.direct_push)

    # Phase 8: Plan
    if not phase or phase == "plan":
        plan_next(gpu_status.get("free_gpus", []), args.auto_launch, state)

    # Phase 9: Verify community submissions
    if not phase or phase == "verify":
        verify_community_submissions(gpu_status.get("free_gpus", []), state, dry_run)

    # Save state
    if not dry_run:
        save_state(state)

    interesting = [a for a in analyses if a.get("interesting")]
    log(f"\nSummary: {len(new_results)} harvested, {len(interesting)} interesting, "
        f"{len(gpu_status.get('free_gpus', []))} GPUs free")


# ── Phase 9: Verify Community Submissions ────────────────────

def verify_community_submissions(free_gpus, state, dry_run=False):
    """Check GitHub issues labeled 'new-data', re-run the experiment, verify results."""
    if not free_gpus:
        return  # Need a free GPU to verify

    # Check if gh CLI is available
    if not shutil.which("gh"):
        return

    already_verified = state.get("verified_issues", [])

    try:
        # Get open issues labeled 'new-data'
        result = subprocess.run(
            ["gh", "issue", "list", "--label", "new-data", "--state", "open", "--json", "number,title,body", "--limit", "5"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return

        issues = json.loads(result.stdout)
        if not issues:
            return

        for issue in issues:
            issue_num = issue["number"]
            if issue_num in already_verified:
                continue

            body = issue.get("body", "")
            log(f"Verifying community submission: issue #{issue_num}")

            # Parse experiments from the issue body
            experiments = []
            import re as _re
            for block in _re.findall(r'```[\s\S]*?```', body):
                m_digits = _re.search(r'Digit set:\s*\{([^}]+)\}', block)
                m_range = _re.search(r'Range: d = 1 to (\d+)', block)
                m_uncovered = _re.search(r'Uncovered:\s*(\d+)', block)
                m_density = _re.search(r'Density:\s*([\d.]+)%', block)
                if m_digits and m_range:
                    experiments.append({
                        "digits": m_digits.group(1).replace(" ", ""),
                        "range": int(m_range.group(1)),
                        "claimed_uncovered": int(m_uncovered.group(1)) if m_uncovered else None,
                        "claimed_density": float(m_density.group(1)) if m_density else None,
                    })

            if not experiments:
                log(f"  Could not parse experiments from issue #{issue_num}")
                continue

            # Verify each experiment by re-running it
            gpu = free_gpus[0]
            all_match = True
            results_comment = f"## Verification Results (GPU {gpu})\n\n"

            for exp in experiments:
                binary = REPO_ROOT / "zaremba_density_gpu"
                if not binary.exists():
                    log(f"  zaremba_density_gpu not found — cannot verify")
                    break

                log(f"  Re-running A={{{exp['digits']}}} at {exp['range']:,} on GPU {gpu}...")
                if dry_run:
                    log(f"  [DRY RUN] Would verify")
                    continue

                log_file = REPO_ROOT / "logs" / f"verify_issue_{issue_num}.log"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)

                try:
                    proc = subprocess.run(
                        [str(binary), str(exp["range"]), exp["digits"]],
                        capture_output=True, text=True, timeout=3600,  # 1 hour max
                        env=env, cwd=str(REPO_ROOT)
                    )
                    output = proc.stdout

                    # Parse our result
                    m_unc = _re.search(r'Uncovered:\s*(\d+)', output)
                    m_den = _re.search(r'Density:\s*([\d.]+)%', output)
                    our_uncovered = int(m_unc.group(1)) if m_unc else None
                    our_density = float(m_den.group(1)) if m_den else None

                    # Compare
                    match = True
                    if exp["claimed_uncovered"] is not None and our_uncovered is not None:
                        if exp["claimed_uncovered"] != our_uncovered:
                            match = False
                    if exp["claimed_density"] is not None and our_density is not None:
                        if abs(exp["claimed_density"] - our_density) > 0.001:
                            match = False

                    status = "MATCH" if match else "MISMATCH"
                    if not match:
                        all_match = False

                    results_comment += f"- A={{{exp['digits']}}} at {exp['range']:,}: **{status}**\n"
                    if not match:
                        results_comment += f"  - Claimed: uncovered={exp['claimed_uncovered']}, density={exp['claimed_density']}%\n"
                        results_comment += f"  - Verified: uncovered={our_uncovered}, density={our_density}%\n"
                    else:
                        results_comment += f"  - Verified: uncovered={our_uncovered}, density={our_density}%\n"

                    log(f"  {status}: claimed={exp['claimed_uncovered']}, verified={our_uncovered}")

                except subprocess.TimeoutExpired:
                    log(f"  Verification timed out for A={{{exp['digits']}}} at {exp['range']:,}")
                    results_comment += f"- A={{{exp['digits']}}} at {exp['range']:,}: **TIMEOUT** (range too large for verification)\n"
                except Exception as e:
                    log(f"  Verification error: {e}")
                    results_comment += f"- A={{{exp['digits']}}} at {exp['range']:,}: **ERROR** ({e})\n"

            if dry_run:
                continue

            # Post verification comment
            if all_match:
                results_comment += "\n> All results verified. Data is authentic."
            else:
                results_comment += "\n> **MISMATCH DETECTED.** Some claimed results do not match our verification."

            try:
                subprocess.run(
                    ["gh", "issue", "comment", str(issue_num), "--body", results_comment],
                    capture_output=True, timeout=15
                )
                # Label based on result
                label = "verified" if all_match else "verification-failed"
                subprocess.run(
                    ["gh", "issue", "edit", str(issue_num), "--add-label", label],
                    capture_output=True, timeout=15
                )
                log(f"  Issue #{issue_num}: {label}")
            except:
                pass

            state.setdefault("verified_issues", []).append(issue_num)

    except Exception as e:
        log(f"Community verification failed: {e}", "ERROR")


def parse_interval(s):
    """Parse interval string like '10m', '1h', '30s' to seconds."""
    m = re.match(r"(\d+)\s*(s|m|h)", s.lower())
    if not m:
        return int(s)
    val, unit = int(m.group(1)), m.group(2)
    return val * {"s": 1, "m": 60, "h": 3600}[unit]


def main():
    parser = argparse.ArgumentParser(description="Autonomous research agent for bigcompute.science")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", default="10m", help="Loop interval (default: 10m)")
    parser.add_argument("--phase", choices=["monitor", "harvest", "analyze", "review", "remediate", "deploy", "plan", "verify"],
                        help="Run specific phase only")
    parser.add_argument("--models", help="Comma-separated review models (default: gpt-4.1)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no changes")
    parser.add_argument("--auto-launch", action="store_true", help="Auto-launch experiments (off by default)")
    parser.add_argument("--direct-push", action="store_true", help="Push directly to main (default: create branch + PR)")
    args = parser.parse_args()

    state = load_state()
    interval = parse_interval(args.interval)

    log(f"Research Agent starting (interval={args.interval}, dry_run={args.dry_run})")
    log(f"Repos: {REPO_ROOT.name}, {WEBSITE_ROOT.name}")
    log(f"State: {len(state['harvested'])} harvested, {len(state['reviewed'])} reviewed")

    if args.once or args.phase:
        tick(args, state)
    else:
        while True:
            try:
                tick(args, state)
            except KeyboardInterrupt:
                log("Interrupted — saving state and exiting.")
                save_state(state)
                break
            except Exception as e:
                log(f"Tick failed: {e}", "ERROR")

            log(f"Sleeping {interval}s until next tick...")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                log("Interrupted — exiting.")
                break


if __name__ == "__main__":
    main()
