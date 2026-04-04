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

EXPERIMENT_KEYWORDS = [
    "zaremba", "kronecker", "class_num", "ramanujan", "char_table",
    "ramsey", "hausdorff", "lyapunov", "minkowski", "flint",
]


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


# ── Phase 3: Analyze (calls Claude API) ──────────────────────

def analyze_results(new_results):
    """Analyze new results using claude CLI (uses Claude Code premium account)."""
    # Try claude CLI first (uses premium account), fall back to API key
    claude_cli = shutil.which("claude")
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not claude_cli and not api_key:
        log("Neither 'claude' CLI nor ANTHROPIC_API_KEY available — skipping analysis.", "WARN")
        return []

    # Load CLAUDE.md for context
    context = ""
    if CLAUDE_MD.exists():
        context = CLAUDE_MD.read_text()[:8000]

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

    try:
        if claude_cli:
            log("Analyzing via claude CLI (premium account)...")
            result = subprocess.run(
                [claude_cli, "-p", prompt],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                log(f"claude CLI failed: {result.stderr[:200]}", "ERROR")
                return []
            text = result.stdout.strip()
        else:
            log("Analyzing via Anthropic API...")
            import httpx
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 2000,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60.0,
            )
            if resp.status_code != 200:
                log(f"Claude API error {resp.status_code}: {resp.text[:200]}", "ERROR")
                return []
            text = resp.json()["content"][0]["text"].strip()

        # Parse JSON — handle mixed prose + JSON output
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        elif "{" in text:
            # Extract first JSON object from mixed text
            start = text.index("{")
            # Find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break
        return json.loads(text).get("analyses", [])
    except json.JSONDecodeError as e:
        log(f"Analysis JSON parse error: {e}", "ERROR")
        log(f"Raw response: {text[:300]}", "ERROR")
        return []
    except Exception as e:
        log(f"Analysis failed: {e}", "ERROR")
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

def remediate(slugs, dry_run=False):
    """Use Claude to read reviews, identify fixable issues, and fix them."""
    if not slugs:
        return

    claude_cli = shutil.which("claude")
    if not claude_cli:
        log("claude CLI not found — cannot remediate.", "WARN")
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

        # Read the finding (trimmed to avoid prompt overload)
        finding_files = list((WEBSITE_ROOT / "src" / "content" / "findings").glob(f"*{slug}*"))
        finding_text = finding_files[0].read_text()[:2000] if finding_files else ""

        prompt = f"""Fix issues in a bigcompute.science finding. Slug: {slug}

Finding (truncated):
{finding_text}

Issues from reviewers:
{json.dumps(issues[:6], indent=2)}

For each: fix (provide old_text/new_text), acknowledge (can't fix now), or disagree (reviewer wrong).
JSON only, no markdown: {{"remediations": [{{"claim":"...","action":"fix|acknowledge|disagree","description":"...","old_text":"...","new_text":"..."}}]}}"""

        try:
            # Use httpx to call Claude API directly (faster than claude CLI)
            import httpx
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                resp = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": "claude-sonnet-4-20250514", "max_tokens": 4000,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=120.0,
                )
                if resp.status_code != 200:
                    log(f"Anthropic API error: {resp.text[:200]}", "ERROR")
                    continue
                text = resp.json()["content"][0]["text"].strip()
            elif claude_cli:
                result = subprocess.run(
                    [claude_cli, "-p", prompt],
                    capture_output=True, text=True, timeout=300,
                )
                text = result.stdout.strip()
            else:
                log("No API key or claude CLI for remediation", "WARN")
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
            except:
                pass

            rem_dir = verifications_dir / "remediations" / slug
            rem_dir.mkdir(parents=True, exist_ok=True)
            for rem in remediations:
                if rem.get("action") in ("fix", "acknowledge", "disagree"):
                    issue_id = rem.get("claim", "unknown")[:40].lower().replace(" ", "-").replace("/", "-")
                    issue_id = re.sub(r'[^a-z0-9-]', '-', issue_id).strip('-')
                    rem_file = rem_dir / f"{issue_id}.json"
                    with open(rem_file, "w") as f:
                        json.dump({
                            "issue_id": issue_id,
                            "finding_slug": slug,
                            "severity": "important" if rem.get("action") == "fix" else "minor",
                            "description": rem.get("description", ""),
                            "status": "resolved" if rem.get("action") == "fix" else rem.get("action", "acknowledged"),
                            "action_taken": f"Auto-fixed by research agent: {rem.get('description', '')}",
                            "resolved_at": datetime.now(timezone.utc).isoformat(),
                            "commits": [commit_hash] if commit_hash and rem.get("action") == "fix" else [],
                        }, f, indent=2)

        except (json.JSONDecodeError, Exception) as e:
            log(f"Remediation failed for {slug}: {e}", "ERROR")


# ── Phase 7: Deploy ──────────────────────────────────────────

def deploy(dry_run=False):
    """Build website, commit both repos, deploy MCP."""
    if dry_run:
        log("[DRY RUN] Would build, commit, push, deploy")
        return

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
    for repo, name in [(REPO_ROOT, "idontknow"), (WEBSITE_ROOT, "bigcompute.science")]:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo),
            capture_output=True, text=True, timeout=30
        )
        changed = [line.split()[-1] for line in status.stdout.strip().split("\n")
                   if line.strip() and not any(x in line for x in [".bin", ".npz", ".csv", "node_modules"])]
        if changed:
            log(f"Committing {name} ({len(changed)} files)...")
            # Add specific tracked files, not -A (avoids huge untracked data files)
            for f in changed[:50]:  # cap at 50 files
                subprocess.run(["git", "add", f], cwd=str(repo), capture_output=True, timeout=30)
            subprocess.run(
                ["git", "commit", "-m",
                 f"research-agent: auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                 f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                cwd=str(repo), capture_output=True, timeout=60
            )
            subprocess.run(["git", "push"], cwd=str(repo), capture_output=True, timeout=120)
            log(f"  Pushed {name}")

    # Deploy MCP
    mcp_dir = WEBSITE_ROOT / "workers" / "mcp"
    if mcp_dir.exists():
        log("Deploying MCP worker...")
        subprocess.run(
            ["npx", "wrangler", "deploy"], cwd=str(mcp_dir),
            capture_output=True, text=True, timeout=120
        )


# ── Phase 8: Plan Next ───────────────────────────────────────

# Known launchable experiments with concrete commands
LAUNCHABLE = {
    "zaremba_density": {
        "binary": "zaremba_density_gpu",
        "results_dir": "scripts/experiments/zaremba-density/results",
        "single_gpu": True,
    },
    "kronecker": {
        "binary": "kronecker_gpu",
        "results_dir": "scripts/experiments/kronecker-coefficients/results",
        "single_gpu": True,
    },
    "ramanujan": {
        "binary": "ramanujan_gpu",
        "results_dir": "scripts/experiments/ramanujan-machine/results",
        "single_gpu": True,
    },
    "class_numbers": {
        "binary": "class_v2",
        "results_dir": "data/class-numbers",
        "single_gpu": True,
    },
}


def plan_next(free_gpus, auto_launch=False, state=None):
    """Use Claude to decide what to run on free GPUs, then launch it."""
    if not free_gpus:
        log("No free GPUs — nothing to plan.")
        return

    log(f"{len(free_gpus)} free GPU(s): {free_gpus}")

    if not auto_launch:
        log("Auto-launch disabled. Use --auto-launch to enable.")
        return

    claude_cli = shutil.which("claude")
    if not claude_cli:
        log("claude CLI not found — cannot plan experiments.", "WARN")
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
        log("Asking Claude what to launch...")
        result = subprocess.run(
            [claude_cli, "-p", prompt],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log(f"claude CLI failed: {result.stderr[:200]}", "ERROR")
            return

        text = result.stdout.strip()
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

            # Launch
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(log_path, "w") as lf:
                subprocess.Popen(
                    cmd.split(),
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
                has_unresolved = any(
                    c.get("verdict", "") in ("DISPUTED", "NEEDS_CLARIFICATION")
                    for c in review.get("claim_reviews", [])
                )
                if has_unresolved and slug and slug not in all_slugs_to_remediate:
                    # Check if we already remediated this slug recently (within last 24h)
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
            deploy(dry_run)

    # Phase 8: Plan
    if not phase or phase == "plan":
        plan_next(gpu_status.get("free_gpus", []), args.auto_launch, state)

    # Save state
    if not dry_run:
        save_state(state)

    interesting = [a for a in analyses if a.get("interesting")]
    log(f"\nSummary: {len(new_results)} harvested, {len(interesting)} interesting, "
        f"{len(gpu_status.get('free_gpus', []))} GPUs free")


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
    parser.add_argument("--phase", choices=["monitor", "harvest", "analyze", "review", "remediate", "deploy", "plan"],
                        help="Run specific phase only")
    parser.add_argument("--models", help="Comma-separated review models (default: gpt-4.1)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no changes")
    parser.add_argument("--auto-launch", action="store_true", help="Auto-launch experiments (off by default)")
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
