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
    """Call Claude API to analyze new results and determine if they're findings."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log("ANTHROPIC_API_KEY not set — skipping analysis. Set it for full cycle.", "WARN")
        return []

    import httpx

    # Load CLAUDE.md for context
    context = ""
    if CLAUDE_MD.exists():
        context = CLAUDE_MD.read_text()[:8000]  # First 8K chars

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
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 2000,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60.0,
        )
        if resp.status_code != 200:
            log(f"Claude API error {resp.status_code}: {resp.text[:200]}", "ERROR")
            return []

        text = resp.json()["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text).get("analyses", [])
    except Exception as e:
        log(f"Analysis failed: {e}", "ERROR")
        return []


# ── Phase 5: Review ──────────────────────────────────────────

def run_reviews(slugs, models=None, dry_run=False):
    """Run peer reviews on findings using available API keys."""
    if not slugs:
        return

    review_script = REPO_ROOT / "scripts" / "reviews" / "run_review.py"
    if not review_script.exists():
        log("Review script not found", "WARN")
        return

    # Determine available models
    available = []
    if os.environ.get("OPENAI_API_KEY"):
        available.extend(models or ["gpt-4.1"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        pass  # Claude reviews handled differently (self-review less useful)

    if not available:
        log("No API keys set for reviews — skipping", "WARN")
        return

    for slug in slugs:
        for model in available:
            log(f"Reviewing {slug} with {model}...")
            if dry_run:
                log(f"  [DRY RUN] Would call run_review.py --slug {slug} --model {model}")
                continue

            env = os.environ.copy()
            env["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
            try:
                subprocess.run(
                    [sys.executable, str(review_script), "--slug", slug,
                     "--model", model, "--provider", "openai", "--skip-existing"],
                    env=env, timeout=600, capture_output=True, text=True,
                )
            except subprocess.TimeoutExpired:
                log(f"  Review timed out for {slug}/{model}", "WARN")

    # Aggregate
    if not dry_run:
        log("Aggregating reviews...")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "reviews" / "aggregate.py")],
                       capture_output=True, timeout=30)
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "reviews" / "sync_website.py")],
                       capture_output=True, timeout=30)
        # Copy manifest to MCP
        manifest = REPO_ROOT / "docs" / "verifications" / "manifest.json"
        mcp_dest = WEBSITE_ROOT / "workers" / "mcp" / "src" / "generated" / "manifest.json"
        if manifest.exists() and mcp_dest.parent.exists():
            import shutil
            shutil.copy2(manifest, mcp_dest)


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
            capture_output=True, text=True, timeout=10
        )
        if status.stdout.strip():
            log(f"Committing {name}...")
            subprocess.run(["git", "add", "-A"], cwd=str(repo), timeout=10)
            subprocess.run(
                ["git", "commit", "-m",
                 f"research-agent: auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                 f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
                cwd=str(repo), capture_output=True, timeout=30
            )
            subprocess.run(["git", "push"], cwd=str(repo), capture_output=True, timeout=60)
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

EXPERIMENT_PRIORITIES = [
    {
        "name": "Zaremba density confirmation",
        "description": "Extend {1,2,k} hierarchy or confirm exception counts at higher decades",
        "gpu_count": 1,
        "priority": 1,
    },
    {
        "name": "Kronecker int128 kernel",
        "description": "Full S_40 triple-sum (8.68T triples, needs int128 arithmetic)",
        "gpu_count": 8,
        "priority": 2,
    },
    {
        "name": "Ramanujan Machine PSLQ",
        "description": "High-precision PSLQ verification for degree 4+ candidates",
        "gpu_count": 1,
        "priority": 3,
    },
    {
        "name": "Class numbers [10^11, 10^12]",
        "description": "Extend Cohen-Lenstra data to next decade",
        "gpu_count": 8,
        "priority": 4,
    },
]


def plan_next(free_gpus, auto_launch=False):
    """Suggest or launch next experiment based on free GPUs."""
    if not free_gpus:
        log("No free GPUs — nothing to plan.")
        return

    log(f"{len(free_gpus)} free GPU(s): {free_gpus}")
    for exp in EXPERIMENT_PRIORITIES:
        if len(free_gpus) >= exp["gpu_count"]:
            log(f"SUGGESTED: {exp['name']} — {exp['description']}")
            if auto_launch:
                log("Auto-launch not yet implemented for safety. Use manual launch.", "WARN")
            return

    log("No suitable experiment for available GPU count.")


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

    # Phase 5: Review (new or updated findings)
    if not phase or phase == "review":
        finding_slugs = [a["slug"] for a in analyses if a.get("slug") and a.get("type") in ("new", "update")]
        if finding_slugs:
            models = args.models.split(",") if args.models else None
            run_reviews(finding_slugs, models, dry_run)

    # Phase 7: Deploy
    if not phase or phase == "deploy":
        if analyses or (phase == "deploy"):
            deploy(dry_run)

    # Phase 8: Plan
    if not phase or phase == "plan":
        plan_next(gpu_status.get("free_gpus", []), args.auto_launch)

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
    parser.add_argument("--phase", choices=["monitor", "harvest", "analyze", "review", "deploy", "plan"],
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
