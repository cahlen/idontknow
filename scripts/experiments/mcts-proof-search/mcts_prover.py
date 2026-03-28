"""
MCTS-enhanced Lean 4 theorem prover.

Builds on the basic prover by adding Monte Carlo Tree Search over the
space of proof tactics. Instead of sampling N independent proofs from the LLM,
we build a search tree where each node is a proof state and each edge is a
tactic suggestion.

This addresses the key bottleneck found in the Zaremba experiment: LLMs
nail proof structure but fail at witness search. MCTS explores systematically.

Architecture:
    - Root node: the initial sorry state
    - Children: tactic suggestions from the LLM (sampled with temperature)
    - Evaluation: Lean compiler check (binary: compiles or not)
    - Backpropagation: UCB1 scores guide exploration vs exploitation
    - Expansion: query LLM for new tactics at promising nodes

Usage:
    python mcts_prover.py \
        --server http://localhost:8000 \
        --file lean4-proving/conjectures/zaremba.lean \
        --budget 1000 \
        --rollouts 32
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class ProofNode:
    """A node in the proof search tree."""
    lean_code: str              # full Lean file at this state
    sorry_line: int             # which sorry we're trying to fill
    tactic: str                 # tactic that produced this node (empty for root)
    parent: Optional['ProofNode'] = None
    children: list['ProofNode'] = field(default_factory=list)
    visits: int = 0
    wins: int = 0               # 1 if this subtree contains a proof, 0 otherwise
    is_proved: bool = False
    lean_error: str = ""        # error from Lean if this node failed

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound for tree search."""
        if self.visits == 0:
            return float('inf')
        exploit = self.wins / self.visits
        explore = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploit + explore


def check_lean(lean_code: str, timeout: int = 120) -> tuple[bool, str]:
    """Run Lean 4 on the given code. Returns (success, output)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(lean_code)
        f.flush()
        try:
            result = subprocess.run(
                ["lean", f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout + result.stderr
            has_sorry = "declaration uses `sorry`" in output
            success = result.returncode == 0 and not has_sorry
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Lean timed out"
        finally:
            os.unlink(f.name)


def query_tactics(server_url: str, prompt: str, model: str = "default",
                  n: int = 8, temperature: float = 0.8) -> list[str]:
    """Get tactic suggestions from the LLM."""
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": temperature,
            "n": n,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return [c["message"]["content"].strip() for c in resp.json()["choices"]]


def clean_suggestion(text: str) -> str:
    """Extract the proof tactic from LLM output."""
    text = text.strip()
    text = re.sub(r"```\w*\n?", "", text).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    text = re.sub(r"^###?\s.*\n+", "", text).strip()

    if ":= by" in text:
        text = text.split(":= by", 1)[1].strip()

    for stop in ["\ntheorem ", "\nlemma ", "\n--", "\n/-", "\n###"]:
        if stop in text:
            text = text[:text.index(stop)].strip()

    if text.startswith("exact") and "⟩" in text:
        depth = 0
        for i, ch in enumerate(text):
            if ch == '⟨': depth += 1
            elif ch == '⟩':
                depth -= 1
                if depth == 0:
                    text = text[:i+1]
                    break

    return text


def build_prompt(context: str, error: Optional[str], tried: list[str]) -> str:
    """Build a prompt that includes previous failed attempts."""
    prompt = """You are an expert Lean 4 theorem prover. Replace `sorry` with a valid proof.

```lean4
""" + context + """
```

"""
    if tried:
        prompt += "These attempts FAILED — try something DIFFERENT:\n"
        for t in tried[-5:]:  # show last 5 failures
            prompt += f"  - {t[:60]}\n"
        prompt += "\n"

    if error:
        prompt += f"Last error: {error[:200]}\n\n"

    prompt += "Provide ONLY the proof tactic, nothing else:\n"
    return prompt


def select_node(root: ProofNode) -> ProofNode:
    """Select the most promising unexpanded node via UCB1."""
    node = root
    while node.children:
        # Pick child with highest UCB1
        node = max(node.children, key=lambda c: c.ucb1)
    return node


def expand(node: ProofNode, server_url: str, model: str,
           context: str, n_children: int = 8) -> list[ProofNode]:
    """Expand a node by querying the LLM for new tactics."""
    tried = [c.tactic for c in node.children]
    prompt = build_prompt(context, node.lean_error, tried)
    suggestions = query_tactics(server_url, prompt, model=model,
                                n=n_children, temperature=0.9)

    new_children = []
    for raw in suggestions:
        tactic = clean_suggestion(raw)
        if not tactic or tactic in tried or "sorry" in tactic:
            continue

        # Apply tactic to the Lean code
        lines = node.lean_code.split("\n")
        lines[node.sorry_line] = lines[node.sorry_line].replace("sorry", tactic, 1)
        new_code = "\n".join(lines)

        child = ProofNode(
            lean_code=new_code,
            sorry_line=node.sorry_line,
            tactic=tactic,
            parent=node,
        )
        node.children.append(child)
        new_children.append(child)

    return new_children


def evaluate(node: ProofNode) -> bool:
    """Check if the node's code compiles successfully."""
    success, output = check_lean(node.lean_code)
    node.is_proved = success
    if not success:
        node.lean_error = output[:500]
    return success


def backpropagate(node: ProofNode, won: bool):
    """Update visit counts and win counts up the tree."""
    while node is not None:
        node.visits += 1
        if won:
            node.wins += 1
        node = node.parent


def mcts_prove(lean_code: str, sorry_line: int, context: str,
               server_url: str, model: str = "default",
               budget: int = 100, rollouts_per_expand: int = 8) -> Optional[str]:
    """
    Run MCTS to find a proof for the sorry at the given line.

    Args:
        budget: total number of Lean evaluations to perform
        rollouts_per_expand: number of LLM suggestions per expansion

    Returns:
        The winning tactic, or None if no proof found.
    """
    root = ProofNode(lean_code=lean_code, sorry_line=sorry_line, tactic="")
    root.visits = 1

    evaluations = 0
    start = time.time()

    while evaluations < budget:
        # SELECT
        node = select_node(root)

        # EXPAND
        children = expand(node, server_url, model, context,
                          n_children=rollouts_per_expand)

        if not children:
            # No new tactics to try — mark this branch as exhausted
            backpropagate(node, False)
            continue

        # EVALUATE each new child
        for child in children:
            evaluations += 1
            won = evaluate(child)

            if won:
                elapsed = time.time() - start
                print(f"    PROVED in {evaluations} evals ({elapsed:.1f}s): {child.tactic[:80]}")
                return child.tactic

            # BACKPROPAGATE
            backpropagate(child, won)

            if evaluations >= budget:
                break

        if evaluations % 50 == 0:
            elapsed = time.time() - start
            unique = len(set(c.tactic for c in root.children))
            print(f"    [{evaluations}/{budget}] {unique} unique tactics tried ({elapsed:.1f}s)")

    return None


def main():
    parser = argparse.ArgumentParser(description="MCTS-enhanced Lean 4 prover")
    parser.add_argument("--server", action="append", default=[],
                        help="LLM server URL")
    parser.add_argument("--file", required=True, help="Lean 4 file")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--budget", type=int, default=200,
                        help="Max Lean evaluations per sorry")
    parser.add_argument("--rollouts", type=int, default=8,
                        help="LLM suggestions per expansion")
    parser.add_argument("--zaremba", action="store_true",
                        help="Use Zaremba-specialized prompts")
    args = parser.parse_args()

    if not args.server:
        args.server = ["http://localhost:8000"]

    os.environ["PATH"] = os.path.expanduser("~/.elan/bin") + ":" + os.environ["PATH"]

    with open(args.file) as f:
        code = f.read()

    # Find sorries
    lines = code.split("\n")
    sorries = []
    for i, line in enumerate(lines):
        if "sorry" in line:
            start = max(0, i - 20)
            end = min(len(lines), i + 5)
            context = "\n".join(lines[start:end])
            sorries.append((i, context))

    print(f"Found {len(sorries)} sorry(s). Budget: {args.budget} evals each.")
    print(f"Servers: {args.server}")

    # Get model name
    try:
        resp = requests.get(f"{args.server[0]}/v1/models", timeout=10)
        model = resp.json()["data"][0]["id"]
        print(f"Model: {model}\n")
    except Exception:
        model = "default"

    proved = 0
    scoreboard = {}

    for i, (line_no, context) in enumerate(sorries):
        print(f"[{i+1}/{len(sorries)}] MCTS proving sorry at line {line_no + 1}...")

        result = mcts_prove(
            code, line_no, context,
            args.server[0], model=model,
            budget=args.budget, rollouts_per_expand=args.rollouts,
        )

        if result:
            code = code.split("\n")
            code[line_no] = code[line_no].replace("sorry", result, 1)
            code = "\n".join(code)
            proved += 1
            short = model.split("/")[-1] if "/" in model else model
            scoreboard[short] = scoreboard.get(short, 0) + 1
        else:
            print(f"    Could not prove sorry at line {line_no + 1}")

    output_path = args.output or args.file
    with open(output_path, "w") as f:
        f.write(code)

    print(f"\n{'='*50}")
    print(f"MCTS Result: {proved}/{len(sorries)} sorries proved")
    print(f"Output: {output_path}")
    if scoreboard:
        print(f"Scoreboard: {scoreboard}")


if __name__ == "__main__":
    main()
