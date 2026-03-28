"""
LLM-assisted Lean 4 theorem prover.

This module provides a bridge between a served LLM and the Lean 4 compiler,
enabling iterative proof search where the LLM generates proof tactics and
Lean provides feedback.

Usage:
    python lean4-proving/prover.py --server http://localhost:8000 --file example.lean

    # Multi-model race (Zaremba pipeline):
    python lean4-proving/prover.py \
        --server http://localhost:8000 --server http://localhost:8001 \
        --file lean4-proving/conjectures/zaremba.lean --zaremba

Architecture:
    1. Parse a Lean 4 file with `sorry` placeholders
    2. For each sorry, ask the LLM to suggest tactics
    3. Try each suggestion with the Lean compiler
    4. If it compiles, accept; otherwise feed the error back to the LLM
    5. Repeat until all sorries are resolved or max attempts reached

    When multiple --server URLs are provided, all servers are queried in
    parallel for each sorry and the first valid proof wins.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests


def query_llm(
    server_url: str,
    prompt: str,
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    n: int = 8,
) -> list[str]:
    """Query the LLM server for proof tactic suggestions."""
    resp = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "stop": ["\n\n", "theorem", "lemma", "example", "#check"],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [c["text"].strip() for c in data["choices"]]


def query_llm_chat(
    server_url: str,
    prompt: str,
    model: str = "default",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    n: int = 8,
) -> list[str]:
    """Query the LLM server using the chat completions API."""
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [c["message"]["content"].strip() for c in data["choices"]]


def check_lean(lean_code: str, timeout: int = 120) -> tuple[bool, str]:
    """Run Lean 4 on the given code and return (success, output)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", delete=False
    ) as f:
        f.write(lean_code)
        f.flush()
        try:
            result = subprocess.run(
                ["lean", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Lean timed out"
        finally:
            os.unlink(f.name)


def extract_sorries(lean_code: str) -> list[tuple[int, str]]:
    """Find all `sorry` occurrences and their surrounding context."""
    lines = lean_code.split("\n")
    sorries = []
    for i, line in enumerate(lines):
        if "sorry" in line:
            # Grab surrounding context (up to 20 lines before, 5 after)
            start = max(0, i - 20)
            end = min(len(lines), i + 5)
            context = "\n".join(lines[start:end])
            sorries.append((i, context))
    return sorries


def build_prompt(context: str, error: Optional[str] = None) -> str:
    """Build a prompt for the LLM to generate a proof tactic."""
    prompt = f"""You are an expert Lean 4 theorem prover. Complete the proof by replacing `sorry` with valid Lean 4 tactics.

```lean4
{context}
```

"""
    if error:
        prompt += f"""The previous attempt failed with this error:
```
{error}
```

Try a different approach. """

    prompt += "Replace `sorry` with the correct proof:\n```lean4\n"
    return prompt


def build_zaremba_prompt(context: str, error: Optional[str] = None) -> str:
    """Build a prompt specialized for Zaremba witness proofs."""
    prompt = """You are an expert Lean 4 theorem prover specializing in number theory.

You need to prove theorems of the form `HasZarembaWitness d 5`, which means:
  ∃ a : Nat, 0 < a ∧ a ≤ d ∧ Nat.gcd a d = 1 ∧ (cfQuotients a d).all (· ≤ 5) = true

To prove this, provide a concrete witness `a` and discharge the obligations computationally.

The proof pattern is:
  exact ⟨WITNESS, by decide, by decide, by native_decide, by native_decide⟩

where WITNESS is a specific natural number. For example:
  -- For d=7, witness a=2:
  exact ⟨2, by decide, by decide, by native_decide, by native_decide⟩

  -- For d=20, witness a=9:
  exact ⟨9, by decide, by decide, by native_decide, by native_decide⟩

The hint comments above each theorem tell you the witness value.

"""
    prompt += f"Here is the theorem to prove:\n```lean4\n{context}\n```\n\n"

    if error:
        prompt += f"The previous attempt failed with:\n```\n{error}\n```\nTry a different witness or approach.\n\n"

    prompt += "Provide ONLY the proof (replacing `sorry`), no explanation:\n```lean4\n"
    return prompt


def get_model_name(server_url: str) -> str:
    """Get the model name from a vLLM server."""
    try:
        resp = requests.get(f"{server_url}/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()["data"]
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "default"


def query_all_servers(
    servers: list[str],
    prompt: str,
    n_per_server: int = 8,
    max_tokens: int = 2048,
    temperature: float = 0.6,
) -> list[tuple[str, str]]:
    """Query all servers in parallel, return (suggestion, server_name) pairs."""
    results = []

    def _query(server_url):
        model = get_model_name(server_url)
        short_name = model.split("/")[-1] if "/" in model else model
        try:
            suggestions = query_llm_chat(
                server_url, prompt, model=model,
                max_tokens=max_tokens, temperature=temperature, n=n_per_server,
            )
            return [(s, short_name) for s in suggestions]
        except Exception as e:
            print(f"    Warning: {server_url} failed: {e}")
            return []

    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        futures = [pool.submit(_query, s) for s in servers]
        for f in as_completed(futures):
            results.extend(f.result())

    return results


def prove_sorry(
    lean_code: str,
    sorry_line: int,
    context: str,
    servers: list[str],
    max_attempts: int = 5,
    zaremba: bool = False,
) -> Optional[tuple[str, str]]:
    """Try to prove a single sorry using LLM(s). Returns (proof, model_name) or None."""
    lines = lean_code.split("\n")
    error = None

    for attempt in range(max_attempts):
        if zaremba:
            prompt = build_zaremba_prompt(context, error)
        else:
            prompt = build_prompt(context, error)

        suggestions = query_all_servers(servers, prompt)

        for suggestion, model_name in suggestions:
            # Clean up the suggestion
            suggestion = suggestion.strip()
            # Strip markdown code fences
            suggestion = re.sub(r"```\w*\n?", "", suggestion).strip()
            if suggestion.endswith("```"):
                suggestion = suggestion[:-3].strip()
            # Strip markdown headers
            suggestion = re.sub(r"^###?\s.*\n+", "", suggestion).strip()

            # If model output a full theorem, extract just the proof body after `:= by`
            if ":= by" in suggestion:
                proof_part = suggestion.split(":= by", 1)[1].strip()
                suggestion = proof_part
            elif "by\n" in suggestion and "theorem" in suggestion:
                proof_part = suggestion.split("by\n", 1)[1].strip()
                suggestion = proof_part

            # Truncate at next theorem/section/markdown to avoid trailing junk
            for stop in ["\ntheorem ", "\nlemma ", "\n--", "\n/-", "\n###", "\n#check", "\n\n\n"]:
                if stop in suggestion:
                    suggestion = suggestion[:suggestion.index(stop)].strip()

            # If the proof is a one-liner like `exact ⟨...⟩`, take just the first line
            # unless it's a multi-line tactic block
            if suggestion.startswith("exact") and "⟩" in suggestion:
                # Find the matching closing bracket
                depth = 0
                for ci, ch in enumerate(suggestion):
                    if ch == '⟨':
                        depth += 1
                    elif ch == '⟩':
                        depth -= 1
                        if depth == 0:
                            suggestion = suggestion[:ci+1]
                            break

            # For Zaremba proofs, the gcd and CF checks need native_decide
            # The model often uses `decide` which times out on computation
            if zaremba:
                suggestion = re.sub(
                    r"by decide, by decide⟩",
                    "by native_decide, by native_decide⟩",
                    suggestion,
                )
                # Also try fixing patterns with all four as decide
                suggestion = re.sub(
                    r"by decide, by decide, by decide, by decide⟩",
                    "by decide, by decide, by native_decide, by native_decide⟩",
                    suggestion,
                )

            # Reject suggestions that still use sorry
            if "sorry" in suggestion:
                continue

            # Replace sorry with the suggestion
            new_lines = lines.copy()
            new_lines[sorry_line] = new_lines[sorry_line].replace(
                "sorry", suggestion, 1
            )
            new_code = "\n".join(new_lines)

            print(f"    Trying [{model_name}]: {suggestion[:80]}...")
            success, output = check_lean(new_code)
            if success:
                return suggestion, model_name
            else:
                # Show first meaningful error line
                for err_line in output.split("\n"):
                    if "error" in err_line.lower():
                        print(f"      Lean: {err_line.strip()[:100]}")
                        break
                error = output

        print(f"    Attempt {attempt + 1}/{max_attempts} failed, retrying...")

    return None


def prove_file(
    filepath: str,
    servers: list[str],
    max_attempts: int = 5,
    zaremba: bool = False,
) -> tuple[str, int, int, dict]:
    """Attempt to prove all sorries in a Lean 4 file.

    Returns (code, proved_count, total_count, scoreboard).
    """
    with open(filepath) as f:
        code = f.read()

    sorries = extract_sorries(code)
    if not sorries:
        print("No `sorry` found in the file.")
        return code, 0, 0, {}

    print(f"Found {len(sorries)} sorry(s) to prove.")
    proved = 0
    scoreboard: dict[str, int] = {}

    for i, (line_no, context) in enumerate(sorries):
        print(f"\n[{i + 1}/{len(sorries)}] Proving sorry at line {line_no + 1}...")
        result = prove_sorry(
            code, line_no, context, servers, max_attempts,
            zaremba=zaremba,
        )
        if result:
            proof, model_name = result
            print(f"    Proved by {model_name}: {proof[:80]}")
            code = code.split("\n")
            code[line_no] = code[line_no].replace("sorry", proof, 1)
            code = "\n".join(code)
            proved += 1
            scoreboard[model_name] = scoreboard.get(model_name, 0) + 1
        else:
            print(f"    Could not prove sorry at line {line_no + 1}")

    return code, proved, len(sorries), scoreboard


def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted Lean 4 theorem prover"
    )
    parser.add_argument("--server", action="append", default=[],
                        help="LLM server URL (can specify multiple for racing)")
    parser.add_argument("--file", required=True,
                        help="Lean 4 file with sorry placeholders")
    parser.add_argument("--output", help="Output file (default: overwrite input)")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max attempts per sorry")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sorries without attempting proofs")
    parser.add_argument("--zaremba", action="store_true",
                        help="Use Zaremba-specialized prompts for witness proofs")
    args = parser.parse_args()

    if not args.server:
        args.server = ["http://localhost:8000"]

    if args.dry_run:
        with open(args.file) as f:
            sorries = extract_sorries(f.read())
        for line_no, context in sorries:
            print(f"--- sorry at line {line_no + 1} ---")
            print(context)
            print()
        return

    print(f"Servers: {', '.join(args.server)}")
    for s in args.server:
        name = get_model_name(s)
        print(f"  {s} -> {name}")
    print()

    result_code, proved, total, scoreboard = prove_file(
        args.file, args.server, args.max_attempts,
        zaremba=args.zaremba,
    )

    output_path = args.output or args.file
    with open(output_path, "w") as f:
        f.write(result_code)

    print(f"\n{'='*50}")
    print(f"Result: {proved}/{total} sorries proved")
    print(f"Output: {output_path}")

    if scoreboard:
        print(f"\nScoreboard:")
        for model, count in sorted(scoreboard.items(), key=lambda x: -x[1]):
            print(f"  {model}: {count} proofs")

    if proved < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
