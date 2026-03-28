"""
LLM-assisted Lean 4 theorem prover.

This module provides a bridge between a served LLM and the Lean 4 compiler,
enabling iterative proof search where the LLM generates proof tactics and
Lean provides feedback.

Usage:
    python lean4-proving/prover.py --server http://localhost:8000 --file example.lean

Architecture:
    1. Parse a Lean 4 file with `sorry` placeholders
    2. For each sorry, ask the LLM to suggest tactics
    3. Try each suggestion with the Lean compiler
    4. If it compiles, accept; otherwise feed the error back to the LLM
    5. Repeat until all sorries are resolved or max attempts reached
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
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


def check_lean(lean_code: str, timeout: int = 60) -> tuple[bool, str]:
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
            success = result.returncode == 0
            output = result.stdout + result.stderr
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


def prove_sorry(
    lean_code: str,
    sorry_line: int,
    context: str,
    server_url: str,
    model: str,
    max_attempts: int = 5,
) -> Optional[str]:
    """Try to prove a single sorry using the LLM."""
    lines = lean_code.split("\n")
    error = None

    for attempt in range(max_attempts):
        prompt = build_prompt(context, error)
        suggestions = query_llm(server_url, prompt, model=model)

        for suggestion in suggestions:
            # Clean up the suggestion
            suggestion = suggestion.strip()
            if suggestion.startswith("```"):
                suggestion = re.sub(r"```\w*\n?", "", suggestion).strip()

            # Replace sorry with the suggestion
            new_lines = lines.copy()
            new_lines[sorry_line] = new_lines[sorry_line].replace(
                "sorry", suggestion, 1
            )
            new_code = "\n".join(new_lines)

            success, output = check_lean(new_code)
            if success:
                print(f"    Proved on attempt {attempt + 1}: {suggestion[:80]}")
                return suggestion
            else:
                error = output

        print(f"    Attempt {attempt + 1}/{max_attempts} failed, retrying...")

    return None


def prove_file(
    filepath: str,
    server_url: str,
    model: str = "default",
    max_attempts: int = 5,
) -> tuple[str, int, int]:
    """Attempt to prove all sorries in a Lean 4 file."""
    with open(filepath) as f:
        code = f.read()

    sorries = extract_sorries(code)
    if not sorries:
        print("No `sorry` found in the file.")
        return code, 0, 0

    print(f"Found {len(sorries)} sorry(s) to prove.")
    proved = 0

    for i, (line_no, context) in enumerate(sorries):
        print(f"\n[{i + 1}/{len(sorries)}] Proving sorry at line {line_no + 1}...")
        result = prove_sorry(
            code, line_no, context, server_url, model, max_attempts
        )
        if result:
            code = code.split("\n")
            code[line_no] = code[line_no].replace("sorry", result, 1)
            code = "\n".join(code)
            proved += 1
        else:
            print(f"    Could not prove sorry at line {line_no + 1}")

    return code, proved, len(sorries)


def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted Lean 4 theorem prover"
    )
    parser.add_argument("--server", default="http://localhost:8000",
                        help="LLM server URL")
    parser.add_argument("--model", default="default",
                        help="Model name on the server")
    parser.add_argument("--file", required=True,
                        help="Lean 4 file with sorry placeholders")
    parser.add_argument("--output", help="Output file (default: overwrite input)")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="Max attempts per sorry")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sorries without attempting proofs")
    args = parser.parse_args()

    if args.dry_run:
        with open(args.file) as f:
            sorries = extract_sorries(f.read())
        for line_no, context in sorries:
            print(f"--- sorry at line {line_no + 1} ---")
            print(context)
            print()
        return

    result_code, proved, total = prove_file(
        args.file, args.server, args.model, args.max_attempts
    )

    output_path = args.output or args.file
    with open(output_path, "w") as f:
        f.write(result_code)

    print(f"\nResult: {proved}/{total} sorries proved")
    print(f"Output: {output_path}")

    if proved < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
