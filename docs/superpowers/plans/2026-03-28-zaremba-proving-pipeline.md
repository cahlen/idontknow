# Zaremba Proving Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download a SOTA proving model, serve it on the 8xB200 cluster, and run the LLM prover against 20 Zaremba's Conjecture sorry-targets to validate the end-to-end pipeline.

**Architecture:** Goedel-Prover-V2-32B served via vLLM with 8-way tensor parallelism, feeding tactic suggestions to a Lean 4 proof checker loop. The prover harness (`prover.py`) is adapted to generate witness-based existential proofs that Lean verifies with `native_decide`.

**Tech Stack:** Lean 4.29.0, vLLM 0.18.0, PyTorch 2.10.0 (CUDA), Goedel-Prover-V2-32B (Qwen3-32B base), Python 3.12

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/download-model.sh` | Exists | HF model download |
| `scripts/serve-model.sh` | Exists | vLLM/SGLang serving |
| `lean4-proving/prover.py` | **Modify** | Adapt prompt + proof pattern for Zaremba witness style |
| `lean4-proving/conjectures/zaremba.lean` | Exists | 20 sorry-targets, already compiles |
| `scripts/run-zaremba.sh` | **Create** | One-command pipeline runner: serve model + run prover |

---

### Task 1: Download Goedel-Prover-V2-32B

**Files:**
- Use: `scripts/download-model.sh`

**Why Goedel-Prover-V2-32B:**
- 88-90% on MiniF2F (Pass@32), outperforms DeepSeek-Prover-V2-671B
- Standard Qwen3-32B transformer — native vLLM support, no MoE complexity
- 64GB at FP16 — fits on a single B200, leaving 7 GPUs free for parallel search
- Whole-proof generation with self-correction — matches our sorry-replacement pattern
- Apache 2.0 license

- [ ] **Step 1: Download the model**

```bash
./scripts/download-model.sh Goedel-LM/Goedel-Prover-V2-32B
```

Expected: Model downloads to `models/Goedel-Prover-V2-32B/`, ~64GB, takes 5-15 min depending on bandwidth.

- [ ] **Step 2: Verify download**

```bash
ls -la models/Goedel-Prover-V2-32B/config.json
python3 -c "
import json
with open('models/Goedel-Prover-V2-32B/config.json') as f:
    cfg = json.load(f)
print(f'Model: {cfg.get(\"model_type\")}, Hidden: {cfg.get(\"hidden_size\")}, Layers: {cfg.get(\"num_hidden_layers\")}')
"
```

Expected: `Model: qwen3, Hidden: 5120, Layers: 64` (or similar Qwen3-32B config)

- [ ] **Step 3: Commit**

```bash
echo "models/" >> .gitignore
git add .gitignore
git commit -m "chore: add models/ to gitignore for downloaded weights"
```

---

### Task 2: Serve the Model with vLLM

**Files:**
- Use: `scripts/serve-model.sh`

- [ ] **Step 1: Start vLLM server**

```bash
source venv/bin/activate
./scripts/serve-model.sh models/Goedel-Prover-V2-32B --engine vllm --tp 8 --port 8000 &
```

Using tp=8 for maximum throughput even though the model fits on 1 GPU — this gives us 8x the batch throughput for parallel proof search.

Expected: Server starts, logs show "Application startup complete" after loading model.

- [ ] **Step 2: Verify the server responds**

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Expected: JSON response listing the model.

- [ ] **Step 3: Test a simple completion**

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/Goedel-Prover-V2-32B",
    "prompt": "theorem test : 1 + 1 = 2 := by\n",
    "max_tokens": 64,
    "temperature": 0.0
  }' | python3 -m json.tool
```

Expected: Response containing something like `norm_num` or `rfl` or `decide`.

---

### Task 3: Adapt prover.py for Zaremba Witness Proofs

**Files:**
- Modify: `lean4-proving/prover.py`

The current prover uses a generic prompt. For Zaremba, the LLM needs to:
1. Understand it must provide a **witness** `a` for the existential
2. Generate the proof pattern: `⟨a, by decide, by decide, by native_decide, by native_decide⟩`
3. Or use a tactic-based approach: `exact ⟨a, by decide, by decide, by native_decide, by native_decide⟩`

- [ ] **Step 1: Test the current prover in dry-run mode**

```bash
source venv/bin/activate
python3 lean4-proving/prover.py --file lean4-proving/conjectures/zaremba.lean --dry-run
```

Expected: Lists all 20+ sorry locations with surrounding context.

- [ ] **Step 2: Add a Zaremba-specialized prompt builder**

Add this function to `lean4-proving/prover.py` after the existing `build_prompt` function (line 95):

```python
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
```

- [ ] **Step 3: Add `--zaremba` flag to use the specialized prompt**

Replace the `prove_sorry` function's prompt building (line 128-129) to support both modes. Modify the `main` function to add the flag:

In the `main` function, after line 205 (`--dry-run` argument), add:

```python
    parser.add_argument("--zaremba", action="store_true",
                        help="Use Zaremba-specialized prompts for witness proofs")
```

In the `prove_sorry` function, change the signature to accept a `zaremba` parameter:

```python
def prove_sorry(
    lean_code: str,
    sorry_line: int,
    context: str,
    server_url: str,
    model: str,
    max_attempts: int = 5,
    zaremba: bool = False,
) -> Optional[str]:
    """Try to prove a single sorry using the LLM."""
    lines = lean_code.split("\n")
    error = None

    for attempt in range(max_attempts):
        if zaremba:
            prompt = build_zaremba_prompt(context, error)
        else:
            prompt = build_prompt(context, error)
        suggestions = query_llm(server_url, prompt, model=model)

        for suggestion in suggestions:
            # Clean up the suggestion
            suggestion = suggestion.strip()
            if suggestion.startswith("```"):
                suggestion = re.sub(r"```\\w*\\n?", "", suggestion).strip()

            # Replace sorry with the suggestion
            new_lines = lines.copy()
            new_lines[sorry_line] = new_lines[sorry_line].replace(
                "sorry", suggestion, 1
            )
            new_code = "\n".join(new_lines)

            success, output = check_lean(new_code)
            if success:
                print(f"    ✓ Proved on attempt {attempt + 1}: {suggestion[:80]}")
                return suggestion
            else:
                error = output

        print(f"    Attempt {attempt + 1}/{max_attempts} failed, retrying...")

    return None
```

Update `prove_file` to pass the flag through:

```python
def prove_file(
    filepath: str,
    server_url: str,
    model: str = "default",
    max_attempts: int = 5,
    zaremba: bool = False,
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
            code, line_no, context, server_url, model, max_attempts,
            zaremba=zaremba,
        )
        if result:
            code = code.split("\n")
            code[line_no] = code[line_no].replace("sorry", result, 1)
            code = "\n".join(code)
            proved += 1
        else:
            print(f"    ✗ Could not prove sorry at line {line_no + 1}")

    return code, proved, len(sorries)
```

Update `main` to wire it through (replace the `prove_file` call at line 217):

```python
    result_code, proved, total = prove_file(
        args.file, args.server, args.model, args.max_attempts,
        zaremba=args.zaremba,
    )
```

- [ ] **Step 4: Increase `check_lean` timeout for `native_decide`**

`native_decide` compiles and runs Lean code, which takes longer than simple tactic checking. In the `check_lean` function, change the default timeout from 60 to 120:

```python
def check_lean(lean_code: str, timeout: int = 120) -> tuple[bool, str]:
```

- [ ] **Step 5: Verify the modified prover still works in dry-run**

```bash
source venv/bin/activate
python3 lean4-proving/prover.py --file lean4-proving/conjectures/zaremba.lean --dry-run --zaremba
```

Expected: Same output as before (dry-run doesn't use --zaremba, just verifies no syntax errors).

- [ ] **Step 6: Commit**

```bash
git add lean4-proving/prover.py
git commit -m "feat: add Zaremba-specialized prompt for witness-based proofs"
```

---

### Task 4: Create the Pipeline Runner Script

**Files:**
- Create: `scripts/run-zaremba.sh`

- [ ] **Step 1: Write the runner script**

Create `scripts/run-zaremba.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Run the Zaremba proving pipeline end-to-end
# Assumes model is downloaded and vLLM server is running on port 8000
# Usage: ./scripts/run-zaremba.sh [--server URL] [--attempts N]
# ============================================================================

SERVER="${1:-http://localhost:8000}"
ATTEMPTS=5
MODEL="models/Goedel-Prover-V2-32B"
LEAN_FILE="lean4-proving/conjectures/zaremba.lean"
OUTPUT_FILE="lean4-proving/conjectures/zaremba_proved.lean"

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server) SERVER="$2"; shift 2 ;;
        --attempts) ATTEMPTS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PATH="$HOME/.elan/bin:$PATH"

echo "=== Zaremba Proving Pipeline ==="
echo "Server: $SERVER"
echo "Model: $MODEL"
echo "Lean file: $LEAN_FILE"
echo "Max attempts per sorry: $ATTEMPTS"
echo ""

# Verify prerequisites
echo "[1/3] Checking prerequisites..."
lean --version || { echo "ERROR: lean not found"; exit 1; }
curl -sf "$SERVER/v1/models" > /dev/null || { echo "ERROR: vLLM server not responding at $SERVER"; exit 1; }
echo "  All good."
echo ""

# Dry run first
echo "[2/3] Scanning for sorry targets..."
source venv/bin/activate
python3 lean4-proving/prover.py --file "$LEAN_FILE" --dry-run
echo ""

# Run the prover
echo "[3/3] Running prover..."
python3 lean4-proving/prover.py \
    --file "$LEAN_FILE" \
    --server "$SERVER" \
    --model "$MODEL" \
    --output "$OUTPUT_FILE" \
    --max-attempts "$ATTEMPTS" \
    --zaremba

echo ""
echo "=== Pipeline complete ==="
echo "Output: $OUTPUT_FILE"

# Verify the output compiles
echo ""
echo "Verifying output with Lean..."
if lean "$OUTPUT_FILE" 2>&1 | grep -q "error"; then
    echo "WARNING: Output file has Lean errors (some sorries may remain)"
    lean "$OUTPUT_FILE" 2>&1 | grep -c "sorry" && echo "sorry warnings remaining"
else
    echo "Output compiles clean!"
fi
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/run-zaremba.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run-zaremba.sh
git commit -m "feat: add one-command Zaremba proving pipeline runner"
```

---

### Task 5: Run the Pipeline End-to-End

**Files:**
- Use: `scripts/run-zaremba.sh`, all components

- [ ] **Step 1: Ensure model is downloaded (Task 1)**

```bash
ls models/Goedel-Prover-V2-32B/config.json
```

- [ ] **Step 2: Start vLLM server in background (if not running)**

```bash
source venv/bin/activate
nohup ./scripts/serve-model.sh models/Goedel-Prover-V2-32B --engine vllm --tp 8 --port 8000 > vllm-server.log 2>&1 &
echo $! > vllm-server.pid
echo "Server PID: $(cat vllm-server.pid)"

# Wait for server to be ready (check every 5s, max 5 min)
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    echo "Waiting for server... ($i)"
    sleep 5
done
```

- [ ] **Step 3: Run the pipeline**

```bash
./scripts/run-zaremba.sh --server http://localhost:8000 --attempts 5
```

Expected output:
```
=== Zaremba Proving Pipeline ===
Found 22 sorry(s) to prove.

[1/22] Proving sorry at line 66...
    ✓ Proved on attempt 1: exact ⟨1, by decide, by decide, by native_decide, by native_decide⟩
[2/22] Proving sorry at line 69...
    ✓ Proved on attempt 1: exact ⟨1, by decide, by decide, by native_decide, by native_decide⟩
...
[21/22] Proving sorry at line 130...
    ✗ Could not prove sorry at line 130  (this is the full conjecture — expected)
[22/22] Proving sorry at line 144...
    ✗ Could not prove sorry at line 144  (Bourgain-Kontorovich — expected)

Result: 20/22 sorries proved
```

The 20 small cases (d=1..20) should be provable. The full conjecture (line 130) and Bourgain-Kontorovich (line 144) are expected to remain as sorry.

- [ ] **Step 4: Verify the output**

```bash
export PATH="$HOME/.elan/bin:$PATH"
lean lean4-proving/conjectures/zaremba_proved.lean 2>&1
```

Expected: Only 2 sorry warnings (the conjecture and Bourgain-Kontorovich), no errors.

- [ ] **Step 5: Commit the proved file**

```bash
git add lean4-proving/conjectures/zaremba_proved.lean
git commit -m "feat: first 20 Zaremba cases proved by LLM pipeline on B200 cluster"
```

---

### Task 6: Record Results and Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the checklist in CLAUDE.md**

Mark the completed items and add results:

```markdown
- [x] Cluster environment actually set up (Lean 4.29.0, vLLM 0.18.0, 8xB200)
- [x] Proving model selected and downloaded (Goedel-Prover-V2-32B)
- [x] Pipeline tested end-to-end on B200 cluster (20/20 Zaremba small cases)
- [ ] Continued fractions exploration started
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update project status — pipeline validated on Zaremba small cases"
```
