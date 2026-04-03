#!/bin/bash
# Launch the research agent with API keys from environment.
#
# Usage:
#   export ANTHROPIC_API_KEY='sk-ant-...'
#   export OPENAI_API_KEY='sk-proj-...'
#   ./scripts/run_agent.sh              # one tick
#   ./scripts/run_agent.sh --loop       # loop every 10m
#   ./scripts/run_agent.sh --dry-run    # report only
#
# Or source your keys from a non-committed file:
#   source ~/.bigcompute_keys && ./scripts/run_agent.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Check keys
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set — analysis phase will be skipped"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set — OpenAI reviews will be skipped"
fi

# Pass API_KEY for review script (it uses API_KEY not OPENAI_API_KEY)
export API_KEY="${OPENAI_API_KEY:-}"

if [ "${1:-}" = "--loop" ]; then
    shift
    exec python3 scripts/research_agent.py --interval "${1:-10m}" "${@:2}"
elif [ "${1:-}" = "--dry-run" ]; then
    exec python3 scripts/research_agent.py --once --dry-run "${@:2}"
else
    exec python3 scripts/research_agent.py --once "$@"
fi
