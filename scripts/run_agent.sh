#!/bin/bash
# Launch the research agent.
#
# The agent works with ANY ONE of these (in priority order):
#   1. Claude Code installed (uses 'claude -p', no API key needed)
#   2. ANTHROPIC_API_KEY set (uses Claude API directly)
#   3. OPENAI_API_KEY set (uses GPT-4.1)
#
# For multi-model peer reviews, OPENAI_API_KEY enables o3-pro/gpt-4.1/o3.
# But even without it, the agent runs the full cycle using whatever is available.
#
# Usage:
#   ./scripts/run_agent.sh              # one tick
#   ./scripts/run_agent.sh --loop       # loop every 10m
#   ./scripts/run_agent.sh --dry-run    # report only
#
# Examples:
#   # Claude Code user (no API keys needed):
#   ./scripts/run_agent.sh
#
#   # Anthropic API only:
#   export ANTHROPIC_API_KEY='sk-ant-...'
#   ./scripts/run_agent.sh
#
#   # OpenAI API only:
#   export OPENAI_API_KEY='sk-proj-...'
#   ./scripts/run_agent.sh
#
#   # Both (best: Claude for analysis, OpenAI for diverse reviews):
#   export ANTHROPIC_API_KEY='sk-ant-...'
#   export OPENAI_API_KEY='sk-proj-...'
#   ./scripts/run_agent.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Check what's available
HAS_CLAUDE=$(which claude 2>/dev/null && echo "yes" || echo "no")
HAS_ANTHROPIC=$([ -n "${ANTHROPIC_API_KEY:-}" ] && echo "yes" || echo "no")
HAS_OPENAI=$([ -n "${OPENAI_API_KEY:-}" ] && echo "yes" || echo "no")

if [ "$HAS_CLAUDE" = "no" ] && [ "$HAS_ANTHROPIC" = "no" ] && [ "$HAS_OPENAI" = "no" ]; then
    echo "ERROR: Need at least one of:"
    echo "  - Claude Code installed (claude CLI)"
    echo "  - ANTHROPIC_API_KEY set"
    echo "  - OPENAI_API_KEY set"
    exit 1
fi

echo "LLM availability: claude=$HAS_CLAUDE anthropic=$HAS_ANTHROPIC openai=$HAS_OPENAI"

# Pass API_KEY for review script
export API_KEY="${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-}}"

if [ "${1:-}" = "--loop" ]; then
    shift
    exec python3 scripts/research_agent.py --interval "${1:-10m}" "${@:2}"
elif [ "${1:-}" = "--dry-run" ]; then
    exec python3 scripts/research_agent.py --once --dry-run "${@:2}"
else
    exec python3 scripts/research_agent.py --once "$@"
fi
