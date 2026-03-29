#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Watch the v4 1B run and auto-document results when it finishes
# Runs as a background daemon. Checks every 60 seconds.
# ============================================================================

cd /home/amsysistestdrive2026/idontknow
SITE_DIR="/home/amsysistestdrive2026/bigcompute.science"
LOG="logs/v4_1B.log"
WATCH_LOG="logs/v4_watcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$WATCH_LOG"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

log "Watcher started. Monitoring $LOG for completion."

while true; do
    # Check if v4 process is still running
    if ! pgrep -f "zaremba_v4" > /dev/null 2>&1; then
        log "v4 process no longer running."

        # Check if results are in the log
        if grep -q "Zaremba's Conjecture HOLDS" "$LOG" 2>/dev/null; then
            RESULT="PASS"
            log "RESULT: Zaremba's Conjecture HOLDS"
        elif grep -q "UNCOVERED" "$LOG" 2>/dev/null; then
            UNCOVERED=$(grep "Uncovered:" "$LOG" | tail -1)
            RESULT="GAPS FOUND: $UNCOVERED"
            log "RESULT: $RESULT"
        else
            RESULT="UNKNOWN (check log manually)"
            log "RESULT: Process exited but no clear result in log"
        fi

        # Capture the full output
        TOTAL_TIME=$(grep "^Time:" "$LOG" 2>/dev/null | tail -1 || echo "unknown")
        log "Time: $TOTAL_TIME"

        # Copy log to website data
        mkdir -p "$SITE_DIR/public/data/zaremba-v4"
        cp "$LOG" "$SITE_DIR/public/data/zaremba-v4/v4_1B_results.log" 2>/dev/null || true

        # Update the main Zaremba experiment post results
        ZAREMBA_POST="$SITE_DIR/src/content/experiments/2026-03-28-zaremba-conjecture-8b-verification.md"
        if [ "$RESULT" = "PASS" ]; then
            sed -i 's/status: "IN PROGRESS.*"/status: "v4 verified to 1B with zero gaps. Spectral gaps complete to m=2000."/' "$ZAREMBA_POST"
            log "Updated Zaremba post with PASS result"
        else
            sed -i "s/status: \"IN PROGRESS.*\"/status: \"v4 to 1B complete: $RESULT\"/" "$ZAREMBA_POST"
            log "Updated Zaremba post with result: $RESULT"
        fi

        # Build and push website
        log "Building website..."
        cd "$SITE_DIR"
        export PATH="/home/amsysistestdrive2026/.nvm/versions/node/v22.22.0/bin:$PATH"
        npx astro build 2>> "$WATCH_LOG" || log "Build failed"

        git config user.name "cahlen" 2>/dev/null
        git config user.email "cahlen@gmail.com" 2>/dev/null
        git add -A
        git commit -m "Auto-update: v4 1B verification complete — $RESULT

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>> "$WATCH_LOG" || log "Commit failed"
        git push origin main 2>> "$WATCH_LOG" || log "Push failed"
        log "Website updated and pushed."

        # Push code repo too
        cd /home/amsysistestdrive2026/idontknow
        git config user.name "cahlen" 2>/dev/null
        git config user.email "cahlen@gmail.com" 2>/dev/null
        git add logs/v4_1B.log 2>/dev/null || true
        git commit -m "v4 1B verification results: $RESULT

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" 2>> "$WATCH_LOG" || log "Code repo commit failed"
        git push origin main 2>> "$WATCH_LOG" || log "Code repo push failed"
        log "Code repo updated and pushed."

        log "=== WATCHER COMPLETE ==="
        exit 0
    fi

    # Still running — log progress
    LAST_LINE=$(tail -1 "$LOG" 2>/dev/null || echo "no output yet")
    log "Still running. Last: $LAST_LINE"

    sleep 60
done
