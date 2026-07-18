#!/usr/bin/env bash
# CAMPAIGN MONITOR (2026-07-18) — the proven v3 state-class pattern, pointed at the campaign.
# Run from the repo root in Git Bash, backgrounded, at launch (runbook §5a):
#   NTFY_URL=https://ntfy.sh/<private-topic> bash scripts/campaign_monitor.sh   # push opt-in
# Emits a line ONLY on change (state-class transitions for our campaign jobs, local record
# counts for the campaign roots, any Eqw, driver-heartbeat staleness >15 min). Quiet = healthy.
#
# NOTE: THE BATCH FILTER IS PART OF THE INSTRUMENT (fixed 2026-07-27). It matched only c1_/h3ss_, so
# under MODE D — where the core is ONE of TWELVE lines and the other ten are the replication legs
# tagged leg1..leg10 — this monitor watched the core and reported the legs as nothing at all. A
# monitor that sees zero rows emits nothing, and silence is this script's own signal for HEALTHY:
# ten dead driver lines would have looked exactly like ten healthy ones. Keep this regex in sync
# with the batch tags in scripts/mode_d_supervisor.ps1 ($legTag + "c1" + "h3ss").
# With NTFY_URL set, every emitted line is ALSO pushed to the phone (ntfy.sh-compatible POST;
# priority high when Eqw or a stale heartbeat is present — the two states that need a human).
# Best-effort: a failed POST never breaks the loop.

notify() {  # $1 = message, $2 = "high" | ""
  [ -n "$NTFY_URL" ] || return 0
  curl -s -m 10 ${2:+-H "Priority: high"} -H "Title: llmrp campaign" \
       -d "$1" "$NTFY_URL" > /dev/null 2>&1 || true
}

notify "campaign monitor armed ($(date '+%m-%d %H:%M'))" ""
PREV=""
while true; do
  QS=$(ssh myriad "qstat 2>/dev/null" 2>/dev/null | awk '/c1_|h3ss_|leg([1-9]|10)_/ {print $1, $3, $5}' | sort | uniq | tr '\n' ';')
  EQW=$(printf '%s' "$QS" | grep -c "Eqw")
  NSEARCH=$(find outputs/campaign_cluster/search -name record.json 2>/dev/null | wc -l)
  NTEST=$(find outputs/campaign_cluster/test -name record.json 2>/dev/null | wc -l)
  STALE=""
  NOW=$(date +%s)
  for hb in outputs/campaign_cluster/driver_status/*.json; do
    [ -f "$hb" ] || continue
    TS=$(python -c "import json,sys;print(int(float(json.load(open(sys.argv[1])).get('ts',0))))" "$hb" 2>/dev/null || echo 0)
    if [ "$TS" -gt 0 ] && [ $((NOW - TS)) -gt 900 ]; then
      STALE="$STALE $(basename "$hb" .json):$(( (NOW - TS) / 60 ))m"
    fi
  done
  STATE="search=$NSEARCH test=$NTEST eqw=$EQW stale=${STALE:-none} | $QS"
  if [ "$STATE" != "$PREV" ]; then
    LINE="$(date '+%m-%d %H:%M') | search=$NSEARCH test=$NTEST eqw=$EQW stale=${STALE:-none}"
    echo "$LINE"
    PRIO=""
    if [ "$EQW" -gt 0 ] || [ -n "$STALE" ]; then PRIO="high"; fi
    notify "$LINE" "$PRIO"
    # full queue detail only when the queue-state part changed
    if [ "${STATE#*|}" != "${PREV#*|}" ]; then
      printf '%s' "$QS" | tr ';' '\n' | sed 's/^/    /' | head -40
    fi
    PREV="$STATE"
  fi
  sleep 300
done
