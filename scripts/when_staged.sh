#!/usr/bin/env bash
# Wait until every frame of the given splits is in the stage log, then run the given command.
# Usage: scripts/when_staged.sh test,val -- <command...>
set -euo pipefail
cd "$(dirname "$0")/.."
splits=$1; shift; [ "$1" = "--" ] && shift
need() { nix develop -c uv run python - "$splits" <<'PY' 2>/dev/null | tail -1
import json, sys, pandas as pd
sp = sys.argv[1].split(",")
F = pd.concat([pd.read_csv("splits/frames.csv"), pd.read_csv("splits/eval_extra.csv")])
want = set(F[F.split.isin(sp)].image_id)
got = {json.loads(l)["image_id"] for l in open("results/stage_log.jsonl")}
print(len(want - got))
PY
}
while :; do n=$(need); echo "$(date +%H:%M) $splits: $n frames still to stage"; [ "$n" = "0" ] && break; sleep 300; done
exec "$@"
