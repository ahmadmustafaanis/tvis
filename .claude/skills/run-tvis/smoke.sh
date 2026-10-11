#!/usr/bin/env bash
# Smoke test for the run-tvis skill. Records buggy/train.py once per injected bug with `tvis run`,
# checks that `tvis_debug.py health` names each bug (with torch blocked, as on a laptop without it),
# then runs every other driver command once. Usage: smoke.sh [OUT_DIR]   (~30 s on CPU)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-.venv/bin/python}
TVIS=${TVIS:-.venv/bin/tvis}
SKILL=.claude/skills/run-tvis
OUT=${1:-$(mktemp -d)}
NO_TORCH='import runpy, sys; sys.modules["torch"] = None; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")'
debug() { "$PY" -c "$NO_TORCH" "$SKILL/tvis_debug.py" "$@"; }

fail=0
for case in none:"no findings" softmax:loss-input-probabilities nozero:stale-gradients \
            unnorm:input-not-normalized frozen:layer-not-optimized dead:gradient-blocked lr:update-too-large; do
  bug=${case%%:*} expected=${case#*:}
  "$TVIS" run --steps 4 --out "$OUT/$bug" "$SKILL/buggy/train.py" --bug "$bug" >/dev/null 2>&1 || true
  if debug health "$OUT/$bug" | grep -q -- "$expected"; then
    echo "ok    --bug $bug → $expected"
  else
    echo "FAIL  --bug $bug: expected '$expected' in:"; debug health "$OUT/$bug"; fail=1
  fi
done

R="$OUT/softmax"
call=$(debug health "$R" --json | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[0]["where"]["call"])')
tid=$(debug call "$R" "$call" | awk '/input +input:/ {print $4; exit}')
"$TVIS" run --steps 4 --out "$R" "$SKILL/buggy/train.py" --bug none >/dev/null 2>&1  # the "fixed" run
for args in "runs $R" "summary $R --run 1" "layers $R --run 1" "params $R --run 1 --step 0" \
            "samples $R --run 1 --top 3" "data $R --run 1 --samples 1" "call $R --run 1 $call" \
            "tensor $R --run 1 $tid --grad" "source $R --run 1 train.py --line 60 --context 2" \
            "timing $R --run 1 --top 3" "compare $R --run 1 --against 0" "link $R --run 1 --page loss"; do
  # shellcheck disable=SC2086
  if debug $args >/dev/null; then echo "ok    $args"; else echo "FAIL  $args"; fail=1; fi
done
if debug shot "$R" --run 1 --page loss --out "$OUT/loss.png" >/dev/null 2>&1; then
  echo "ok    shot → $OUT/loss.png"
else
  echo "skip  shot (no Chrome/Chromium; set TVIS_CHROME)"
fi
echo "runs in $OUT"
exit $fail
