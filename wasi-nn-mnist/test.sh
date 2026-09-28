#!/usr/bin/env bash
# Post every sample to a running demo and check the predicted digit.
# Usage: ./test.sh [base-url]   (default http://127.0.0.1:3000)
set -uo pipefail
base="${1:-http://127.0.0.1:3000}"
here="$(cd "$(dirname "$0")" && pwd)"
pass=0
fail=0
check() { # file expected-digit
  local f="$1" want="$2" out got conf
  if ! out=$(curl -sS -X POST --data-binary @"$f" "$base/classify"); then
    echo "FAIL  $(basename "$f")  (curl error)"; fail=$((fail + 1)); return
  fi
  got=$(printf '%s' "$out" | sed -nE 's/.*"digit":([0-9]).*/\1/p')
  conf=$(printf '%s' "$out" | sed -nE 's/.*"confidence":([0-9.]+).*/\1/p')
  if [ "$got" = "$want" ]; then
    echo "PASS  $(basename "$f")  digit=$got confidence=$conf"; pass=$((pass + 1))
  else
    echo "FAIL  $(basename "$f")  want=$want got='$got'  body: $out"; fail=$((fail + 1))
  fi
}
for f in "$here"/samples/digit-*.png "$here"/samples/digit-*.u8; do
  check "$f" "$(basename "$f" | sed -E 's/^digit-([0-9]).*/\1/')"
done
check "$here/samples/mnist_5.jpg" 5        # ort's own test image
check "$here/samples/zoo-test0.f32le" 3    # model-zoo test tensor (random noise, expected class 3)
echo "passed=$pass failed=$fail"
[ "$fail" = 0 ]
