#!/bin/bash
# Verify task 04 in $1. Exit 0 = pass.
W="$1"
cd "$W" || exit 1

[ -f ANSWER.txt ] || { echo "FAIL: ANSWER.txt not created"; exit 1; }

got=$(tr -d ' \t\n\r' < ANSWER.txt)
if [ "$got" != "0.734" ]; then
  echo "FAIL: ANSWER.txt = '$got', expected '0.734'"
  exit 1
fi
exit 0
