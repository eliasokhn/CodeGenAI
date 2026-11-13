#!/usr/bin/env bash
# Resumable, stall-aware download of the pinned Qwen snapshot into models/ (dev-laptop helper;
# the normal path is huggingface_hub's cache, this exists for flaky connections).
set -u
REV=2e1fd397ee46e1388853d2af2c993145b0f1098a
DIR="models/Qwen2.5-Coder-1.5B-Instruct"
LOG="results/model_download.log"
: > "$LOG"
tr -d '\r' < "$DIR/.files.txt" | while IFS= read -r f; do
  [ -z "$f" ] && continue
  for attempt in $(seq 1 60); do
    if curl -L -sS -C - --speed-limit 20000 --speed-time 30 --retry 3 \
        -o "$DIR/$f" "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/$REV/$f"; then
      break
    fi
    echo "retry $attempt for $f ($(stat -c %s "$DIR/$f" 2>/dev/null || echo 0) bytes)" >> "$LOG"
    sleep 3
  done
  echo "fetched $f $(stat -c %s "$DIR/$f" 2>/dev/null || echo 0) bytes" >> "$LOG"
done
echo "DONE $DIR" >> "$LOG"
