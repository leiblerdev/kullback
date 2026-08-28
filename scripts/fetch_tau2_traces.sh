#!/usr/bin/env sh
# Downloads Sierra's public tau2-bench retail runs used by the offline slice into data/raw/.
# data/raw is gitignored: raw traces are never committed.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
dest="$here/data/raw"
mkdir -p "$dest"
base="https://sierra-tau-bench-public.s3.amazonaws.com/submissions"
for p in \
  "claude-3-7-sonnet_anthropic_2024-06-20/trajectories/claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json" \
  "claude-sonnet-4-5_sierra_2026-02-26/trajectories/claude-sonnet-4-5_enabled_retail_gpt-5.2_4trials.json"; do
  f=$(basename "$p")
  if [ -f "$dest/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -fL --retry 3 -o "$dest/$f.part" "$base/$p" && mv "$dest/$f.part" "$dest/$f"
  fi
done
ls -l "$dest"
