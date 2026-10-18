#!/bin/bash
# One frame per shot, 0.6 s before each shot's end, into frames/<shot>.png. Usage: frames.sh <stem>
set -e
cd "$(dirname "$0")"
rm -rf frames && mkdir frames
python3 -c "
import json;sh=json.load(open('$1.shots.json'));t=0
for m in sh: print(m['shot'], round(t+m['length']-0.6,2)); t+=m['length']-0.3
" | while read n t; do /opt/homebrew/bin/ffmpeg -v error -y -ss $t -i $1.mp4 -frames:v 1 frames/$n.png; echo "$n $t"; done
