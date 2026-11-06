#!/bin/bash
# The right walk, call by call through the tools CLI; "$i.path" arguments are read off earlier answers.
set -euo pipefail
tools walk /solution/walk.json
