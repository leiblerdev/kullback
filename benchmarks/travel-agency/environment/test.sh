#!/bin/bash
# The grade: the tool server replays its own call log on a fresh world and grades it (runtime.py verify).
mkdir -p /logs/verifier
python3 /tests/runtime.py verify /tests/grade.json --logs /logs/verifier
