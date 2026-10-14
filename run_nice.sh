#!/bin/bash
# Launch any long solver run at the lowest CPU priority so interactive work on the machine is not slowed down.
# Usage: ./run_nice.sh python run_rd.py      (or any command; env variables pass through)
exec nice -n 19 "$@"
