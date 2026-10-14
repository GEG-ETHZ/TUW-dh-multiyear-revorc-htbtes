#!/bin/bash
# Delete everything a run generates (results, cached revORC maps, byte-code), keep inputs and code.
# Usage: ./tools/remove_outputs.sh      (asks before deleting)
cd "$(dirname "$0")/.." || exit 1
echo "This removes: results/* , data/revorc_maps_cache/ , __pycache__/ , gurobi.log"
read -r -p "Continue? [y/N] " ans
[ "$ans" = y ] || [ "$ans" = Y ] || exit 0
find results -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
rm -rf data/revorc_maps_cache
find . -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f gurobi.log
echo "done"
