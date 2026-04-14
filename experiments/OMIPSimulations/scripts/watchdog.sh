#!/bin/bash
# Watchdog that keeps store.sh jobs alive for the given CONFIGs.
# Usage: ./watchdog.sh orca halfdegree tenthdegree
# Run inside tmux from the same directory as store.sh.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <config1> [config2] ..."
    echo "Example: $0 orca halfdegree"
    exit 1
fi

CONFIGS=("$@")

while true; do
    for cfg in "${CONFIGS[@]}"; do
        if ! squeue -u "$USER" -n "store_${cfg}" -h | grep -q .; then
            echo "$(date): store_${cfg} not found, relaunching"
            ./store.sh "$cfg"
        else
            echo "$(date): store_${cfg} is running"
        fi
    done
    sleep 3600
done
