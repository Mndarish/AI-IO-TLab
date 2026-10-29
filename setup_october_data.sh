#!/bin/bash
# Download the 14th/15th October TLab datasets into the repo root (run once on Vanda).
set -euo pipefail
cd "${AIRIO_REPO_DIR:-$HOME/Air-IO}"

get() {   # get <folder> <url>
    local dir="$1" url="$2" zip="${2##*/}"
    if [ -d "$dir" ]; then echo "$dir already exists"; return; fi
    wget -c "$url" -O "$zip"
    unzip -q "$zip"
    rm -f "$zip"
}

BASE=https://github.com/souvik0306/AirIMU/releases/download/15th_oct
get T-Lab_14th_October_dataset "$BASE/T-Lab_14th_October_dataset.zip"
get T-Lab_15th_October_dataset "$BASE/T-Lab_15th_October_dataset.zip"

for d in T-Lab_14th_October_dataset T-Lab_15th_October_dataset; do
    n=$(find "$d" -name imu.csv | wc -l)
    echo "$d: $n flights"
done
echo "expected: 23 and 21"
