#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$1" ]; then
    echo "usage: ./train.sh <output_dir>"
    exit 1
fi
out=$(realpath -m "$1")
mkdir -p "$out"
cd "$here"
python train.py "$out" "$here/../data/public"
