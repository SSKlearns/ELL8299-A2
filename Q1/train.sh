#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$1" ]; then
    echo "usage: ./train.sh <absolute_path_to_output_dir>"
    exit 1
fi
mkdir -p "$1"
cd "$here"
python train.py "$1" "$here/../data/public"
