#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$2" ]; then
    echo "usage: ./detect.sh <input_jsonl> <output_dir>"
    exit 1
fi
inp=$(realpath "$1")
out=$(realpath -m "$2")
cd "$here"
python detect.py "$inp" "$out"
