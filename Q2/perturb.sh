#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$3" ]; then
    echo "usage: ./perturb.sh <input_jsonl> <output_dir> <seed>"
    exit 1
fi
inp=$(realpath "$1")
out=$(realpath -m "$2")
cd "$here"
python perturb.py "$inp" "$out" "$3"
