#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$3" ]; then
    echo "usage: ./infer.sh bias|qa <absolute_path_to_output_dir> <input_jsonl>"
    exit 1
fi
mode=$1
out=$2
inp=$(realpath "$3")
mkdir -p "$here/../predictions/Q1"
cd "$here"
python infer.py "$mode" "$out" "$inp" "$here/../predictions/Q1/output_$mode.jsonl"
