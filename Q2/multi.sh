#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
export HF_HOME=${HF_HOME:-$here/../hf-cache}
if [ -f ../.venv/bin/activate ]; then
    source ../.venv/bin/activate
fi
data=$here/../data/public
inp=$data/q2_prompts_dev.jsonl
tag=$1
shift
root=$here/../out/q2_multi/$tag
res=$here/../results/Q2/multi_$tag
mkdir -p "$root" "$res"
for s in "$@"; do
    out=$root/s$s
    mkdir -p "$out"
    if [ -f "$root/setup_args" ]; then
        python setup.py "$out" $(cat "$root/setup_args")
    else
        ./setup.sh "$out"
    fi
    ./generate.sh "$inp" "$out" "$s"
    python ../eval.py q2_gen "$out/generated_$s.jsonl" "$inp"
    ./perturb.sh "$out/generated_$s.jsonl" "$out" "$s"
    python para_sample.py "$out/generated_$s.jsonl" "$out" "$s"
    python make_human.py "$inp" "$out"
    for c in clean short substitution paraphrase parasample; do
        ./detect.sh "$out/generated_${c}_$s.jsonl" "$out"
    done
    ./detect.sh "$out/human.jsonl" "$out"
    ./detect.sh "$out/human_short.jsonl" "$out"
    python report.py "$out" | tee "$res/det_s$s.txt"
    plain=$here/../out/q2_multi/plain/s$s
    if [ ! -f "$plain/generated_$s.jsonl" ]; then
        python setup.py "$plain" 0.25 0.0
        ./generate.sh "$inp" "$plain" "$s"
    fi
    python ../eval.py q2_ppl "$inp" "$out/generated_$s.jsonl" "$plain/generated_$s.jsonl" | tee "$res/ppl_s$s.txt"
done
