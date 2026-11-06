#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
export HF_HOME=${HF_HOME:-$here/../hf-cache}
if [ -f ../.venv/bin/activate ]; then
    source ../.venv/bin/activate
fi
what=$1
data=$here/../data/public
res=$here/../results/Q2
base=$here/../out/q2
mkdir -p "$res" "$base"

one() {
    local out=$1
    local inp=$2
    local tag=$3
    ./generate.sh "$inp" "$out" 42
    python ../eval.py q2_gen "$out/generated_42.jsonl" "$inp"
    ./perturb.sh "$out/generated_42.jsonl" "$out" 42
    python make_human.py "$inp" "$out"
    for c in clean short substitution paraphrase; do
        ./detect.sh "$out/generated_${c}_42.jsonl" "$out"
    done
    ./detect.sh "$out/human.jsonl" "$out"
    ./detect.sh "$out/human_short.jsonl" "$out"
    python ../eval.py q2_det "$out" | tee "$res/${tag}_det.txt"
    python variants.py "$out" 42 | tee "$res/${tag}_variants.txt"
}

plain() {
    local inp=$1
    local out=$2
    python setup.py "$out" 0.25 0.0
    ./generate.sh "$inp" "$out" 42
}

if [ "$what" = "plain" ]; then
    plain "$data/q2_prompts_dev.jsonl" "$base/plain"
elif [ "$what" = "dev" ]; then
    g=${2:-0.25}
    d=${3:-2.0}
    s=${4:-unique}
    tag=g${g}_d${d}_${s}
    out=$base/$tag
    python setup.py "$out" "$g" "$d" "$s"
    one "$out" "$data/q2_prompts_dev.jsonl" "$tag"
    if [ ! -f "$base/plain/generated_42.jsonl" ]; then
        plain "$data/q2_prompts_dev.jsonl" "$base/plain"
    fi
    python ../eval.py q2_ppl "$data/q2_prompts_dev.jsonl" "$out/generated_42.jsonl" "$base/plain/generated_42.jsonl" | tee "$res/${tag}_ppl.txt"
elif [ "$what" = "sweep" ]; then
    for gd in "0.25 1.0" "0.25 1.5" "0.25 2.0" "0.25 2.5" "0.25 3.0" "0.5 2.0"; do
        ./run.sh dev $gd unique
    done
elif [ "$what" = "seeds" ]; then
    out=$base/seeds
    ./setup.sh "$out"
    head -n 8 "$data/q2_prompts_dev.jsonl" > "$out/small.jsonl"
    ./generate.sh "$out/small.jsonl" "$out" 1
    cp "$out/generated_1.jsonl" "$out/first_1.jsonl"
    ./generate.sh "$out/small.jsonl" "$out" 1
    ./generate.sh "$out/small.jsonl" "$out" 2
    python -c "
import json
a = [json.loads(l)['token_ids'] for l in open('$out/first_1.jsonl')]
b = [json.loads(l)['token_ids'] for l in open('$out/generated_1.jsonl')]
c = [json.loads(l)['token_ids'] for l in open('$out/generated_2.jsonl')]
print('same seed gives same output:', a == b)
print('different seeds give different output:', all(x != y for x, y in zip(a, c)))
" | tee "$res/seed_check.txt"
elif [ "$what" = "smoke" ]; then
    out=$base/smoke
    python setup.py "$out"
    head -n 8 "$data/q2_prompts_dev.jsonl" > "$out/small.jsonl"
    one "$out" "$out/small.jsonl" smoke
    plain "$out/small.jsonl" "$out/plain"
    python ../eval.py q2_ppl "$out/small.jsonl" "$out/generated_42.jsonl" "$out/plain/generated_42.jsonl"
elif [ "$what" = "qseed" ]; then
    s=$2
    shift 2
    if [ ! -f "$base/plain_s$s/generated_$s.jsonl" ]; then
        python setup.py "$base/plain_s$s" 0.25 0.0
        ./generate.sh "$data/q2_prompts_dev.jsonl" "$base/plain_s$s" "$s"
    fi
    for d in "$@"; do
        out=$base/g0.25_d${d}_s$s
        python setup.py "$out" 0.25 "$d" oz
        ./generate.sh "$data/q2_prompts_dev.jsonl" "$out" "$s"
        python ../eval.py q2_ppl "$data/q2_prompts_dev.jsonl" "$out/generated_$s.jsonl" "$base/plain_s$s/generated_$s.jsonl" | tee "$res/g0.25_d${d}_s${s}_ppl.txt"
    done
elif [ "$what" = "timing" ]; then
    out=$base/timing
    ./setup.sh "$out"
    python -c "
import json
from common import read, write
p = read('$data/q2_prompts_dev.jsonl')
pp = []
hh = []
for k in range(5):
    for r in p:
        if len(pp) < 300:
            pp.append({'id': 'q2_prompts_test_%05d' % len(pp), 'prompt': r['prompt']})
        hh.append({'id': 'q2_human_test_%05d' % len(hh), 'text': r['human_text']})
write('$out/prompts_300.jsonl', pp)
write('$out/human_500.jsonl', hh)
"
    start=$(date +%s)
    ./generate.sh "$out/prompts_300.jsonl" "$out" 42
    mid=$(date +%s)
    ./detect.sh "$out/human_500.jsonl" "$out"
    end=$(date +%s)
    python ../eval.py q2_gen "$out/generated_42.jsonl" "$out/prompts_300.jsonl"
    echo "generate 300 prompts: $((mid - start)) s (limit 1800 s), detect 500 texts: $((end - mid)) s (limit 900 s)" | tee "$res/timing.txt"
else
    echo "usage: ./run.sh smoke | seeds | plain | dev <gamma> <delta> [plain|unique|weighted] | sweep | timing"
    exit 1
fi
