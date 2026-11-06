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
res=$here/../results/Q1
preds=$here/../predictions/Q1
mkdir -p "$res"

if [ "$what" = "base" ]; then
    python base.py bias "$data/q1_bbq_dev.jsonl" "$res/base_bias_dev.jsonl"
    python base.py qa "$data/q1_qa_dev.jsonl" "$res/base_qa_dev.jsonl"
    python ../eval.py q1_bias "$res/base_bias_dev.jsonl" "$data/q1_bbq_dev.jsonl" | tee "$res/base_bias_dev.txt"
    python ../eval.py q1_qa "$res/base_qa_dev.jsonl" "$data/q1_qa_dev.jsonl" | tee "$res/base_qa_dev.txt"
elif [ "$what" = "step1" ]; then
    out=$(realpath -m "${2:-$here/../out/q1_step1}")
    python train.py "$out" "$data" words
    ./infer.sh bias "$out" "$data/q1_bbq_dev.jsonl"
    python ../eval.py q1_bias "$preds/output_bias.jsonl" "$data/q1_bbq_dev.jsonl" | tee "$res/step1_bias_dev.txt"
    ./infer.sh qa "$out" "$data/q1_qa_dev.jsonl"
    python ../eval.py q1_qa "$preds/output_qa.jsonl" "$data/q1_qa_dev.jsonl" | tee "$res/step1_qa_dev.txt"
elif [ "$what" = "step2" ]; then
    out=$(realpath -m "${2:-$here/../out/q1}")
    ./train.sh "$out" | tee "$res/step2_train_log.txt"
    ./infer.sh bias "$out" "$data/q1_bbq_dev.jsonl"
    python ../eval.py q1_bias "$preds/output_bias.jsonl" "$data/q1_bbq_dev.jsonl" | tee "$res/step2_bias_dev.txt"
    ./infer.sh qa "$out" "$data/q1_qa_dev.jsonl"
    python ../eval.py q1_qa "$preds/output_qa.jsonl" "$data/q1_qa_dev.jsonl" | tee "$res/step2_qa_dev.txt"
elif [ "$what" = "timing" ]; then
    out=$(realpath -m "${2:-$here/../out/q1}")
    tdir=$here/../out/q1_timing
    mkdir -p "$tdir"
    python -c "
import json
from common import read, write
b = read('$data/q1_bbq_dev.jsonl')
q = read('$data/q1_qa_dev.jsonl')
bb = []
for k in range(3):
    for r in b:
        bb.append({'id': 'q1_bbq_test_%05d' % len(bb), 'category': r['category'], 'context': r['context'], 'question': r['question'], 'options': r['options']})
qq = []
for k in range(5):
    for r in q:
        qq.append({'id': 'q1_utility_test_%05d' % len(qq), 'question': r['question'], 'options': r['options']})
write('$tdir/bias_1800.jsonl', bb)
write('$tdir/qa_500.jsonl', qq)
"
    start=$(date +%s)
    ./infer.sh bias "$out" "$tdir/bias_1800.jsonl"
    mid=$(date +%s)
    ./infer.sh qa "$out" "$tdir/qa_500.jsonl"
    end=$(date +%s)
    echo "bias 1800 items: $((mid - start)) s (limit 1800 s), qa 500 items: $((end - mid)) s (limit 1800 s)" | tee "$res/timing.txt"
    wc -l "$preds/output_bias.jsonl" "$preds/output_qa.jsonl" | tee -a "$res/timing.txt"
    ./infer.sh bias "$out" "$data/q1_bbq_dev.jsonl"
    ./infer.sh qa "$out" "$data/q1_qa_dev.jsonl"
else
    echo "usage: ./run.sh base | step1 [out_dir] | step2 [out_dir] | timing [out_dir]"
    exit 1
fi
