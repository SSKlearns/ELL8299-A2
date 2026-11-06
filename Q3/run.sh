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
res=$here/../results/Q3
out=$(realpath -m "${2:-$here/../out/q3}")
mkdir -p "$res"

if [ "$what" = "train" ]; then
    ./train.sh "$out" | tee "$res/train_log.txt"
elif [ "$what" = "dev" ]; then
    for f in q3_qa_dev q3_qa_dev_ood; do
        python -c "
import json
from common import read, write
write('$out/${f}_input.jsonl', [{'id': r['id'], 'question': r['question']} for r in read('$data/$f.jsonl')])
"
        ./infer.sh "$out/${f}_input.jsonl" "$out"
        python ../eval.py q3 "$out/pred_${f}_input.jsonl" "$data/$f.jsonl" | tee "$res/${f}_insample.txt"
    done
elif [ "$what" = "timing" ]; then
    python -c "
import json
from common import read, write
r = read('$data/q3_qa_train.jsonl')[:1000]
write('$out/timing_1000.jsonl', [{'id': 'q3_qa_test_id_%05d' % i, 'question': x['question']} for i, x in enumerate(r)])
"
    start=$(date +%s)
    ./infer.sh "$out/timing_1000.jsonl" "$out"
    end=$(date +%s)
    echo "infer 1000 questions: $((end - start)) s (limit 2700 s)" | tee "$res/timing.txt"
else
    echo "usage: ./run.sh train | dev | timing [out_dir]"
    exit 1
fi
