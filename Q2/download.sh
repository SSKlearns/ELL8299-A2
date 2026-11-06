#!/bin/bash
set -e
repo=SSK0908/ell8299
rev=698f049022bf472d11cc25a2f5bb8dc681e59041
part=q2
if [ -z "$1" ]; then
    echo "usage: ./download.sh <output_dir>"
    exit 1
fi
out=$(realpath -m "$1")
mkdir -p "$out"
python -c "
import os
import sys
import shutil
from huggingface_hub import snapshot_download
out, repo, rev, part = sys.argv[1:5]
tmp = os.path.join(out, 'hf_download')
snapshot_download(repo_id=repo, revision=rev, allow_patterns=[part + '/*'], local_dir=tmp)
for f in os.listdir(os.path.join(tmp, part)):
    shutil.copy(os.path.join(tmp, part, f), os.path.join(out, f))
shutil.rmtree(tmp)
print('downloaded', sorted(os.listdir(out)))
" "$out" "$repo" "$rev" "$part"
