
set -euo pipefail


src=en
tgt=de
image=$1
name=$2

MODEL_PATH=checkpoints/$name/last10.ensemble.pt
IMG_FEAT_PATH=~/${image}/$name/commute/en-de/
commute_path=checkpoints/$name/commute_result


mkdir -p $commute_path

# 1. scoring correct Set
python3 fairseq_cli/generate.py data-bin/commute/correct \
    --path $MODEL_PATH \
    --task image_mmt \
    --gen-subset test \
    --source-lang en --target-lang de \
    --score-reference \
    --batch-size 64 \
    --image-feat-path $IMG_FEAT_PATH \
    --image-feat-dim 384 \
    --output $commute_path/correct.txt \
    > $commute_path/final_score_correct.log

# scoring incorrect Set
python3 fairseq_cli/generate.py data-bin/commute/incorrect \
    --path $MODEL_PATH \
    --task image_mmt \
    --gen-subset test \
    --source-lang en --target-lang de \
    --score-reference \
    --batch-size 64 \
    --image-feat-path $IMG_FEAT_PATH \
    --image-feat-dim 384 \
    --output $commute_path/incorrect.txt \
    > $commute_path/final_score_incorrect.log

# compute accuracy
python commute.py $commute_path
