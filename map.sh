#!/usr/bin/bash
set -e

# set device
gpu=6

# set task
task=multi30k-en2de

who=test2   #test, test1, test2
#who=commute
random_image_translation=0 #1
length_penalty=0.8

image_feat=vit_tiny_patch16_384
image_feat_path=~/$image_feat
image_feat_dim=192

# data set
batch_size=1
beam=1
src_lang=en
tgt_lang=de
data_dir=multi30k.en-de


model_dir=checkpoints/$task-$image_feat-entropy_test
checkpoint=last10.ensemble.pt

export CUDA_VISIBLE_DEVICES=$gpu

map_dir=$model_dir/map

if [ ! -d $map_dir ]; then
    mkdir -p $map_dir
fi


cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint
  --gen-subset $who 
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --quiet 
  --task image_mmt
  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --output $map_dir/hypo.txt" 
# --remove-bpe
if [ $random_image_translation -eq 1 ]; then
cmd=${cmd}" --random-image-translation "
fi

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $map_dir/hypo.txt $map_dir/hypo.sorted

