#! /usr/bin/bash
set -e

name=$1

if [ -z "$name" ]; then
  echo "Error: model name argument is required."
  exit 1
fi

device=4
gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$device


for task in multi30k-en2de multi30k-en2fr; do

#task=multi30k-en2de
image_feat=vit_small_patch16_384
save_dir=checkpoints/$task-$image_feat-$name

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/shell.sh


if [ $task == 'multi30k-en2de' ]; then
	src_lang=en
	tgt_lang=de
    data_dir=multi30k.en-de
elif [ $task == 'multi30k-en2fr' ]; then
	src_lang=en
	tgt_lang=fr
    data_dir=multi30k.en-fr
elif [ $task == 'multi30k-de2en' ]; then
	src_lang=de
	tgt_lang=en
    data_dir=multi30k.en-de
elif [ $task == 'multi30k-fr2en' ]; then
	src_lang=fr
	tgt_lang=en
    data_dir=multi30k.en-fr
fi

if [ $image_feat == "vit_tiny_patch16_384" ]; then
	image_feat_path=~/$image_feat
	image_feat_dim=192
elif [ $image_feat == "vit_small_patch16_384" ]; then
	image_feat_path=~/$image_feat
	image_feat_dim=384
elif [ $image_feat == "vit_base_patch16_384" ]; then
	image_feat_path=~/$image_feat
	image_feat_dim=768
elif [ $image_feat == "vit_large_patch16_384" ]; then
	image_feat_path=~/$image_feat
	image_feat_dim=1024
fi

image_feat_path=~/vit_small_patch16_384/layer12


criterion=label_smoothed_cross_entropy
fp16=0 #0
lr=0.005
warmup=2000
max_tokens=8192 #4096
update_freq=1
keep_last_epochs=10
patience=10
max_update=8000
dropout=0.3

arch=image_multimodal_transformer_SA_top


fairseq-train data-bin/$data_dir \
--save-dir $save_dir \
--distributed-world-size $gpu_num -s $src_lang -t $tgt_lang \
--arch $arch \
--dropout $dropout \
--criterion $criterion --label-smoothing 0.1 \
--task image_mmt --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr $lr --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup \
--max-tokens $max_tokens --update-freq $update_freq --max-update $max_update \
--find-unused-parameters \
--share-all-embeddings \
--patience $patience \
--clip-norm 10.0 \
--weight-decay 0.0001 \
--keep-last-epochs $keep_last_epochs \
--eval-bleu --eval-tokenized-bleu --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
2>&1 | tee $save_dir/train.log



ensemble=10
batch_size=128
beam=5
random_image_translation=0 #1
length_penalty=1.0

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$save_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi


for who in test test2; do

    output=$save_dir/translation_$who.log

    cmd="fairseq-generate data-bin/$data_dir
      -s $src_lang -t $tgt_lang 
      --path $save_dir/$checkpoint 
      --gen-subset $who 
      --batch-size $batch_size --beam $beam --lenpen $length_penalty 
      --quiet --remove-bpe
      --task image_mmt
      --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
      --output $save_dir/hypo.txt" 

    if [ $random_image_translation -eq 1 ]; then
        cmd=${cmd}" --random-image-translation "
    fi

    cmd=${cmd}" | tee "${output}
    eval $cmd

    python3 rerank.py $save_dir/hypo.txt $save_dir/hypo_${who}.sorted

    if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
        ref=data/multi30k/test.2016.de
    elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
        ref=data/multi30k/test.2017.de
    elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
        ref=data/multi30k/test.coco.de
    elif [ $task == "multi30k-en2de" ] && [ $who == "test3" ]; then
        ref=data/multi30k/test.CoMMuTE.de

    elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
        ref=data/multi30k/test.2016.fr
    elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
        ref=data/multi30k/test.2017.fr
    elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
        ref=data/multi30k/test.coco.fr
    
    elif [ $task == "multi30k-de2en" ] && [ $who == 'test' ]; then
        ref=data/multi30k/test.2016.en
    elif [ $task == "multi30k-de2en" ] && [ $who == 'test1' ]; then
        ref=data/multi30k/test.2017.en
    elif [ $task == "multi30k-de2en" ] && [ $who == 'test2' ]; then
        ref=data/multi30k/test.coco.en

    elif [ $task == "multi30k-fr2en" ] && [ $who == 'test' ]; then
        ref=data/multi30k/test.2016.en
    elif [ $task == "multi30k-fr2en" ] && [ $who == 'test1' ]; then
        ref=data/multi30k/test.2017.en
    elif [ $task == "multi30k-fr2en" ] && [ $who == 'test2' ]; then
        ref=data/multi30k/test.coco.en
    fi	

    hypo=$save_dir/hypo_${who}.sorted
    python3 meteor.py $hypo $ref > $save_dir/meteor_$who.log
    cat $save_dir/meteor_$who.log

    # python3 cal_acc.py $hypo $who $task > $save_dir/probing_$who.log
    # cat $save_dir/probing_$who.log

done


# masks=(mask1 mask2 mask3 mask4 maskc maskp)
# for mask in "${masks[@]}"; do
#     bash translate_mask.sh $task $image_feat $mask
#     # bash translate_mask.sh $task $image_feat $mask 1
#     bash translate_mask.sh $task $image_feat $mask 2
# done


model_path=$save_dir/$checkpoint
commute_path=$save_dir/commute_result

if [ ! -d $commute_path ];  then
    mkdir -p $commute_path
fi


# 1. scoring correct Set
python3 fairseq_cli/generate.py data-bin/$data_dir/correct \
    --path $model_path \
    --task image_mmt \
    --gen-subset test \
    --source-lang $src_lang --target-lang $tgt_lang \
    --score-reference \
    --batch-size 64 \
    --image-feat-path $image_feat_path/commute/en-$tgt_lang \
    --image-feat-dim $image_feat_dim \
    --output $commute_path/correct.txt \
    > $commute_path/final_score_correct.log

# scoring incorrect Set
python3 fairseq_cli/generate.py data-bin/$data_dir/incorrect \
    --path $model_path \
    --task image_mmt \
    --gen-subset test \
    --source-lang $src_lang --target-lang $tgt_lang \
    --score-reference \
    --batch-size 64 \
    --image-feat-path $image_feat_path/commute/en-$tgt_lang \
    --image-feat-dim $image_feat_dim \
    --output $commute_path/incorrect.txt \
    > $commute_path/final_score_incorrect.log

# compute accuracy
python commute.py $commute_path


done


echo "-------- GOD IS GOOD ! ---------"

