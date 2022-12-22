# model experiment
BERT_BASE_DIR=bert-base-chinese
# BERT_BASE_DIR=./pretrain_model/xlm-roberta-base
# BERT_BASE_DIR=xlm-roberta-base

DATA_DIR=/root/ccks/BERT-BiLSTM-CRF-NER-pytorch/ccks2021/bio
OUTPUT_DIR=./checkpoints/debug
export CUDA_VISIBLE_DEVICES=0

python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 128 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --train_batch_size 32 \
    --eval_batch_size 512 \
    --num_train_epochs 10 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR