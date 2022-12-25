DATA_DIR=/root/ccks/BERT-BiLSTM-CRF-NER-pytorch/ccks2021/bio

# BERT_BASE_DIR=bert-base-chinese
# BERT_BASE_DIR=hfl/chinese-bert-wwm-ext
# BERT_BASE_DIR=hfl/chinese-roberta-wwm-ext
# BERT_BASE_DIR=hfl/chinese-macbert-base 
BERT_BASE_DIR=hfl/chinese-electra-180g-base-discriminator
# BERT_BASE_DIR=hfl/chinese-pert-base
# BERT_BASE_DIR=hfl/chinese-lert-base

OUTPUT_DIR=./checkpoints/chinese-electra-180g-base-discriminator
export CUDA_VISIBLE_DEVICES=0

python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 64 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --test_file ${DATA_DIR}/test.txt \
    --learning_rate 3e-5 \
    --train_batch_size 32 \
    --eval_batch_size 512 \
    --num_train_epochs 12 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR \
    --fgm False