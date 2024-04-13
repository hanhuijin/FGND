#!/bin/bash

for i in {1..20}
do
  python ./src/train_new.py \
      --debug \
      --datapath data// \
      --seed $i \
      --dataset cora \
      --type densegcn \
      --nhiddenlayer 1 \
      --nbaseblocklayer 14 \
      --hidden 128 \
      --epoch 400 \
      --lr 0.008 \
      --weight_decay 0.0005 \
      --early_stopping 400 \
      --sampling_percent 0.2 \
      --dropout 0.8 \
      --normalization AugNormAdj \
       \
done