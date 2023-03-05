#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
dataset=electricity
pred_len=96
wavelet=haar
data=custom
wavelet_j=3

node_dim=40
topk=6
n_gnn_layer=3
batch_size=32
dropout=0.30
learning_rate=0.001

seq_len=96
model=WaveForM


if [ $dataset = weather ]; then
  n_point=21
elif [ $dataset = electricity ]; then
  n_point=321
elif [ $dataset = traffic ]; then
  n_point=862
elif [ $dataset = temperature ]; then
  n_point=6
elif [ $dataset = solar ]; then
  n_point=137
fi

exp_description=${dataset:0:3}_"$model"_"$seq_len"_"$pred_len"

nohup python -u run.py \
  --model_id "$dataset"_"$exp_description" \
  --model $model \
  --data $data \
  --root_path ./dataset/"$dataset"/ \
  --data_path "$dataset".csv \
  --checkpoints ./checkpoints/ \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --n_points $n_point \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --itr 1 \
  --batch_size $batch_size \
  --node_dim $node_dim \
  --subgraph_size $topk \
  --n_gnn_layer $n_gnn_layer \
  --wavelet $wavelet \
  --wavelet_j $wavelet_j \
  --des exp > out_"$exp_description" &
