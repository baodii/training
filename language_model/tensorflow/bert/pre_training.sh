# export CUDA_VISIBLE_DEVICES=1
TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
python run_pretraining.py \
  --bert_config_file=./cleanup_scripts/wiki/bert_config.json \
  --output_dir=./tmp/output/ \
  --input_file="./cleanup_scripts/tfrecord/part*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=./cleanup_scripts/wiki/tf1_ckpt/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --num_gpus=8 \
  --train_batch_size=24

