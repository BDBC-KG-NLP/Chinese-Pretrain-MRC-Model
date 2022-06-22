export CUDA_VISIBLE_DEVICES=0,1,2,3         # 设置显卡
export pretrain_dir=./output                # 预训练模型目录
export lm=chinese-roberta-wwm-ext-large     # 原始预训练模型名称
export dataset=./datasets                   # 数据目录
export cache_dir=./save_cache               # 缓存目录
export output_dir=output                    # 训练结果目录
export task=train_roberta_mrc               # 训练任务


nohup python main_bert.py \
  --model_type roberta \
  --summary log/$task \
  --feature_dir $cache_dir/merge_random_0430 \
  --model_name_or_path $pretrain_dir/$lm/pytorch_model.bin \
  --config_name $pretrain_dir/$lm/config.json \
  --tokenizer_name $pretrain_dir/$lm/vocab.txt \
  --threads 12 \
  --warmup_ratio 0.1 \
  --logging_ratio 0.1 \
  --save_ratio 0.1 \
  --do_lower_case \
  --data_dir $dataset \
  --train_file merge_random_train_mrc.json.txt \
  --predict_file merge_random_test_mrc.json.txt \
  --test_file merge_random_test_mrc.json.txt \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16  \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 32 \
  --max_answer_length 128 \
  --n_best_size 10 \
  --output_dir $output_dir/$task \
  --do_fgm \
  --gc \
  --do_train \
  --overwrite_output_dir \
  --evaluate_during_training \
  --version_2_with_negative \ 
      > log_train_roberta_mrc.out 2>&1 &
  # --do_test \
  # --overwrite_cache \
  # --data_start_point 0 \
  # --data_end_point 1000000 \
  # --data_example_span 100000 \
