export CUDA_VISIBLE_DEVICES=0,1       # 设置显卡
export pretrain_dir=./output          # 模型目录
export lm=train_roberta_mrc           # 模型模型名称
export dataset=./datasets             # 数据文件夹
export cache_dir=./save_cache         # 数据缓存文件夹
export output=./output                # 测试结果目录
export task=test_model                # 训练任务

nohup python main_bert.py \
  --model_type bert \
  --summary log/$task \
  --feature_dir $cache_dir \
  --model_name_or_path $pretrain_dir/$lm/pytorch_model.bin \
  --config_name $pretrain_dir/$lm/config.json \
  --tokenizer_name $pretrain_dir/$lm/vocab.txt  \
  --threads 24 \
  --logging_ratio 0.1 \
  --do_lower_case \
  --data_dir $dataset \
  --predict_file test.txt \
  --test_prob_file test_prob.pkl \
  --per_gpu_eval_batch_size 16 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 32 \
  --max_answer_length 128 \
  --n_best_size 10 \
  --do_eval \
  --version_2_with_negative \
  --output_dir $output/$task > log_test_.out 2>&1 &
 # --overwrite_cache \
