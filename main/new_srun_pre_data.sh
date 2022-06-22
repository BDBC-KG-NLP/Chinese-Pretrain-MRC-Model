export cache_dir=./save_cache                                           # features 目录
export dataset=./datasets/dataset

nohup srun python -u pre_cached.py \
  --feature_dir $cache_dir/merge_random_0430 \
  --data_dir $dataset \
  --train_file merge_random_train_mrc.json.txt  \
  --predict_file merge_random_test_mrc.json.txt \
  --test_file merge_random_test_mrc.json.txt \
  --data_start_point 0 \
  --data_end_point 100000 \
  --data_example_span 100000 > log_pre_cache 2>&1 &
  # --overwrite_cache \
