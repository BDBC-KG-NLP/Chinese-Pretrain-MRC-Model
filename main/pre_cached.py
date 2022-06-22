import argparse
import glob
import json
import logging
import os
import pickle
import random
import timeit
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from tqdm import tqdm, trange

from helper import SquadResult, MyProcessor, squad_convert_examples_to_features_orig
from transformers import (
    BertConfig,
    BertTokenizer,
)


logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, set_type='train', output_examples=False):
    if args.local_rank not in [-1, 0] and set_type == 'train':
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.feature_dir if args.feature_dir else r"./temp"
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            set_type,
            args.data_start_point,
            str(args.max_seq_length),
        ),
    )

    logger.info("Creating features from dataset file at %s", input_dir)

    if not args.data_dir and ((is_evaluate and not args.predict_file) or (not is_evaluate and not args.train_file)):
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

        if args.version_2_with_negative:
            logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

        tfds_examples = tfds.load("squad")
        examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=set_type == 'train')
    else:
        processor = MyProcessor()
        if set_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        elif set_type == 'train':
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
        elif set_type == 'test':
            examples = processor.get_test_examples(args.data_dir, filename=args.test_file)

    start_point = args.data_start_point
    end_point = min(start_point + args.data_example_span, len(examples))
    logger.info("start: %s; end %s, len(examples): %s", start_point, end_point,len(examples))
    examples = examples[start_point:end_point]

    features, dataset = squad_convert_examples_to_features_orig(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=set_type == 'train',
        return_dataset="pt",
        threads=args.threads,
    )

    if args.local_rank in [-1, 0]:
        logger.info("Saving new features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
        logger.info("features len %s, dataset len %s, examples len %s", len(features), len(dataset),len(examples))

    if args.local_rank == 0 and not is_evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    print("main")
    parser = argparse.ArgumentParser()
    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--feature_dir",
        default=None,
        type=str,
        help="The input feature dir. Should contain the cached_features_file for the task."
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="The input test file.",
    )
    parser.add_argument(
        "--data_start_point", 
        type=int, 
        default=0, 
        help="dataset start point for converting examples"
    )
    parser.add_argument(
        "--data_end_point", 
        type=int, 
        default=100000, 
        help="dataset end point for converting examples"
    )
    parser.add_argument(
        "--data_example_span", 
        type=int, 
        default=100000, 
        help="dataset cache span"
    )
    args = parser.parse_args()

    print("parser")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    print("cuda")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    start = args.data_start_point
    end = args.data_end_point
    span = args.data_example_span
    print(start)
    for start_point in range(start, end, span):
        args.data_start_point = start_point
        # dataset, examples, features = load_and_cache_examples(args, tokenizer, set_type='train', output_examples=True)
        dataset, examples, features = load_and_cache_examples(args, tokenizer, set_type='dev', output_examples=True)

if __name__ == "__main__":
    main()
