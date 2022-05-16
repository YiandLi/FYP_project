# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import os

import numpy
import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append("/".join(rootPath.split("/")[:-2]))
# print(sys.path)

from src.bert_models.training.trainer import Trainer
from src.bert_models.training.utils import init_logger, set_seed
from src.bert_models.training.data_loader import load_and_cache_examples
from transformers import BertTokenizer, BertForPreTraining
import warnings

warnings.filterwarnings("ignore")


def main(args):
    # 设置日志格式
    init_logger()
    logger = logging.getLogger(__name__)
    # 设置所有随机数
    set_seed(args)
    # 初始化Tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path
    )
    
    # 测试Tokenizer
    # tmp_res = tokenizer.tokenize("[MASK]")
    # print("tmp_res: ", tmp_res)
    
    # 得到dataset和sample——weights list
    train_dataset, train_sample_weights = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, dev_sample_weights = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, test_sample_weights = load_and_cache_examples(args, tokenizer, mode="test")
    
    # logger.info("train_dataset: ", len(train_dataset))
    # logger.info("train_sample_weights: ", len(train_sample_weights))
    # logger.info("dev_dataset: ", len(dev_dataset))
    # logger.info("dev_sample_weights: ", len(dev_sample_weights))
    # logger.info("test_dataset: ", len(test_dataset))
    # logger.info("test_sample_weights: ", len(test_sample_weights))
    
    trainer = Trainer(
        args,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        train_sample_weights=train_sample_weights,
        dev_sample_weights=dev_sample_weights,
        test_sample_weights=test_sample_weights,
    )
    
    # logger.info("\n-----------------------------------------------")
    # for k in args.__dict__:
    #     logger.info(k + ": " + str(args.__dict__[k]))
    # logger.info("-----------------------------------------------\n")
    
    if args.do_train:
        trainer.train()
    
    if args.do_eval:
        trainer.load_model()
        # trainer.evaluate("dev")
        trainer.evaluate("test")
    
    if args.do_demo:
        trainer.load_model()
        while True:
            print("\n>>> Entry the comment: ")
            user_input = input()
            if user_input == "exit":
                exit()
            tok_dic = tokenizer(user_input)
            inputs = {'input_ids': torch.tensor([tok_dic['input_ids']]),
                      'attention_mask': torch.tensor([tok_dic['attention_mask']]),
                      'label_ids': torch.tensor([0]),
                      'token_type_ids': torch.tensor([tok_dic['token_type_ids']])}
            outputs = trainer.model(**inputs)
            _, logits = outputs[:2]
            preds = numpy.argmax(logits.detach().numpy(), axis=1)
            print(">>> Emotional tendency:", ["Bad", "Neural", "Good"][preds[0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load models")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--label_file", default=None, type=str, help="Label file for level 1 label")
    parser.add_argument(
        "--label2freq_dir", default=None, type=str,
        help="path to the level 1 label2freq dict;"
    )
    
    parser.add_argument('--seed', type=int, default=1225, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    # Adam中的lr参数
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    
    # 梯度累加则实现了batchsize的变相扩大
    # 如果accumulation_steps为8，则batchsize '变相' 扩大了8倍
    # 是我们这种乞丐实验室解决显存受限的一个不错的trick，使用时需要注意，学习率也要适当放大。
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    
    # ----------------------------------------------------------------------
    
    # sep
    parser.add_argument("--processor_sep", default=",",
                        type=str, help="seperator for dataset.")
    
    parser.add_argument("--embeddings_learning_rate", default=1e-4,
                        type=float, help="The learning rate for Adam.")
    parser.add_argument("--encoder_learning_rate", default=5e-4,
                        type=float, help="The learning rate for Adam.")
    parser.add_argument("--classifier_learning_rate", default=5e-4,
                        type=float, help="The learning rate for Adam.")
    
    # 是否使用 use_lstm
    parser.add_argument("--use_lstm", action="store_true",
                        help="Whether to use lstm.")
    
    parser.add_argument("--patience", default=6, type=int,
                        help="patience for early stopping ")
    parser.add_argument("--metric_key_for_early_stop", default="macro avg__f1-score", type=str,
                        help="metric name for early stopping ")
    
    # 可以选择用不同的aggregator
    parser.add_argument("--aggregator_names", default="bert_pooler", type=str,
                        help="Model type selected in the list: [bert_pooler, slf_attn_pooler, max_pooler, avg_pooler, dr_pooler, ] ")
    
    # 针对不均衡样本
    # 使用不同的loss
    parser.add_argument("--loss_fct_name", type=str, default="ce",
                        help="main loss function: "
                             "(1) 'ce', cross entropy loss; "
                             "(2) 'focal', focal loss; "
                             "(3) 'dice', dice loss;")
    
    parser.add_argument(
        "--focal_loss_gamma", default=2.0, type=float,
        help="gamma in focal loss"
    )
    # use_class_weights
    parser.add_argument("--use_class_weights", action="store_true",
                        help="whether to use class weights; ")
    
    # 是否使用weighted sampler
    parser.add_argument("--use_weighted_sampler", action="store_true",
                        help="use weighted sampler")
    
    parser.add_argument(
        "--model_name_or_path", default=None, type=str,
        help="path to the pretrained lm;"
    )
    
    # 是否使用 multi-sample dropout
    parser.add_argument("--use_ms_dropout", action="store_true",
                        help="whether to use multi-sample dropout; ")
    # 复制多少次
    parser.add_argument(
        "--dropout_num", default=4, type=int,
        help="how many dropout samples to draw;"
    )
    
    # multi-sample dropout 是否求平均
    parser.add_argument(
        "--ms_average", action="store_true",
        help="whether to average the logits from multiple dropout samples or just adding them up;"
    )
    
    # CL
    parser.add_argument("--contrastive_loss", default=None, type=str,
                        help="which contrastive loss to use: "
                             "(1) 'ntxent_loss';"
                             "(2) 'supconloss';")
    parser.add_argument("--what_to_contrast", default=None, type=str,
                        help="what to contrast in each batch: "
                             "(1) 'sample';"
                             "(2) 'sample_and_class_embeddings';")
    parser.add_argument(
        "--ntxent_loss_weight", default=0.5, type=float,
        help="loss weight for ntxent"
    )
    # contrastive_temperature
    parser.add_argument(
        "--contrastive_temperature", default=0.5, type=float,
        help="temperature for contrastive loss"
    )
    
    # rdrop
    parser.add_argument("--use_rdrop", action="store_true",
                        help="")
    # show demo
    parser.add_argument("--do_demo", action="store_true",
                        help="")

    # CAN 概率校准
    parser.add_argument("--do_can_adjustment", action="store_true",
                        help="whether to do probability adjustment, following the CAN paper; ")
    parser.add_argument("--can_alpha", type=float, default=2.0,
                        help=" alpha para of CAN;")
    parser.add_argument("--can_iters", type=int, default=1,
                        help=" num of iterations for CAN")
    parser.add_argument("--can_threshold", type=float, default=0.9,
                        help=" threshold para of CAN;")
    args = parser.parse_args()
    main(args)
