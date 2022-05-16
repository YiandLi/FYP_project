# TODO 设置程序所有的默认设置
import os
import random
import logging

import torch
import numpy as np
from sklearn.metrics import classification_report


def get_labels(label_file):
    """每行表示一个label，所以逐行读取得到label列表
    """
    return [label.strip() for label in open(label_file, 'r', encoding='utf-8')]


def init_logger():
    """ 设置日志格式
    """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    """ 根据args.seed 设置所有随机种子
    """
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if not args.no_cuda and torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    classification_report_dict = classification_report(intent_labels, intent_preds, output_dict=True)
    for key0, val0 in classification_report_dict.items():
        if isinstance(val0, dict):
            for key1, val1 in val0.items():
                results[key0 + "__" + key1] = val1
        else:
            results[key0] = val0
    return results


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]
