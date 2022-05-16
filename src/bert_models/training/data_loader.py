import os
import copy
import json
import logging

import math

import torch
from torch.utils.data import TensorDataset
from src.bert_models.training.utils import get_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """
    
    def __init__(self, guid, words,
                 label_level=None,):
        self.guid = guid
        self.words = words
        self.label_level = label_level
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids,
                 attention_mask,
                 token_type_ids,
                 label_id,):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """Processor for the BERT data set """
    
    def __init__(self, args):
        self.args = args
        self.labels_level = get_labels(args.label_file)
    
    @classmethod
    def _read_file(cls, input_file, skip_first_line=False):
        """ 每一行存入列表 lines
        """
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if skip_first_line:
                    if i == 0:
                        continue
                
                lines.append(line.strip())
            return lines
    
    def _create_examples(self, lines, set_type, sep=","):
        """
        根据 行内容列表 创建样例
        ----------
        lines：行列表
        set_type："train, dev, test"
        sep：分隔符号","
        -------
        返回实例列表
        """
        examples = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            line = line.split(sep)
            
            # id："train-01"
            id_ = line[0]
            guid = "%s-%s" % (set_type, id_)
            
            # 1. input_text
            words = line[1].split(" ")
            words = [w.strip() for w in words if len(w.strip()) > 0]
            
            # 标签
            # 如果是测试集，则设置标签为0
            if set_type == "test":
                label_level = 0
            # 如果不是测试集则创建InputExample类，并加入examples列表
            else:
                # 如果不符合数据要求，则打印出来
                if not len(line) == 3:
                    print(line)
                
                label_name = line[2]
                
                label_level = self.labels_level.index(label_name)
            
            examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    label_level=label_level,
                )
            )
        return examples
    
    def get_examples(self, mode):
        """ 根据args中的文件地址和传入设定mode创建实例列表
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, "{}.txt".format(mode))
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(lines=self._read_file(data_path),
                                     set_type=mode, sep=self.args.processor_sep)



def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 label2freq=None,
                                 label_list=None,
                                 ):
    if max_seq_len < 3:
        print("max_seq_len should be larger than 3 because of two special token and here is {}".format(max_seq_len))
        exit()
    
    # Setting based on the current models type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    
    features = []
    sample_weights = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        # Tokenize
        tokens = tokenizer.tokenize(" ".join(example.words))
        # print(tokens, " ".join(example.words), )
        
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
        
        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        
        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        # 求需要padding的长度
        padding_length = max_seq_len - len(input_ids)
        # 在input_ids后面加上padding id
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        # 得到标签索引
        label_id = int(example.label_level)
        
        # sample weight：对每个样本加权重，即样本数多的类别样本权重低
        # 1/频率平方根
        samp_weight = math.sqrt(
            1 / label2freq[label_list[label_id]]
        )
        
        sample_weights.append(samp_weight)
        
        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label_level: %s (id = %d)" % (example.label_level, label_id))
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
            )
        )
    
    return features, sample_weights


def load_and_cache_examples(args, tokenizer, mode):
    processor = Processor(args)
    
    # level 标签的频次
    # { "1": 4444, ... }
    label2freq = json.load(
        open(args.label2freq_dir, "r", encoding="utf-8"),
    )
    
    # 加载label list
    label_list = get_labels(args.label_file)
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            args.model_name_or_path.split("/")[-1],
            args.max_seq_len
        )
    )
    cached_sampling_weights_file = os.path.join(
        args.data_dir,
        'cached_sampling_weights_{}_{}_{}_{}'.format(
            mode,
            args.task,
            args.model_name_or_path.split("/")[-1],
            args.max_seq_len
        )
    )
    
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        sampling_weights = torch.load(cached_sampling_weights_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
        
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        # 得到转化的输入列表，和sample weight权值列表
        features, sampling_weights = convert_examples_to_features(
            examples,
            args.max_seq_len,
            tokenizer,
            label2freq=label2freq,
            label_list=label_list,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info("Saving features into cached file %s", cached_sampling_weights_file)
        torch.save(sampling_weights, cached_sampling_weights_file)
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
    
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_label_id,
    )
    
    return dataset, sampling_weights
