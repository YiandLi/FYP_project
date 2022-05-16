# TODO: 定义模型 ClsBERT
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig
from src.bert_models.models.classifier import Classifier, MultiSampleClassifier
from src.bert_models.training.dice_loss import DiceLoss
from src.bert_models.training.focal_loss import FocalLoss
from src.classic_models.models.aggregator_layer import AggregatorLayer
from src.classic_models.models.encoders import BiLSTMEncoder
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
import logging

logger = logging.getLogger(__name__)


class ClsBERT(BertPreTrainedModel):
    """ 定义分类器
        参数：
        过程：
            1。 设置Encoder （args.use_lstm --> self.lstm：BiLSTMEncoder， 默认 self.bert）
            2。 设置aggregator （args.aggregator_names --> self.aggregators：nn.ModuleList()，默认为BertPooler）
            3。 设置分类器 （self.classifier_level_2）
            4。 设置类的权重 （args.use_weighted_sampler --> self.class_weights_level_1，self.class_weights_level_2）
    """
    
    def __init__(self, config,
                 args,
                 label_list,
                 label2freq,
                 ):
        
        # TODO 初始化模型
        super(ClsBERT, self).__init__(config)
        self.args = args
        self.args.hidden_size = config.hidden_size
        self.args.hidden_dim = config.hidden_size
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.num_labels = len(label_list)
        # 如果使用LSTM Encoder，则添加"BiLSTMEncoder模型"作为类属性
        self.lstm = None
        if self.args.use_lstm:
            self.lstm = BiLSTMEncoder(
                args,
            )
        
        # TODO: aggregator层
        # 默认使用 BertPooler ( [cls] )，如果指定用其他的aggregator，则添加到 self.aggregators
        #  [bert_pooler, slf_attn_pooler, max_pooler, avg_pooler, dr_pooler, ]
        self.aggregator_names = self.args.aggregator_names.split(",")
        self.aggregator_names = [w.strip() for w in self.aggregator_names]
        self.aggregator_names = [w for w in self.aggregator_names if w]
        self.aggregators = nn.ModuleList()
        # bert-pooler直接在aggregator_names中判断，其余的放在aggregators列表中
        for aggre_name in self.aggregator_names:
            if aggre_name == "bert_pooler":
                continue
            else:
                aggregator_op = AggregatorLayer(self.args, aggregator_name=aggre_name)
                self.aggregators.append(aggregator_op)
        
        # TODO: 分类层
        # self.classifier_level_1 = Classifier(
        #     args,
        #     input_dim=self.args.hidden_size * len(self.aggregator_names),
        #     num_labels=self.num_labels_level_1,
        # )
        if self.args.use_ms_dropout:
            self.classifier = MultiSampleClassifier(
                args,
                input_dim=self.args.hidden_size,
                num_labels=self.num_labels,
            )
        else:
            self.classifier = Classifier(
                args,
                input_dim=self.args.hidden_size,
                num_labels=self.num_labels,
            )
        
        # TODO：class weights
        class_weights = []
        for i, lab in enumerate(label_list):
            class_weights.append(label2freq[lab])
        class_weights = [1 / w for w in class_weights]
        
        if self.args.use_weighted_sampler:
            class_weights = [math.sqrt(w) for w in class_weights]
        else:
            class_weights = [w for w in class_weights]
        
        self.class_weights = F.softmax(torch.FloatTensor(
            class_weights
        ).to(self.args.device))
    
    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                label_ids=None,
                ):
        # TODO: 得到模型输出
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        sequence_output = outputs[0]
        bert_pooled_output = outputs[1]  # [CLS]
        
        # TODO: 得到池化结果 pooled_outputs = sum([cls], aggregators(sequence_output))
        list_pooled_outpts = []
        if "bert_pooler" in self.aggregator_names:
            list_pooled_outpts.append(bert_pooled_output)
        
        # 其余池化方式的结果放入list_pooled_outpts列表中
        for aggre_op in self.aggregators:
            pooled_outputs_ = aggre_op(sequence_output, mask=attention_mask)
            list_pooled_outpts.append(pooled_outputs_)
        
        pooled_outputs = sum(list_pooled_outpts)
        
        # print("pooled_outputs: ", pooled_outputs.shape)
        
        # TODO: 得到分类结果 logits
        logits = self.classifier(pooled_outputs)
        
        outputs = (logits,)  # add hidden states and attention if they are here
        
        if self.args.use_rdrop:
            og_seed = self.args.seed
            self.args.seed = 42
            from src.bert_models.training.utils import set_seed
            set_seed(self.args)
            # TODO: 得到模型输出
            outputs_fl = self.bert(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
            
            bert_pooled_output_fl = outputs_fl[1]  # [CLS]
            
            list_pooled_outpts_rd = []
            if "bert_pooler" in self.aggregator_names:
                list_pooled_outpts_rd.append(bert_pooled_output_fl)
            
            pooled_outputs_rd = sum(list_pooled_outpts_rd)
            
            logits_rdrop = self.classifier(pooled_outputs_rd)
            
            self.args.seed = og_seed
        # todo ================
        
        # TODO: 计算损失 loss
        if label_ids is not None:
            if self.args.use_class_weights:
                weight = self.class_weights
            else:
                weight = None
            
            if self.args.loss_fct_name == "focal":
                loss_fct = FocalLoss(
                    gamma=self.args.focal_loss_gamma,
                    alpha=weight,
                    reduction="mean"
                )
            elif self.args.loss_fct_name == "dice":
                loss_fct = DiceLoss(
                    with_logits=True,
                    smooth=1.0,
                    ohem_ratio=0,
                    alpha=0.01,
                    square_denominator=True,
                    index_label_position=True,
                    reduction="mean"
                )
            else:
                loss_fct = nn.CrossEntropyLoss(weight=weight)
            
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                label_ids.view(-1)
            )  # 2.4207
            
            if self.args.use_rdrop:
                loss += loss_fct(
                    logits_rdrop.view(-1, self.num_labels),
                    label_ids.view(-1)
                )
                loss += F.kl_div(F.log_softmax(logits_rdrop, dim=-1), F.softmax(logits, dim=-1), reduction='sum')
                loss += F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits_rdrop, dim=-1), reduction='sum')
                loss /= 4
            
            # TODO: 对比学习的损失计算
            if self.args.contrastive_loss is not None:
                if self.args.contrastive_loss == "ntxent_loss":  # https://paperswithcode.com/method/nt-xent
                    loss_fct_contrast = NTXentLoss(
                        temperature=self.args.contrastive_temperature,
                        distance=DotProductSimilarity(),
                    )
                elif self.args.contrastive_loss == "supconloss":
                    loss_fct_contrast = SupConLoss(
                        temperature=self.args.contrastive_temperature,
                        distance=DotProductSimilarity(),
                    )
                else:
                    raise ValueError("unsupported contrastive loss function: {}".format(self.args.use_contrastive_loss))
                
                if self.args.what_to_contrast == "sample":
                    embeddings = pooled_outputs
                    labels = label_ids.view(-1)
                
                elif self.args.what_to_contrast == "sample_and_class_embeddings":
                    embeddings = torch.cat(
                        [pooled_outputs, self.classifier.linear.weight],
                        dim=0
                    )
                    labels = torch.cat(
                        [
                            label_ids.view(-1),
                            torch.arange(0, self.num_labels).to(self.args.device)
                        ],
                        dim=-1
                    )
                else:
                    raise ValueError("unsupported contrastive features: {}".format(self.args.what_to_contrast))
                
                contra_loss = loss_fct_contrast(
                    embeddings,  # [ b_s, 768 ]
                    labels  # [ b_s, 1 ]
                )
                
                loss = loss + \
                       self.args.ntxent_loss_weight * contra_loss
            
            outputs = (loss,) + outputs
        
        return outputs
