# TODO: 设置训练器
import json
import logging
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import BertConfig
from src.bert_models.models import ClsBERT
from src.bert_models.training.can import can_adjustment
from src.bert_models.training.utils import compute_metrics, get_labels

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None,
                 train_sample_weights=None,
                 dev_sample_weights=None,
                 test_sample_weights=None,
                 ):
        # -- 通过传入数据进行初始化
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_sample_weights = train_sample_weights
        self.dev_sample_weights = dev_sample_weights
        self.test_sample_weights = test_sample_weights
        
        # -- 通过args的文件地址 得到两级标签列表
        self.label_list = get_labels(args.label_file)
        
        # -- 通过args的标签频率地址 得到标签频率表 --》为了后面weight sampling
        self.label2freq = json.load(
            open(args.label2freq_dir, "r", encoding="utf-8"),
        
        )
        
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.args.device = self.device
        
        # -- 通过设置类得到 config和model类对象 BertConfig, ClsBERT
        self.config_class, self.model_class = BertConfig, ClsBERT
        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        finetuning_task=args.task,
                                                        gradient_checkpointing=True  # 以计算时间换内存的方式，显著减小模型训练对GPU的占用
                                                        # batch_size * max_length
                                                        )
        
        # --！！ 创建模型 model_class = 'ClsBERT'
        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config,  # 初始化BertModel
            args=args,
            label_list=self.label_list,
            label2freq=self.label2freq,
        )
        self.model.to(self.device)
        
        # for early  stopping
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = 0.0
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False
    
    def train(self):
        
        # TODO：创建train_dataloader
        # args.use_weighted_sampler --> 是否使用 train_sample_weights
        #   使用函数 WeightedRandomSampler(samples_weight, samples_num)生成采样index列表。
        #       samples_weight的数量等于我们训练集总样本的数量
        #       samples_weight的每一项代表该样本种类占总样本的比例的倒数
        #       samples_num 为我们想采集多少个样本，可以重复采集
        #   根据samples_weight生成长度为samples_num的列表，元素为[0,..,len(samples_weight)-1]。
        if self.args.use_weighted_sampler:
            train_sampler = WeightedRandomSampler(
                self.train_sample_weights,
                len(self.train_sample_weights),
            )
        else:
            train_sampler = RandomSampler(self.train_dataset)
        
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,  # 定义取batch的方法，是一个迭代器， 每次生成一个index列表
            batch_size=self.args.train_batch_size
        )
        
        # TODO：定义最大步数t_total
        # gradient_accumulation_steps = default(1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            # 总步数 = epochs * 数据集切分得到的batch数量 // 1
            # gradient_accumulation_steps(default=1)
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
        # TODO：定义调度器和优化器
        # for n, p in self.model.named_parameters():
        #     print(n)
        
        optimizer_grouped_parameters = []
        # embedding部分
        embeddings_params = list(self.model.bert.embeddings.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in embeddings_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.embeddings_learning_rate,
             },
            {'params': [p for n, p in embeddings_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.embeddings_learning_rate,
             }
        ]
        
        # encoder + bert_pooler 部分
        encoder_params = list(self.model.bert.encoder.named_parameters())
        if "bert_pooler" in self.model.aggregator_names:
            encoder_params = encoder_params + list(self.model.bert.pooler.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.encoder_learning_rate,
             },
            {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.encoder_learning_rate,
             }
        ]
        
        # linear层 + 初始化的aggregator部分
        classifier_params = list(self.model.classifier.named_parameters()) + \
                            list(self.model.aggregators.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay,
             "lr": self.args.classifier_learning_rate,
             },
            {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             "lr": self.args.classifier_learning_rate,
             }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        
        if self.args.do_swa:
            from torchcontrib.optim import SWA
            start_step = t_total * 3 // 4
            freq_step = len(train_dataloader)
            optimizer = SWA(optimizer, swa_start=start_step, swa_freq=freq_step, swa_lr=5e-5)
        
        # scheduler = get_linear_schedule_with_warmup(
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )
        
        # TODO: Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)
        
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        all_f1 = []
        for i in range(len(self.label_list) + 1):
            all_f1.append([])
        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'label_ids': batch[3],
                          }
                
                # 得到 损失+输出
                outputs = self.model(**inputs)
                loss = outputs[0]
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()
                
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    optimizer.step()  # 更新 权重参数
                    scheduler.step()  # 更新 学习率参数
                    self.model.zero_grad()
                    global_step += 1
                    
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate("dev")
                        for i in range(len(all_f1) - 1):
                            all_f1[i].append(results[str(i) + '__f1-score'])
                        all_f1[-1].append(results['macro avg__f1-score'])
                        
                        if results.get(self.metric_key_for_early_stop, 0.0) > self.best_score:
                            self.best_score = results.get(self.metric_key_for_early_stop)
                            self.early_stopping_counter = 0
                            if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                                self.save_model()
                        
                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= self.patience:
                                self.do_early_stop = True
                        
                        logger.info("*" * 50)
                        logger.info("current step score for metric_key_for_early_stop: {}".format(
                            results.get(self.metric_key_for_early_stop, 0.0)))
                        logger.info("best score for metric_key_for_early_stop: {}".format(self.best_score))
                        logger.info("*" * 50)
                        
                        if self.do_early_stop:
                            logger.info("Early Stop !!")
                            break
                
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
                
                if self.do_early_stop:
                    epoch_iterator.close()
                    break
            
            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
            
            if self.do_early_stop:
                epoch_iterator.close()
                break
        
        if self.args.do_swa:
            optimizer.swap_swa_sgd()
        
        if global_step == 0:
            logger.info("global_step is zero, training epoches may be zero")
            exit()
        
        self.draw_f1(all_f1)  # 画出 f1 图像
        return global_step, tr_loss / global_step, all_f1
    
    def draw_f1(self, all_f1):
        """
        设置图例 plt.plot 加参数 label=" " ； 然后最后加个 plt.legend(loc='best')
        设置横坐标，加参数横坐标个数和其标注 plt.xticks(range(3), ("asda", "asd", "asda"))
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5), dpi=200)
        for i in range(len(all_f1)):
            plt.plot(all_f1[i], '-.')
        plt.plot(all_f1[-1], 'r')
        plt.xticks([])
        plt.ylabel('F1 value', fontsize=14)
        plt.savefig("f1_score.jpg")
        plt.show()
    
    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
        
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        
        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        
        self.model.eval()
        
        logger.info("Do evaluating for {} batches".format(len(eval_dataloader)))
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'label_ids': batch[3],
                          'token_type_ids': batch[2]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            
            # label prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['label_ids'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['label_ids'].detach().cpu().numpy(), axis=0)
        
        eval_loss = eval_loss / nb_eval_steps
        best_f1_score = -1
        if self.args.do_can_adjustment:
            #######################
            # implement CAN method: CAN概率校准方法
            #######################
            
            # 网格搜索
            list_can_alphas = [0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            list_can_threshold = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
            
            for alpha in list_can_alphas:
                for thres in list_can_threshold:
                    
                    preds_adjusted = can_adjustment(
                        preds,
                        # 默认训练集样本频率为先验
                        priors=np.array([1, 1, 1], dtype=np.float),
                        alpha=alpha,
                        iters=1,
                        threshold=thres,
                    )
                    
                    # label prediction result
                    preds_adjusted = np.argmax(preds_adjusted, axis=1)
                    
                    results = compute_metrics(preds_adjusted, out_label_ids)
                    logger.info("***** Eval results with CAN*****")
                    logger.info("  self.args.can_alpha = %s", alpha)
                    logger.info("  self.args.can_iters = %s", 1)
                    logger.info("  self.args.can_threshold = %s", thres)
                    for key in sorted(results.keys()):
                        if key != self.metric_key_for_early_stop:
                            continue
                        logger.info("  %s = %s", key, str(results[key]))
                        if best_f1_score < results[key]:
                            best_f1_score = results[key]
                    
                    logger.info("***** Eval results *****")
        
        results_dict = {
            "loss": eval_loss
        }
        
        # label prediction result
        preds = np.argmax(preds, axis=1)
        
        results = compute_metrics(preds, out_label_ids)
        
        # 这里改metric for early stop 的名称
        for key_, val_ in results.items():
            results_dict[key_] = val_
            results_dict[self.metric_key_for_early_stop] = best_f1_score
        
        logger.info("***** Eval results *****")
        
        if mode == "dev":
            for key in sorted(results_dict.keys()):
                logger.info("  %s = %s", key, str(results_dict[key]))
        
        # 将预测结果写入文件
        if mode == "test":
            f_out = open(os.path.join(self.args.model_dir, "prediction.csv"), "w", encoding="utf-8")
            f_out.write("id,label" + "\n")
            
            list_preds = preds.tolist()
            for i, pred_label_id in enumerate(list_preds):
                pred_label_name = self.label_list[pred_label_id]
                trans_map = [" ", "bad", "","neural", "","good"]
                
                f_out.write("%s,%s" % (str(i), (trans_map[int(pred_label_name)])) + "\n")
        
        return results_dict
    
    def save_model(self):
        # Save models checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)
        
        # TODO: 保存.args文件
        args_path = os.path.join(self.args.model_dir, "args.json")
        if not os.path.exists(args_path):
            with open(args_path, 'w') as f:
                json.dump(self.args.__dict__, f)
        
        logger.info("Saving models checkpoint to %s", self.args.model_dir)
    
    def load_model(self):
        # Check whether models exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        
        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          config=self.config,
                                                          args=self.args,
                                                          label_list=self.label_list,
                                                          label2freq=self.label2freq,
                                                          )
            self.model.to(self.device)
            # logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some models files might be missing...")
