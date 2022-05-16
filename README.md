## System Structure
```
./
├── README.md
├── datasets # please download from source
├── experiments # save the experiments result and model 
└── src
    ├── bert_models
    │   ├── models
    │   │   ├── classifier.py
    │   │   ├── demo.py  # for single prediction
    │   │   ├── file_utils.py
    │   │   └── modeling_bert.py
    │   └── training
    │       ├── can.py
    │       ├── data_loader.py
    │       ├── dice_loss.py
    │       ├── focal_loss.py
    │       ├── main.py
    │       ├── trainer.py
    │       └── utils.py
    └── classic_models
        ├── models
        │   ├── aggregator_layer.py
        │   ├── classifier.py
        │   ├── embedding_layer.py
        │   ├── encoders.py
        │   └── modeling.py
        ├── modules
        │   ├── avg_pool.py
        │   ├── child_dynamic_routing.py
        │   ├── child_rnns.py
        │   ├── child_sep_conv.py
        │   ├── identity_op.py
        │   ├── max_pool.py
        │   ├── null_op.py
        │   ├── positional_embedding.py
        │   └── self_attn_pool.py
        ├── training
        │   ├── data_loader.py
        │   ├── focal_loss.py
        │   ├── main.py
        │   ├── trainer.py
        │   └── utils.py
        └── utils
            ├── model_utils.py
            └── text_utils.py

```

## Data set
download the file from [https://www.aliyundrive.com/s/dWXFzwNYexe](https://www.aliyundrive.com/s/dWXFzwNYexe) and replace `datasets` dict.

## Other Model
More model: [https://huggingface.co/models](https://huggingface.co/models)

## how to use
Given a parameter example:
```
--task FYP
--model_dir ./experiments/outputs/FYP_ouput
--data_dir datasets/fyp_data/splits_135/fold_0
--label_file datasets/fyp_data/labels_level_135.txt
--label2freq_dir datasets/fyp_data/label2freq_level.json
--metric_key_for_early_stop "macroavg__f1-score"
--model_name_or_path bert-base-uncased
--loss_fct_name ce
--seed 1234
--embeddings_learning_rate 3e-5
--encoder_learning_rate 5e-5
--classifier_learning_rate 9e-4
--train_batch_size 4
--num_train_epochs 1
--warmup_steps 370
--max_seq_len 4
--dropout_rate 0.15
--logging_steps 2
--patience 100
--do_train
``` 

The information of each parameter could been seen in `main.py` or `-help` order.
   
