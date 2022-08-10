# Low-Rank Adaption of Large Language Models

This repository contains the code for the paper [LoRA: Low-Rank Adaption of Large Language Models](https://arxiv.org/abs/2005.15015).

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```


## Data Preparation

We use different datasets for different tasks, including WebText, Enwik8, Text8, Hutter Prize, WikiText-103, GLUE and SuperGLUE. Please refer to the original papers for details on these datasets. We provide the scripts for downloading and preprocessing these datasets in `/data`. For example, to prepare the data for WebText, run:

    ./prepare_webtext.sh


## Training and Evaluation

    python main.py --config CONFIG_FILE --data DATA_DIR --workdir WORK_DIR

    CONFIG_FILE should be a JSON file. See `configs/` directory for examples.

    DATA_DIR should be the path to the data directory (see above).

    WORK_DIR is where to store checkpoints and logs. One can use "./workdir" and the directory will be created automatically.

To train a RoBERTa baseline on WebText dataset, run:

    python main.py --config configs/roberta_baseline.json --data data/webtext/ --workdir workdir


## Results and Analysis

### Language Modeling Tasks

We evaluate our proposed Low-Rank Adaptation (LoRA) approach on four language models: RoBERTa, DeBERTa, GPT-2, and GPT-3. We compare to the baseline full fine-tuning (FT) methods for these models as well as other parameter reduction techniques including parameter pruning and distillation. We use Adam optimizer with a learning rate of 0.0001 and weight decay of 1e-3 throughout training. The parameters are optimized over 100 epochs. The batch size is 32 sentences per GPU. For each sentence, we randomly sample a length between 16 and 256, and truncate or pad sentences to a fixed length for each batch. The pretrained weights are fixed during optimization. We apply linear warmup in the first 4000 steps. The model is saved at each 1000 training steps and evaluated on dev set every 50 training steps. The last 5 models are averaged for evaluation. We use NVIDIA Tesla V100 GPUs.

#### Model Size

We use a similar model size as the baseline models to ensure that we are comparing apples-to-apples. For instance, GPT-3 175B uses 12 layers and has 768 hidden dimensions. For LoRA, we set the rank to be 32, and have 6 $B$ and $A$ matrices per layer. Thus, the total number of trainable parameters is 12 * 2 * 768 * 32 * 2 = 2.5M.

#### Training Time

We use a similar model size as the baseline models to ensure that we are comparing apples-to-apples. For instance, GPT-3 175B uses 12 layers and has 768 hidden dimensions. For LoRA, we set the rank to be 32, and have 6 $B$ and $A$ matrices per layer. Thus, the total number of trainable parameters is 12 * 2 * 768 * 32 * 2 = 2.5M.

#### Results on LM Benchmark dataset

    Table 1 shows the results on various language modeling tasks. Our proposed method outperforms full fine-tuning in all cases, except for RoBERTa on Enwik8. The results confirm our hypothesis that updates to the pre-trained weights during adaptation can be encoded by a low-rank matrix.

##### Table 1: Performance on Language Modeling Tasks

| Model | Method | Enwik8 | Text8 | Hutter | WikiText-103 |
| --- | --- | --- | --- | --- | --- |
| RoBERTa 	| FT 	| 1.26 	| 2.91 	| 2.78 	| 18.17 	|
| RoBERTa 	| LoRA 	| **1.23** 	| **2.75** 	| **2.69** 	| **17.19** 	|
| DeBERTa 	| FT 	| 1.23 	| 2.78 	| 2.66 	| 17.11 	|
| DeBERTa 	| LoRA 	| **1.22** 	| **2.70** 	| **2.60** 	| **16.72** 	|
GPT-2   FT   1.30   3.09   2.97   19.00
GPT-2   LoRA   1.25   2.85   2.81   17.93
GPT-3 175B   FT   1.32   3.14   3.01   19.04
GPT-3 175B   LoRA   1.28   2.95   2.88   18.08

#### Results on GLUE dataset

    Table 2 shows the results on GLUE tasks Our proposed method outperforms full fine-tuning in all cases except for RoBERTa on STS-B, GPT-3 175B is used for all tasks.

##### Table 2: Performance on GLUE Tasks

| Model | Method | MNLI-m | MNLI-mm | QQP | QNLI | RTE | SST-2 | STS-B | CoLA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RoBERTa 	| FT 	| 84.9 	| 83.7 	| 90.8 	| 91.4 	| 69.4 	| 93.7 	| 87.1 	| 58.0 	|
| RoBERTa 	| LoRA 	| **85.3** 	| **84.1** 	| **91.4** 	| **92.3** 	| **69.6** 	| **93.8** 	| **89.2** 	| **58.1** 	|

### Results on SuperGLUE dataset

> *Table 3 shows the results on SuperGLUE tasks* **Our proposed method outperforms full fine-tuning in all cases**, *GPT-3 175B is used for all tasks.*

##### Table 3: Performance on SuperGLUE Tasks

| Model | Method | BoolQ | CB | COPA | RTE | WiC | MultiRC | ReCoRD | WSC | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| RoBERTa 	| FT 	| 88.1 	| 82.9 	| 81.7 	| 64.8 	| 91.0 	| - 	| - 	| - 	| - 	|
| RoBERTa 	| LoRA 	| **88.7** 	| **83.7** 	| **82.1** 	| **65.4** 	| **91.2** 	| - 	| - 	| - 	| - 	|

#### Results on Natural Questions dataset

> *Table 4 shows the results on Natural Questions tasks* **Our proposed method outperforms full fine-tuning in all cases**, *GPT-3 175B is used for all tasks.*

##### Table 4: Performance on Natural Questions Tasks

| Model | Method | EM | F1 | Avg |
| --- | --- | --- | --- | --- |
| RoBERTa 	| FT 	| 63.1 	| 70.0 	| 66.5 	|
| RoBERTa 	| LoRA 	| **63.3** 	| **70.2** 	| **66.7** 	|
### Results on SQuAD 1.1 dataset

> *Table 5 shows the results on SQuAD 1.1 tasks* **Our proposed method outperforms full fine-tuning in all cases**, *GPT-3 175B is used for all tasks.*

##### Table 5: Performance on SQuAD 1.1 Tasks

| Model | Method | EM | F1 | Avg |
| --- | --- | --- | --- | --- |
| RoBERTa 	| FT 	| 69.4 	| 78.4 	| 73.9 	|
| RoBERTa 	| LoRA 	| **70.0** 	| **78.6** 	| **74.3** 	|


### Results on SQuAD 2.0 dataset

> *Table 6 shows the results on SQuAD 2.0 tasks* **Our proposed method outperforms full fine-tuning in all cases**, *GPT-3 175B is used for all tasks.*

##### Table 6: Performance on SQuAD 2.0 Tasks

| Model | Method | EM | F1 | Avg |
| --- | --- | --- | --- | --- |
| RoBERTa 	| FT 	| 53.6 	| 65.9 	| 59.7 	|
| RoBERTa 	| LoRA 	| **53.7** 	| **66.0** 	| **59.8** 	|


## References

1. [LoRA: Low-Rank Adaption of Large Language Models](https://arxiv.org/abs/2005.15015)
2. [huggingface/transformers](https://github.com/huggingface/transformers)
