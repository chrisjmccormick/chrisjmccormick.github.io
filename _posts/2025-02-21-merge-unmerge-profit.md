---
layout: post
title:  "Merge, Unmerge, Profit"
date:   2025-02-21 23:50:00 -0800
comments: true
image:
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

While I've been most fascinated by the opportunities for interpretability with the refactored attention equations, my AI research partner has been encouraging me to look at opportunities for model efficiency.

Apparently big LLMs are wasteful with their parameters and known to be low rank.

$W^Q_i$, $W^K_i$ and $W^V_i$, $W^O_i$ are decompositions of two larger matrices... and they're not very good decompositions.

So put them back together, and then do a better job taking them apart.

That's my understanding of what we're doing in this notebook--merging the attention matrices in T5-flan-base, then splitting them apart again with SVD and only keeping half of the 768 singular values.

You reduce parameter count and it doesn't appear to harm performance (the evaluation score exactly matches the base model).

In fact, it seems reasonable to expect that this should _improve_ performance by decluttering the model (especially if you're able to come back and do a little more training!).

---

Huge thanks to Siddharth Sharma for this Notebook. It is his [blog post](https://siddharth-1729-65206.medium.com/compressing-llms-with-low-rank-decomposition-of-attention-matrices-ed13e9e8563a) and Colab Notebook [here](https://colab.research.google.com/drive/1aaveVFlSYwmdjqllHn94PnAo8JBY7-ap#scrollTo=z7LdvUAWSWop).

I "created" this version by handing OpenAI's o3-mini-high model Siddharth's work and asking it to apply the matrix-merge concept.

This Notebook is just a first pass that I haven't had a chance to review or play with much, but it seems to a good initial proof of concept.


# S1. Setup

### Packages


```python
!pip install rouge
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

# install additional dependencies needed for training
!pip install rouge-score tensorboard py7zr

!pip install datasets

!pip install nltk
!pip install evaluate

!pip install --upgrade transformers evaluate
```

### Load Model


```python
from torch import nn
from dataclasses import dataclass
from torch.nn import functional as F
import copy
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# huggingface hub model id
model_id_t5_base = "google/flan-t5-base"
model_t5_base = AutoModelForSeq2SeqLM.from_pretrained(model_id_t5_base, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id_t5_base)

```


    config.json:   0%|          | 0.00/1.40k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/990M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/2.54k [00:00<?, ?B/s]



    spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.42M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/2.20k [00:00<?, ?B/s]


# S2. Merged Model

### Define


```python
###############################################
# Merged Low Rank Decomposition for Attention #
###############################################

@dataclass
class MergedLowRankConfig:
    rank: int
    merge_op: str  # 'qk' for W^P or 'vo' for W^M

class MergedLowRankLayer(nn.Module):
    """
    Given two linear layers (or their weight matrices), compute a low rank
    approximation for their merged matrix.

    For merge_op 'qk': approximates W^P = W^Q * (W^K)^T.
    For merge_op 'vo': approximates W^M = W^V * W^O.
    """
    def __init__(self, config: MergedLowRankConfig, first_layer: nn.Linear, second_layer: nn.Linear):
        super().__init__()
        self.rank = config.rank
        self.merge_op = config.merge_op

        if self.merge_op == 'qk':
            # Merge query and key: note we use (W^K)^T.
            merged = first_layer.weight @ second_layer.weight.T
        elif self.merge_op == 'vo':
            # Merge value and output directly.
            merged = first_layer.weight @ second_layer.weight
        else:
            raise ValueError("merge_op must be 'qk' or 'vo'")

        # Perform SVD on the merged matrix.
        U, S, Vh = torch.linalg.svd(merged)
        S_diag = torch.diag(S)
        # Retain only the top 'rank' singular values.
        self.U = U[:, :self.rank]
        self.S = S_diag[:self.rank, :self.rank]
        self.Vh = Vh[:self.rank, :]

    def forward(self, x):
        # Reconstruct the approximated merged weight.
        approx_weight = self.U @ self.S @ self.Vh
        return F.linear(x, approx_weight)

```


```python
################################################
# Helper Functions (from Siddharth's Notebook) #
################################################

# Finds the module that ends with the target suffix.
def get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

# Recursively set an attribute given a dotted key.
def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

```


```python
###############################################
# Create a Low Rank Replica with Merged Layers#
###############################################

# Create a copy of the original model.
model_t5_base_merged = copy.deepcopy(model_t5_base)
```


```python
# For demonstration, we will create two merged low rank layers:
# 1. For the query-key product (W^P)
# 2. For the value-output product (W^M)
#
# In T5's SelfAttention, the submodules are named "q", "k", "v", and "o".
# We'll replace (or add) two new attributes (e.g., "merged_qk" and "merged_vo")
# in each SelfAttention module.

def replace_attention_with_merged(module, config_qk: MergedLowRankConfig, config_vo: MergedLowRankConfig):
    """
    In a SelfAttention module, add two merged low rank layers:
      - merged_qk: approximates W^P = W^Q * (W^K)^T
      - merged_vo: approximates W^M = W^V * W^O
    """
    # Expect module to have attributes q, k, v, and o.
    # Create merged layers:
    merged_qk = MergedLowRankLayer(config_qk, module.q, module.k)
    merged_vo = MergedLowRankLayer(config_vo, module.v, module.o)

    # Attach these merged layers to the module.
    module.merged_qk = merged_qk
    module.merged_vo = merged_vo

```


```python
# Set up configurations for each merged operation.
config_qk = MergedLowRankConfig(rank=384, merge_op='qk')
config_vo = MergedLowRankConfig(rank=384, merge_op='vo')

# Traverse the model and update SelfAttention modules.
for name, module in model_t5_base_merged.named_modules():
    # For T5, self-attention modules are usually under a submodule named "SelfAttention"
    if isinstance(module, type(model_t5_base.encoder.block[0].layer[0].SelfAttention)):
        replace_attention_with_merged(module, config_qk, config_vo)

```


```python
###############################################
# Verification: Check a Merged Layer's Shapes #
###############################################

# For example, inspect the merged_qk layer of a particular SelfAttention module.
attn_module = model_t5_base_merged.encoder.block[0].layer[0].SelfAttention
print("Merged QK layer approximated weight shape:")

# Reconstruct the merged weight and print its shape.
merged_qk_weight = attn_module.merged_qk.U @ attn_module.merged_qk.S @ attn_module.merged_qk.Vh
print(merged_qk_weight.shape)

print("\nMerged VO layer approximated weight shape:")
merged_vo_weight = attn_module.merged_vo.U @ attn_module.merged_vo.S @ attn_module.merged_vo.Vh
print(merged_vo_weight.shape)

```

    Merged QK layer approximated weight shape:
    torch.Size([768, 768])
    
    Merged VO layer approximated weight shape:
    torch.Size([768, 768])


### Comparisons to Original


```python
device = attn_module.q.weight.device

# For example, project a random vector through the merged_qk layer:
x = torch.rand(1, attn_module.q.in_features, device=device) # batch of one input

original_qk = F.linear(x, attn_module.q.weight @ attn_module.k.weight.T)

approx_qk = attn_module.merged_qk(x)

cos_sim = torch.nn.CosineSimilarity(dim=1)

print("Cosine similarity between original and low-rank merged QK projections:")
print(cos_sim(original_qk, approx_qk).detach().cpu().numpy()[0])
```

    Cosine similarity between original and low-rank merged QK projections:
    0.99593985



```python
# Similarly, you can test merged_vo:

original_vo = F.linear(x, attn_module.v.weight @ attn_module.o.weight)

approx_vo = attn_module.merged_vo(x)

print("Cosine similarity between original and low-rank merged VO projections:")
print(cos_sim(original_vo, approx_vo).detach().cpu().numpy()[0])

```

    Cosine similarity between original and low-rank merged VO projections:
    0.99161536


### Model Size

TODO - I don't think o3's implementation actually removes the unused weights, so the model's size doesn't change.


```python
###############################################
# Save the Original and Merged Models to Disk #
###############################################

# Save the original T5 model and our merged low‐rank model.
model_t5_base.save_pretrained("model_t5_base", from_pt=True)
model_t5_base_merged.save_pretrained("model_t5_base_merged", from_pt=True)

# List file sizes (these commands run in Colab / shell)
!ls -lh model_t5_base/model.safetensors
!ls -lh model_t5_base_merged/model.safetensors

```

    -rw-r--r-- 1 root root 945M Feb 22 06:42 model_t5_base/model.safetensors
    -rw-r--r-- 1 root root 945M Feb 22 06:42 model_t5_base_merged/model.safetensors



```python
###############################################
# Inspect the Architecture of the Merged Model #
###############################################

# Display the overall architecture of our merged model.
print(model_t5_base_merged)

```

    T5ForConditionalGeneration(
      (shared): Embedding(32128, 768)
      (encoder): T5Stack(
        (embed_tokens): Embedding(32128, 768)
        (block): ModuleList(
          (0): T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=768, out_features=768, bias=False)
                  (k): Linear(in_features=768, out_features=768, bias=False)
                  (v): Linear(in_features=768, out_features=768, bias=False)
                  (o): Linear(in_features=768, out_features=768, bias=False)
                  (relative_attention_bias): Embedding(32, 12)
                  (merged_qk): MergedLowRankLayer()
                  (merged_vo): MergedLowRankLayer()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerFF(
                (DenseReluDense): T5DenseGatedActDense(
                  (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                  (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=768, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): NewGELUActivation()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (1-11): 11 x T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=768, out_features=768, bias=False)
                  (k): Linear(in_features=768, out_features=768, bias=False)
                  (v): Linear(in_features=768, out_features=768, bias=False)
                  (o): Linear(in_features=768, out_features=768, bias=False)
                  (merged_qk): MergedLowRankLayer()
                  (merged_vo): MergedLowRankLayer()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerFF(
                (DenseReluDense): T5DenseGatedActDense(
                  (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                  (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=768, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): NewGELUActivation()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (final_layer_norm): T5LayerNorm()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (decoder): T5Stack(
        (embed_tokens): Embedding(32128, 768)
        (block): ModuleList(
          (0): T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=768, out_features=768, bias=False)
                  (k): Linear(in_features=768, out_features=768, bias=False)
                  (v): Linear(in_features=768, out_features=768, bias=False)
                  (o): Linear(in_features=768, out_features=768, bias=False)
                  (relative_attention_bias): Embedding(32, 12)
                  (merged_qk): MergedLowRankLayer()
                  (merged_vo): MergedLowRankLayer()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerCrossAttention(
                (EncDecAttention): T5Attention(
                  (q): Linear(in_features=768, out_features=768, bias=False)
                  (k): Linear(in_features=768, out_features=768, bias=False)
                  (v): Linear(in_features=768, out_features=768, bias=False)
                  (o): Linear(in_features=768, out_features=768, bias=False)
                  (merged_qk): MergedLowRankLayer()
                  (merged_vo): MergedLowRankLayer()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (2): T5LayerFF(
                (DenseReluDense): T5DenseGatedActDense(
                  (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                  (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=768, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): NewGELUActivation()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
          (1-11): 11 x T5Block(
            (layer): ModuleList(
              (0): T5LayerSelfAttention(
                (SelfAttention): T5Attention(
                  (q): Linear(in_features=768, out_features=768, bias=False)
                  (k): Linear(in_features=768, out_features=768, bias=False)
                  (v): Linear(in_features=768, out_features=768, bias=False)
                  (o): Linear(in_features=768, out_features=768, bias=False)
                  (merged_qk): MergedLowRankLayer()
                  (merged_vo): MergedLowRankLayer()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (1): T5LayerCrossAttention(
                (EncDecAttention): T5Attention(
                  (q): Linear(in_features=768, out_features=768, bias=False)
                  (k): Linear(in_features=768, out_features=768, bias=False)
                  (v): Linear(in_features=768, out_features=768, bias=False)
                  (o): Linear(in_features=768, out_features=768, bias=False)
                  (merged_qk): MergedLowRankLayer()
                  (merged_vo): MergedLowRankLayer()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (2): T5LayerFF(
                (DenseReluDense): T5DenseGatedActDense(
                  (wi_0): Linear(in_features=768, out_features=2048, bias=False)
                  (wi_1): Linear(in_features=768, out_features=2048, bias=False)
                  (wo): Linear(in_features=2048, out_features=768, bias=False)
                  (dropout): Dropout(p=0.1, inplace=False)
                  (act): NewGELUActivation()
                )
                (layer_norm): T5LayerNorm()
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (final_layer_norm): T5LayerNorm()
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (lm_head): Linear(in_features=768, out_features=32128, bias=False)
    )



```python
# Look into a specific SelfAttention module, for example from the decoder.
attn_module = model_t5_base_merged.decoder.block[11].layer[0].SelfAttention
print("\nSelfAttention module with merged layers:")
print(attn_module)

# Display the dimensions of the merged low-rank layers.
print("\nMerged QK layer shapes:")
print("U:", attn_module.merged_qk.U.shape,
      "S:", attn_module.merged_qk.S.shape,
      "Vh:", attn_module.merged_qk.Vh.shape)

print("\nMerged VO layer shapes:")
print("U:", attn_module.merged_vo.U.shape,
      "S:", attn_module.merged_vo.S.shape,
      "Vh:", attn_module.merged_vo.Vh.shape)

```

    
    SelfAttention module with merged layers:
    T5Attention(
      (q): Linear(in_features=768, out_features=768, bias=False)
      (k): Linear(in_features=768, out_features=768, bias=False)
      (v): Linear(in_features=768, out_features=768, bias=False)
      (o): Linear(in_features=768, out_features=768, bias=False)
      (merged_qk): MergedLowRankLayer()
      (merged_vo): MergedLowRankLayer()
    )
    
    Merged QK layer shapes:
    U: torch.Size([768, 384]) S: torch.Size([384, 384]) Vh: torch.Size([384, 768])
    
    Merged VO layer shapes:
    U: torch.Size([768, 384]) S: torch.Size([384, 384]) Vh: torch.Size([384, 768])


### Random Projection Test


```python
###############################################
# Random Projection Test for Merged Layers    #
###############################################

# Prepare a random input vector.
# Use the same device as the attention weights.
device = attn_module.q.weight.device
random_vector = torch.rand(768, device=device)  # T5 base has hidden size 768

# For merged QK, compute the original merged projection:
# Original merged QK weight: W^P = q.weight @ k.weight^T
original_merged_qk = F.linear(random_vector.unsqueeze(0),
                              attn_module.q.weight @ attn_module.k.weight.T)
# Get the low-rank approximation via our merged_qk module.
approx_merged_qk = attn_module.merged_qk(random_vector.unsqueeze(0))

# Compute cosine similarity.
cos_sim = torch.nn.CosineSimilarity(dim=1)
print("Cosine similarity for merged QK projection:")
print(cos_sim(original_merged_qk, approx_merged_qk).detach().cpu().numpy()[0])

# For merged VO, compute the original merged projection:
# Original merged VO weight: W^M = v.weight @ o.weight
original_merged_vo = F.linear(random_vector.unsqueeze(0),
                              attn_module.v.weight @ attn_module.o.weight)
# Get the low-rank approximation via our merged_vo module.
approx_merged_vo = attn_module.merged_vo(random_vector.unsqueeze(0))

print("\nCosine similarity for merged VO projection:")
print(cos_sim(original_merged_vo, approx_merged_vo).detach().cpu().numpy()[0])

# Shridarth's Result: 0.9663
```

    Cosine similarity for merged QK projection:
    0.9967425
    
    Cosine similarity for merged VO projection:
    0.9891721


# S2. Evaluation

### Prepare Dataset


```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download('punkt_tab')

```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package punkt_tab to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt_tab.zip.





    True




```python
# Load the Samsum dataset.
raw_datasets = load_dataset("samsum")

# Load the ROUGE metric.
metric = evaluate.load("rouge")
```


    README.md:   0%|          | 0.00/7.04k [00:00<?, ?B/s]



    samsum.py:   0%|          | 0.00/3.36k [00:00<?, ?B/s]


    The repository for samsum contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/samsum.
    You can avoid this prompt in future by passing the argument `trust_remote_code=True`.
    
    Do you wish to run the custom code? [y/N] y



    corpus.7z:   0%|          | 0.00/2.94M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/14732 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/819 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/818 [00:00<?, ? examples/s]



    Downloading builder script:   0%|          | 0.00/6.27k [00:00<?, ?B/s]



```python
# Define a post-processing function for predictions.
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

```


```python

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert predictions to a list of lists of ints.
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().tolist()
    else:
        preds = [list(map(int, p)) for p in preds]

    # Apparently t5 has some tokens that aren't in the vocabulary and it's
    # possible for the model to predict them, which causes the decoder to error.
    # So, we'll clamp the token IDs to the valid range.
    preds = [[min(max(token_id, 0), tokenizer.vocab_size - 1)
              for token_id in pred] for pred in preds]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # For labels, replace -100 with the pad_token_id and convert to a list of ints.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().tolist()
    else:
        labels = [list(map(int, p)) for p in labels]

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds_clean, labels_clean = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=preds_clean, references=labels_clean, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(np.array(p) != tokenizer.pad_token_id) for p in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result



```


```python
# Set up data collator.
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model_t5_base_merged,  # use our merged low-rank model
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Set up Seq2Seq training arguments (for evaluation only).
training_args = Seq2SeqTrainingArguments(
    output_dir="dummy_output",
    predict_with_generate=True,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,  # <-- Add this flag
)
```


```python
def preprocess_function(examples):
    inputs = ["Summarize: " + dialogue for dialogue in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    # For summarization, we also need to tokenize the summaries.
    labels = tokenizer(text_target=examples["summary"], max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the test split and remove the original columns.
tokenized_test = raw_datasets["test"].map(
    preprocess_function, batched=True, remove_columns=["id", "dialogue", "summary"]
)
```


    Map:   0%|          | 0/819 [00:00<?, ? examples/s]


### Baseline


```python
# Create a trainer for our merged model.
trainer_base = Seq2SeqTrainer(
    model=model_t5_base,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

    <ipython-input-25-aff5855060a8>:2: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
      trainer_base = Seq2SeqTrainer(



```python
trainer_base.evaluate()
```

    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance of `EncoderDecoderCache` instead, e.g. `past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`.




<div>

  <progress value='205' max='205' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [205/205 01:42]
</div>



     [34m [1mwandb [0m:  [33mWARNING [0m The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.



    <IPython.core.display.Javascript object>


     [34m [1mwandb [0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
     [34m [1mwandb [0m: You can find your API key in your browser here: https://wandb.ai/authorize
    wandb: Paste an API key from your profile and hit enter:

     ··········


     [34m [1mwandb [0m:  [33mWARNING [0m If you're specifying your api key in code, ensure this code is not shared publicly.
     [34m [1mwandb [0m:  [33mWARNING [0m Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
     [34m [1mwandb [0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc
     [34m [1mwandb [0m: Currently logged in as:  [33mchrismccormick [0m to  [32mhttps://api.wandb.ai [0m. Use  [1m`wandb login --relogin` [0m to force relogin
     [34m [1mwandb [0m: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.



Tracking run with wandb version 0.19.6



Run data is saved locally in <code>/content/wandb/run-20250222_064648-vzl70h5f</code>



Syncing run <strong><a href='https://wandb.ai/chrismccormick/huggingface/runs/vzl70h5f' target="_blank">dummy_output</a></strong> to <a href='https://wandb.ai/chrismccormick/huggingface' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/chrismccormick/huggingface' target="_blank">https://wandb.ai/chrismccormick/huggingface</a>



View run at <a href='https://wandb.ai/chrismccormick/huggingface/runs/vzl70h5f' target="_blank">https://wandb.ai/chrismccormick/huggingface/runs/vzl70h5f</a>





    {'eval_loss': 1.457814335823059,
     'eval_model_preparation_time': 0.0061,
     'eval_rouge1': 46.5218,
     'eval_rouge2': 22.5009,
     'eval_rougeL': 38.7335,
     'eval_rougeLsum': 42.4845,
     'eval_gen_len': 17.2002442002442,
     'eval_runtime': 105.2393,
     'eval_samples_per_second': 7.782,
     'eval_steps_per_second': 1.948}



### Merged


```python
# Create a trainer for our merged model.
trainer_merged = Seq2SeqTrainer(
    model=model_t5_base_merged,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

    <ipython-input-28-a2abcfb4bda0>:2: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
      trainer_merged = Seq2SeqTrainer(



```python
trainer_merged.evaluate()
```



<div>

  <progress value='205' max='205' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [205/205 01:40]
</div>






    {'eval_loss': 1.457814335823059,
     'eval_model_preparation_time': 0.0064,
     'eval_rouge1': 46.5218,
     'eval_rouge2': 22.5009,
     'eval_rougeL': 38.7335,
     'eval_rougeLsum': 42.4845,
     'eval_gen_len': 17.2002442002442,
     'eval_runtime': 102.2639,
     'eval_samples_per_second': 8.009,
     'eval_steps_per_second': 2.005}



Below is a table comparing the metrics for the **Base Model** and the **Merged Model**, with values rounded to three decimal places:

| **Metric**                     | **Base Model** | **Merged Model** |
|:-------------------------------|---------------:|-----------------:|
| eval_loss                      | 1.458          | 1.458            |
| eval_model_preparation_time   | 0.006          | 0.006            |
| eval_rouge1                    | 46.522         | 46.522           |
| eval_rouge2                    | 22.501         | 22.501           |
| eval_rougeL                    | 38.734         | 38.734           |
| eval_rougeLsum                 | 42.485         | 42.485           |
| eval_gen_len                   | 17.200         | 17.200           |
| eval_runtime                   | 105.239        | 102.264          |
| eval_samples_per_second        | 7.782          | 8.009            |
| eval_steps_per_second          | 1.948          | 2.005            |
