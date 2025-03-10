---
layout: post
title:  "Inspecting Head Behavior through Pattern Projections"
date:   2025-02-20 17:00:00 -0800
comments: true
image:
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

In this post, I explore a novel way to interpret attention head functionality in transformer models by reformulating the Query and Key projections into a single matrix, $W^P = (W^Q)^T W^K$.

This formulation allows us to view each head as a pattern extractor operating in model space. By projecting an input embedding onto $W^P$, we obtain a “pattern vector” that can be directly compared to vocabulary embeddings using cosine similarity.

Compared to looking at attention scores for example inputs, this technique has a nice property of allowing us to look at what words the head _would_ attend to if they were present in the input. i.e., we can learn about what words a head would most attend to if they were present.


The goal in this post is just to demonstrate the potential of the approach through some initial experiments.

The experiments are:

**BERT Without Context**

I applied this pattern technique directly to the heads in the early layers of BERT, by picking arbitrary single words to use as input to all of the heads.

**BERT PEV Vectors**

Look at the heads' inherent biases towards positions by comparing the position encoding vectors to a head pattern.

**Token Evolution in GPT-2**

This section is mostly educational, demonstrating how a Decoder's prediction embedding gradually becomes more and more similar to the eventual predicted word.

It was an important stepping stone, though, for confirming that I could in fact compare the hidden state meaningfully at any layer to the vocabulary embeddings. The key was to apply the final layer normalization to the hidden state before taking the dot product with the vocabulary.

**GPT-2 With Context**

Having confirmed that we can put the hidden states into the same embedding space as the vocabulary, I was able to perform this head pattern analysis on all layers and heads of GPT-2. The initial results seem meaningful enough to believe that the approach works, but I haven't tried analyzing an example in much detail yet.

**Understanding and Interpretability**

I'm hoping this technique will be a valuable teaching tool, and perhaps lead toward more advanced interpretability research.

Let's dive into the code and see what insights we can uncover!

# ▂▂▂▂▂▂▂▂▂▂▂▂

# S1. BERT Heads

A naive approach seems to work well for at least the first few layers of  BERT, where, even without adding position information or running the tokens through the model, we can simply compare the head patterns back to the input embeddings with cosine similarity.

Let's check out some results!


### 1.1. Analysis Functions

I'm doing this analysis first on bert-base-uncased, due to its simplicity and "small" size.

First, load the model and extract its embedding layer.


```python
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import scipy.spatial
import pandas as pd

# Load BERT-base model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Extract embedding layer
embedding_layer = model.embeddings.word_embeddings.weight.detach().cpu().numpy()

```


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]


Next, a function for constructing the Pattern projection matrix $W^P_i$ for a given head in a given layer of BERT.


```python
def get_BERT_WP(layer, head):
    """
    Computes the pattern projection matrix W^P for a given layer and head.
    """
    # Extract query (W_Q) and key (W_K) weights from the specified layer
    W_Q = model.encoder.layer[layer].attention.self.query.weight.detach().cpu().numpy()
    W_K = model.encoder.layer[layer].attention.self.key.weight.detach().cpu().numpy()

    # Determine the size of each head's projection
    head_size = W_Q.shape[0] // num_heads

    # Slice out the parameters for the chosen head
    W_Q_i = W_Q[head * head_size : (head + 1) * head_size, :]
    W_K_i = W_K[head * head_size : (head + 1) * head_size, :]

    # Compute the pattern matrix: (W_Q_i)^T dot W_K_i
    W_P_i = np.dot(W_Q_i.T, W_K_i)
    return W_P_i
```

Similarity metrics--cosine similarity is what seems to work, I guess the  vectors need to be normalized.


```python
# Compute dot product similarity
def dot_product_similarity(vec, matrix):
    return np.dot(matrix, vec)

# Compute cosine similarity
def cosine_similarity(vec, matrix):
    return 1 - scipy.spatial.distance.cdist([vec], matrix, metric="cosine")[0]

```

For a given head and input word, find the closest matching vocabulary embeddings to the resulting pattern vector.


```python
def find_head_matches(W_P_i, input_word, k=15):

    # Some tokens may get split up...
    tokens = tokenizer.tokenize(input_word)

    if len(tokens) != 1:
        print(f"Warning: The word '{input_word}' was tokenized into multiple tokens: {tokens}. Using the first token.")

    token = tokens[0]

    # Convert the token to its corresponding ID in the vocabulary.
    word_id = tokenizer.convert_tokens_to_ids(token)

    # Extract their original embeddings
    word_emb = np.array(embedding_layer[word_id])

    # Project the token embedding to get the pattern vector.
    pattern = np.dot(word_emb, W_P_i)

    # Caclulate cosine similarities.
    similarities = cosine_similarity(pattern, embedding_layer)

    # Sort to retrieve the top k.
    top_indices = similarities.argsort()[-k:][::-1]

    top_words = []

    # Construct a list of tuples to return, (word_str, similarity)
    for idx in top_indices:
        # Convert the vocabulary index back into a token string.
        word_str = tokenizer.convert_ids_to_tokens(int(idx))

        top_words.append((word_str, similarities[idx]))

    return top_words
```

### 1.2. Probing Layers 1 - 4

I had GPT suggest a few words to try, and found the closest matching embeddings for all of the heads in the first four layers.

Note: I'm using zero-based layer numbers here to discuss the heads, but I should probably change that?

Skimming through, here were some interesting examples:


**Disambiguating Heads**

These all seem to be examples where the head is looking for a context word to clarify the right meaning of the input word.

For example, when given the word **run**, these two heads appear to look for context for it:

<br/>

_Layer 0, head 1_

**run** --> `election, innings, theatre, mayor, sales, reelected, theaters, selling, wickets, theater, commercials, electoral, elections, gallons, elected`

* There are multiple contexts that it appears to be looking for: running for election, running a production, a run in baseball or cricket...

<br/>

_Layer 1, head 3_

**run** --> `pitcher, home, inning, goalkeeper, wickets, schumacher, pitchers, bowler, baseball, wicket, shortstop, nfl, mlb, ##holder, drivers`

* Seems focused on associating run with sports.

<br/>

_Layer 0, head 2_

**dog** --> `hot, sent, watch, guard, radio, guide, send, hound, unsuccessfully, voice, neck, sends, success, feel, mas`

**bed** --> `truck      river    playoff      creek      speed       flow    vehicle    lecture       fish     stream     flower    thunder      drain     narrow        dry`

**drive** --> `disk       disc       leaf      flash       club   magnetic      wheel       gene    reverse        rip       data      blood commercially    serpent    captive`


**Unsure...**

Not sure how to describe these associations...

<br/>

_Layer 0, head 3_

**happy** --> `make, making, made, makes, people, not, ##made, women, ##ria, something, felt, city, dee, men, paper`

<br/>

_Layer 3, head 8_

**happy** --> `picture, faces, genesis, aftermath, emotional, expression, concurrently, pictures, expressions, emotion, hearts, account, jasper, mental, disorders`

**Special Tokens**

Many of the results were a pattern closely resembling the special tokens. (Matching the finding in "What Does BERT Look At?", [here](https://arxiv.org/pdf/1906.04341))

For example, layer 1 head 1,

**couch** --> `[CLS], [MASK], [SEP], ##⁄, ##rricular, ##fully, ##vances, ##ostal, pmid, ##⋅, ##atable, ##tained, ##lessly, ##genase, ##ingly`
            
With a big drop-off in cosine similarities: 0.55, 0.29, 0.23, 0.17, 0.14, ...


**Self-Attending**

Some head patterns matched the input word and its synonyms, implying the head is attending to the input token.

_Layer 2, head 6_

**couch** --> `couch, sofa, lagoon, ##ppel`

With cosine similarities: 0.23, 0.2, 0.17, 0.17

**Code**

The below code runs this analysis for the specified words and every head in the specified layers.


```python
# Select a sample set of words
words = ["couch", "dog", "run", "happy"]

# Layers to process (0 and 1)
layers = [0, 1, 2, 3]

# Look at all heads.
num_heads = model.config.num_attention_heads

# Store results
results = []

# For each of the layers / heads / words...
for layer in layers:
    for head in range(num_heads):

        # Get the Pattern projection matrix for this head.
        W_P_i = get_BERT_WP(layer, head)

        # For each of the words...
        for word in words:

            # Find the matching word embeddings.
            matches = find_head_matches(W_P_i, word, k=15)

            # Separate the words and scores.
            top_k_strs = ""
            top_k_sims = ""

            # Turn them into strings
            for word_str, sim in matches:
                top_k_strs += f"{word_str:>8}, "
                top_k_sims += f"{sim:.2}, "

            # Add the result as a row.
            results.append({
                "Word": word,
                "Layer": layer,
                "Head": head,
                "Top-k": top_k_strs,
                "Scores": top_k_sims
            })

# Convert results to DataFrame and display
df_results = pd.DataFrame(results)

# Set pandas precision to 3 decimal points
pd.options.display.float_format = '{:.3f}'.format

# Force pandas to display the full table.
if False:
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

display(df_results)

df_results.to_csv("bert_head_results.csv")
```



  <div id="df-bdac3328-52b0-427c-9eab-82ae8a2af8b2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Word</th>
      <th>Layer</th>
      <th>Head</th>
      <th>Top-k</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>couch</td>
      <td>0</td>
      <td>0</td>
      <td>for,     with,        ",       on,       ...</td>
      <td>0.16, 0.15, 0.14, 0.14, 0.13, 0.13, 0.12, 0.12...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dog</td>
      <td>0</td>
      <td>0</td>
      <td>engine,    roman,     rome,     html,  vehic...</td>
      <td>0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.13, 0.13...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>run</td>
      <td>0</td>
      <td>0</td>
      <td>an,    [SEP],      the,   [MASK],       ...</td>
      <td>0.14, 0.14, 0.14, 0.13, 0.12, 0.11, 0.1, 0.1, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>happy</td>
      <td>0</td>
      <td>0</td>
      <td>[MASK],        -,       to,       on,       ...</td>
      <td>0.27, 0.16, 0.16, 0.16, 0.14, 0.14, 0.14, 0.14...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>couch</td>
      <td>0</td>
      <td>1</td>
      <td>[MASK],    [CLS],        -,        ,,       ...</td>
      <td>0.44, 0.27, 0.16, 0.15, 0.15, 0.15, 0.14, 0.14...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>187</th>
      <td>happy</td>
      <td>3</td>
      <td>10</td>
      <td>[CLS],    [SEP],        ¨,      ##⋅,   forg...</td>
      <td>0.29, 0.2, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0....</td>
    </tr>
    <tr>
      <th>188</th>
      <td>couch</td>
      <td>3</td>
      <td>11</td>
      <td>[CLS],    [SEP], allmusic,  credits, sherlo...</td>
      <td>0.22, 0.12, 0.089, 0.088, 0.087, 0.086, 0.084,...</td>
    </tr>
    <tr>
      <th>189</th>
      <td>dog</td>
      <td>3</td>
      <td>11</td>
      <td>[CLS],      ##⁄,   ##icio, ##igraphy, trans...</td>
      <td>0.25, 0.12, 0.11, 0.1, 0.1, 0.1, 0.099, 0.097,...</td>
    </tr>
    <tr>
      <th>190</th>
      <td>run</td>
      <td>3</td>
      <td>11</td>
      <td>according,      ##ː,  checked,     took, depen...</td>
      <td>0.12, 0.11, 0.1, 0.098, 0.095, 0.094, 0.093, 0...</td>
    </tr>
    <tr>
      <th>191</th>
      <td>happy</td>
      <td>3</td>
      <td>11</td>
      <td>[CLS],    [SEP],     icao,   nothin, someth...</td>
      <td>0.28, 0.15, 0.1, 0.1, 0.099, 0.09, 0.09, 0.09,...</td>
    </tr>
  </tbody>
</table>
<p>192 rows × 5 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bdac3328-52b0-427c-9eab-82ae8a2af8b2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bdac3328-52b0-427c-9eab-82ae8a2af8b2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bdac3328-52b0-427c-9eab-82ae8a2af8b2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-d4c71aac-5edb-4081-aca1-a583443aba48">
  <button class="colab-df-quickchart" onclick="quickchart('df-d4c71aac-5edb-4081-aca1-a583443aba48')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-d4c71aac-5edb-4081-aca1-a583443aba48 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_9cb18eba-648b-44f4-bd11-585631f91b13">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_results')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_9cb18eba-648b-44f4-bd11-585631f91b13 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_results');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### 1.3. Probing Specific Heads

When suspecting a behavior in a particular head, this version loops through different groups of words to try on that head.


```python
layer = 0
head = 3

print(f"\n==== Layer {layer}, Head {head} ====\n")

W_P_i = get_BERT_WP(layer, head)

word_groups = [
    # Note: Some words break into multiple tokens, so I avoid them:
    #  "joyful", "ecstatic", "elated", "cheery"
    ["happy", "content", "cheerful", "sad", "miserable", "depressed"],
    ["scared", "confused", "hopeful", "discouraged"],
    ["cat", "wolf", "puppy"],
    ["walk", "walking", "walked"],
    ["jump", "jumping", "jumped"],
    ["justice", "freedom", "democracy"]
]

for input_words in word_groups:
    for input_word in input_words:

        print(f"{input_word:>10}: ", end="")

        matches = find_head_matches(W_P_i, input_word)

        for word, sim in matches:
            # Print out the matching words, padded to 10 characters each
            print(f"{word:>10} ", end="")

        print()

    print("\n\n--------------------------------\n\n")



```

    
    ==== Layer 0, Head 3 ====
    
         happy:       make     making       made      makes     people        not     ##made      women      ##ria  something       felt       city        dee        men      paper 
       content:       make     making     [MASK]       made      makes        una        lot       town         ir        con      arrow       city     people     ##made       nick 
      cheerful:     [MASK]       arte        una        cod        338         ag         im         ir       city        apr        336       ##cc        268        pot       ##tm 
           sad:     [MASK]      [CLS]     making       make       made      paper      light      makes technology        tan     ##made        obe         ir   business          я 
     miserable:     making     [MASK]       make        fir       made      jenny        una     ##made       city       lara      makes       veil         li        bar        sci 
     depressed:     [MASK] technology     making       make     ##made      [CLS]   veronica       made      wasps        una      craft      spice       nora      ##bic      paper 
    
    
    --------------------------------
    
    
        scared:     [MASK]      [CLS]        fir        una      paper      craft      ##ior        hoc          я      ##owe         ir      ##mas    nothing         im technology 
      confused:     nobody      never        not    nothing    without      trust         un    absence      grant      women      paper technology  resistant    lacking      force 
       hopeful:     [MASK]         un       city        all      paper        una       town       mist    country      women        art    village       para       ##cc      light 
    discouraged:     [MASK]        una      forte       diva         ag      katie        app   jennifer      jenny    olympia technology       nora     ##ever      disco        hoc 
    
    
    --------------------------------
    
    
           cat:     [MASK]      [CLS]      fifty        una     amazon     couple      bride     twenty          þ        abe        quo      forty      abdul         50      ##iss 
          wolf:     [MASK]      ##ada      ##oor    counter     melody        ##中         sv   dementia    lullaby        ##ধ       ##ad        pen        ##བ     ##oche        ##道 
         puppy:     [MASK]      [CLS]        una         un         im        nec        266      disco        336      paper technology        338        334        ina        pac 
    
    
    --------------------------------
    
    
          walk:       help    helping        let        saw   watching     helped      helps    watched       make    letting      watch       made       seen     making         on 
       walking:        pro      probe    feeling      woman        saw       tech      women      watch      paper         op       view         ec        pod    preview      angel 
        walked:        ...     [MASK]         as          ?       when       lara         va         if         be    feeling    miranda      ##zzi      ##was         16       tech 
    
    
    --------------------------------
    
    
          jump:       make       help       made        saw      watch     making       seen        let     helped        see       from     seeing   watching      force    helping 
       jumping:     [MASK]         un        una      women     people technology      woman         ag      paper       town      force     spirit communication       diva        sam 
        jumped:     [MASK]       lara      force      [CLS]        pac         un       tech         ai      ##ida   anything         if        ##g    grayson         be        ... 
    
    
    --------------------------------
    
    
       justice:        men       help     people    company         di      light       team       make       film      women      power       game         to        man      force 
       freedom:       help      offer     escape       make      grant       give    company      break      force    display         di     demand      given technology    request 
     democracy:       team        the       foot      ##zak development     ##tech     [MASK]       berg   training      paper      power       body        gym      probe      child 
    
    
    --------------------------------
    
    



```python
layer = 0
head = 2

print(f"\n==== Layer {layer}, Head {head} ====\n")

W_P_i = get_BERT_WP(layer, head)

word_groups = [
    # Test different furniture and objects to check if it maps "couch" to physical items
    ["couch", "chair", "table", "bed", "sofa"],

    # Test different animals to see if "dog" behavior generalizes
    ["dog", "cat", "lion", "elephant", "wolf"],

    # Test different movement-related words to see if "run" finds action-based contexts
    ["run", "sprint", "walk", "fly", "drive"],

    # Test security & alert-related words based on "dog" results (guard, watch, radio)
    ["guard", "watch", "alert", "detect", "surveillance"],

    # Test whether it groups abstract concepts (following "happy" head test)
    ["freedom", "justice", "law", "rights", "democracy"]
]

for input_words in word_groups:
    for input_word in input_words:

        print(f"{input_word:>10}: ", end="")

        matches = find_head_matches(W_P_i, input_word)

        for word, sim in matches:
            # Print out the matching words, padded to 10 characters each
            print(f"{word:>10} ", end="")

        print()

    print("\n\n--------------------------------\n\n")

```

    
    ==== Layer 0, Head 2 ====
    
         couch:         em      truck         ar       cell        car        sea      tanks         tr      rotor        cam       bull      ##car        org       ##em        kim 
         chair:       rear      sedan       club       tree       bank     season    vehicle        car     parish   vehicles     church        bed    dresser   argument    partial 
         table:      round      water       hill     rounds    writing       left       hall     opened     center   periodic      heart       todd       turn     square     accept 
           bed:      truck      river    playoff      creek      speed       flow    vehicle    lecture       fish     stream     flower    thunder      drain     narrow        dry 
          sofa:         ha         da         um         na        inn         om         sa        org       ##la     granny         em       ##ha        ser          \     ##dran 
    
    
    --------------------------------
    
    
           dog:        hot       sent      watch      guard      radio      guide       send      hound unsuccessfully      voice       neck      sends    success       feel        mas 
           cat:        del      bihar      ##del         lo      hotel     hotels      della      ##wal       hold         le      pussy      molly     ##slin        ##a         po 
          lion:     george     angelo        sea      ##con       tree   mountain      train     canyon       over       cell      voice      river   answered      ##jer      ##tia 
      elephant:        fan        mac      ##lio      ##ier         sy     ##ille       theo       ##zy      ##ian         ty         ti        yan    ##illon      ##ach      hagen 
          wolf:         li        mis        ##x       semi       ##zi       auto    thought    shatter      space        ##t      rosen      ##tia         un      ##isi       gray 
    
    
    --------------------------------
    
    
           run:       home      child     prison     narrow    patient     remote     female      crowd       fish imagination     willow   transfer    limited   festival    reduced 
        sprint:        fan      ##ber      ##del       sara        ram      ##hom       star         om      ##fra      laura      ##ele       ##lo      ##fan      ##run     ##ndra 
          walk:        dry     square      heath    florida         em       rose     cotton       snow      earth         up   patricia      marsh       dead      grass      media 
           fly:       swan       pigs       ears       fish        ear      drops      slave      drama      crane     dragon      paris    battles     grapes        eye       eyes 
         drive:       disk       disc       leaf      flash       club   magnetic      wheel       gene    reverse        rip       data      blood commercially    serpent    captive 
    
    
    --------------------------------
    
    
         guard:     called      honor     honour       goal      waist      stood        sin     neural      named        off         nu   followed       cell       star   memorial 
         watch:      wrist      night       bird      human       hand   finished       palm   thousand     humans      dream       turn       farm      devon       case     streak 
         alert: information   national    earlier         nu     ##ency        lin      sound     higher      chief      after        cal      match      jones       peek     tissue 
        detect:       ##le      ##iti      ##ele        mor         er      ##hom      worry          )       theo       ##is       ##el         im     ##dran       ##lo         um 
    surveillance:       shah         za        sha         im         om        com      ##uit      marsh      ##com       jana        pri       tito        ser        jen       bell 
    
    
    --------------------------------
    
    
       freedom:   catholic       held        ned       pere   relative       poet       vida    reserve   homeland      media environmental   democrat   ordained       ##ia          ) 
       justice:       deep      inter      raven        ang   criminal         pl      sound       ##to      upper        air       dora       park     formal        dev    primary 
           law:       bird       fish        dog       gift         je      match      sound       barn     purple     forest      andre       live       pack     rolled competition 
        rights:        fan       hair       left      heard     called        all     naming       film       have  universal      human       came        bel      words broadcasting 
     democracy:       ##ri     pseudo          *         ka       ##re       ##io      ##con       sven       ##di         ki       theo      ##ana       bank          )        nor 
    
    
    --------------------------------
    
    


## 1.4. PEV Analysis

This approach also helps in evaluating the inherent bias that each head has towards different sequence positions.

For a given head, we can pick an anchor point in the sequence (here I used 256), produce its pattern vector, and then compare the pattern to each of the PEVs and plot the scores to visualize.


```python
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
# 1. Load Model and Embeddings
# ──────────────────────────────────────────────────────────────────────────

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Word embeddings
embedding_layer = model.embeddings.word_embeddings.weight.detach().cpu().numpy()

# Positional embeddings (shape: [max_position_embeddings, hidden_size])
position_embedding_layer = model.embeddings.position_embeddings.weight.detach().cpu().numpy()

num_heads = model.config.num_attention_heads

# ──────────────────────────────────────────────────────────────────────────
# 2. Functions for Computing the Head Pattern
# ──────────────────────────────────────────────────────────────────────────

def get_pattern(W_P_i, pos=256, word=None):
    """
    Returns the pattern vector for a given head's projection matrix W_P_i.

    If 'word' is None, we use only the positional embedding at 'pos'.
    If 'word' is provided, we sum the word embedding with the positional embedding at 'pos'.
    """
    # Positional vector at the chosen anchor
    pos_vector = position_embedding_layer[pos]

    if word is not None:
        # Tokenize the input word (take the first subword if multiple)
        tokens = tokenizer.tokenize(word)
        token_id = tokenizer.convert_tokens_to_ids(tokens[0])
        word_vector = embedding_layer[token_id]
        # Sum the word embedding + positional embedding
        input_vector = pos_vector + word_vector
    else:
        # Use only the positional embedding
        input_vector = pos_vector

    # Compute the pattern vector
    pattern = np.dot(input_vector, W_P_i)
    return pattern

def cosine_similarity(vec, matrix):
    """
    Computes the cosine similarity between a single vector and each row in a matrix.
    """
    # shape(matrix) = (vocab_size or max_position, hidden_size)
    # shape(vec) = (hidden_size,)
    # We'll use scipy's cdist for convenience
    import scipy.spatial
    return 1 - scipy.spatial.distance.cdist([vec], matrix, metric="cosine")[0]

# ──────────────────────────────────────────────────────────────────────────
# 3. Plotting and Analysis
# ──────────────────────────────────────────────────────────────────────────

def plot_positional_similarity(layer, head, pos=256, word=None):
    """
    Plots the cosine similarity between the head's pattern (from PEV at 'pos'
    plus optional 'word' embedding) and all positional embeddings.
    """

    W_P_i = get_BERT_WP(layer, head)

    pattern = get_pattern(W_P_i, pos, word)

    similarities = cosine_similarity(pattern, position_embedding_layer)

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(similarities)), similarities, marker='o', linestyle='-')

    title_str = f"Position Bias\nLayer {layer}, Head {head}"

    if word is not None:
        title_str += f", Word='{word}'"

    plt.title(title_str)
    plt.xlabel("Position Index")
    plt.ylabel("Cosine Similarity")

    # Remove the grid
    plt.grid(False)

    # Horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--')

    # Vertical line at the anchor position
    plt.axvline(x=pos, color='red', linestyle='--')

    plt.show()

def find_head_matches_position(W_P_i, pos=256, word=None, k=15):
    """
    Returns the top-k positions (from the positional embeddings) that have the highest
    cosine similarity with the head pattern computed from:
      - The PEV at position 'pos' (if word is None), or
      - The sum of the PEV at 'pos' + the 'word' embedding (if word is not None).
    """
    pattern = get_pattern(W_P_i, pos, word)
    similarities = cosine_similarity(pattern, position_embedding_layer)
    top_indices = similarities.argsort()[-k:][::-1]
    matches = [(int(idx), similarities[idx]) for idx in top_indices]
    return matches


```

Plot the position bias for a specific layer and head.


```python
# Example 1: Pure positional input at pos=256
plot_positional_similarity(layer=0, head=2, pos=256, word=None)
```

<img src='https://lh3.googleusercontent.com/d/1XtPWYbI-_eJkLw40_oXkylSPgocbYQr-' alt='Position bias of layer 0 head 2' width='500'/>

Show the top similarities


```python
W_P_i = get_BERT_WP(layer, head)

top_matches = find_head_matches_position(W_P_i, pos=256, word=None, k=10)

print("Top 10 positional matches (pos=256, no word):")
for idx, sim in top_matches:
    print(f"  Position {idx}, Similarity: {sim:.3f}")
```

    Top 10 positional matches (pos=256, no word):
      Position 255, Similarity: 0.754
      Position 254, Similarity: 0.633
      Position 256, Similarity: 0.609
      Position 58, Similarity: 0.582
      Position 372, Similarity: 0.555
      Position 293, Similarity: 0.523
      Position 373, Similarity: 0.493
      Position 287, Similarity: 0.491
      Position 214, Similarity: 0.483
      Position 83, Similarity: 0.483


How does the inclusion of a word impact the results?


```python
# Example 2: Positional + Word embedding (pos=256, word='cat')
print(f"\n--- Positional + Word Input (pos=256, word='cat') ---")
plot_positional_similarity(layer=0, head=2, pos=256, word="cat")

```

<img src='https://lh3.googleusercontent.com/d/1ho99aUAAF7aMWmmgtL24w-McC3mJ0lDB' alt='Impact of adding a word along with the PEV' width='500'/>


```python

top_matches = find_head_matches_position(W_P_i, pos=256, word="cat", k=10)
print("Top 10 positional matches (pos=256, word='cat'):")
for idx, sim in top_matches:
    print(f"  Position {idx}, Similarity: {sim:.3f}")

```

Plot a grid for all 12 heads or all 12 layers.


```python
layer = 0
head = 5
pos = 256
word = None

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for layer in range(12):
    ax = axes[layer]

    W_P_i = get_BERT_WP(layer, head)
    pattern = get_pattern(W_P_i, pos, word)
    similarities = cosine_similarity(pattern, position_embedding_layer)

    ax.plot(range(len(similarities)), similarities, marker='o', linestyle='-')

    title_str = f"Layer {layer}, Head {head}"

    if word is not None:
        title_str += f", Word='{word}'"

    ax.set_title(title_str)
    ax.set_xlabel("Position Index")
    ax.set_ylabel("Cosine Similarity")

    ax.grid(False)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.axvline(x=pos, color='red', linestyle='--')

plt.tight_layout()
plt.show()

```

---

Biases for **head 0** in **all 12 layers**.



<img src='https://lh3.googleusercontent.com/d/1hetbYy9F-aEOGoIILAkanD-Ez1nsLUkE' alt='BERT position bias for head 0 in all layers' width='900'/>


---



Position biases for **head 5** in all layers:



<img src='https://lh3.googleusercontent.com/d/1j1ZCe_xWLBGlMS1IfnkrgexAGt7Sw_UT' alt='BERT position bias for head 5 in all layers' width='900'/>

---

We can look at the average across all heads...

First, do the math.


```python
all_similarities = np.zeros(len(position_embedding_layer))
all_sims_norm = np.zeros(len(position_embedding_layer))

norms = []

# Average the bias across all positions.
for layer_i in range(12):
    for head_i in range(12):

        # W_P for this head.
        W_P_i = get_BERT_WP(layer_i, head_i)

        # Pattern using PEV 256 as input.
        pattern = get_pattern(W_P_i, pos=256, word=None)

        # Compare pattern to all PEVs.
        similarities = cosine_similarity(pattern, position_embedding_layer)

        # Calculate a running sum.
        all_similarities += similarities

        norm = np.linalg.norm(similarities)

        # Also try a normalized version in case the norms vary a lot.
        all_sims_norm += similarities / norm

        norms.append(norm)

# Average
all_similarities /= (12 * 12)

all_sims_norm /= (12 * 12)

```

Create the plot.


```python
similarities = all_similarities
#similarities = all_sims_norm
pos=256
word = None

plt.figure(figsize=(10, 4))

plt.plot(range(len(similarities)), similarities, marker='o', linestyle='-')

plt.title("Overall Average Position Attention")
plt.xlabel("Position Index")
plt.ylabel("Avg. Cosine Similarity")

# Remove the grid
plt.grid(False)

# Horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='--')

# Vertical line at the anchor position
plt.axvline(x=pos, color='red', linestyle='--')

plt.show()
```

<img src='https://lh3.googleusercontent.com/d/1Nbif0lyMLyUVkVn1byU_0aqdK3NObv27' alt='Average position bias' width='500'/>

## 1.5. Further Work

It'd be interesting to try:

1. For each head, for the positions that it's highly biased towards, add that PEV to every word embedding in the vocabulary before doing the similarity search.
    * That might give clearer results in general,
    * and it'd be interesting to see how / whether the position changes what words the head is looking for.
2. Look at how the hidden states in later BERT layers might be brought into the vocabulary space (if they aren't already).
   * The Masked Language Modeling head might serve as the embedding space on the output--or perhaps it's only relevant to the `[MASK]` token.




# ▂▂▂▂▂▂▂▂▂▂▂▂

# S2. GPT-2: Layer-Wise Evolution of an Embedding


This section illustrates something that's already well-understood--that the Decoder layers take the word embedding for the previous token and gradually evolve it to be similar to the embedding for the next token that the model will predict.

## 2.1. Load Model

We start by loading GPT‑2 and its tokenizer from the Hugging Face Transformers library. GPT‑2’s token embeddings are stored in the model’s transformer component (specifically in `wte`).


```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT‑2 model and tokenizer.
# Note: GPT‑2 uses a byte-level BPE tokenizer, so tokenization behavior may differ from BERT.
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Extract the token embeddings.
# In GPT‑2, embeddings are stored in model.transformer.wte.
embedding_layer = model.transformer.wte.weight.detach().cpu().numpy()

# Print the shape of the embedding matrix to inspect vocabulary size and embedding dimension.
print("Embedding matrix shape:", embedding_layer.shape)

# Retrieve the LM head matrix
lm_head_embeddings = model.lm_head.weight.detach().cpu().numpy()

```


    config.json:   0%|          | 0.00/665 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/548M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]


    Embedding matrix shape: (50257, 768)


## 2.2. Run Sequence



1. **Preparing the Input and Obtaining Hidden States:**  
   We tokenize an input sentence (e.g., `"The cat sat on the"`) and perform a forward pass with `output_hidden_states=True` so that we receive the embedding output as well as the output from each transformer layer.

2. **Predicting the Next Token:**  
   Using the final logits (after all layers), we determine the next token by selecting the one with the highest probability. Since GPT‑2 ties its input embeddings with its LM head weights, we can retrieve the predicted token’s embedding from the same matrix.

3. **Layer-wise Analysis:**  
   For each layer (starting from the initial embedding), we:
   - Extract the hidden state corresponding to the last token.
   - Compute the dot product similarity between that hidden state and the predicted token’s embedding.
       - This requires first applying the final output layer normalization to the hidden state to bring it back into the vocabulary embedding space.
   - Retrieve the top tokens that are most similar to the hidden state.
   
This lets us observe how, as the data passes through each layer, the representation of the context (here, the last token) evolves toward the characteristics of the token that the model will eventually predict.



Run our sentence through the model and get the hidden states, plus the final predicted token.


```python
import torch

# Example sentence.
input_text = " The cat sat on the"
inputs = tokenizer(input_text, return_tensors="pt")

# Get outputs with all hidden states.
outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple: (embedding output, layer 1, ..., layer 12)

# Determine the predicted next token using the final logits.
logits = outputs.logits
predicted_token_id = torch.argmax(logits[0, -1, :]).item()
predicted_token = tokenizer.decode([predicted_token_id])
print("Predicted next token:", predicted_token)

# Retrieve predicted token's embedding from LM head.
predicted_emb = lm_head_embeddings[predicted_token_id]
```

    Predicted next token:  floor


## 2.3. Compare Hidden States


```python
# Sort similarities and return the top k words with their scores.
def get_top_k_words(similarities, k=5):
    top_indices = similarities.argsort()[-k:][::-1]
    top_words = []

    for idx in top_indices:
        # Convert the vocabulary index back into a token string.
        word_str = tokenizer.convert_ids_to_tokens(int(idx))

        if word_str[0] == "Ġ":
            word_str = word_str[1:]  # Remove the leading space.
        else:
            #word_str = "·" + word_str  # Add a symbol to indicate a subword.
            #word_str = "~" + word_str  # Add a symbol to indicate a subword.
            word_str = "…" + word_str  # Add a symbol to indicate a subword.

        top_words.append((word_str, similarities[idx]))

    return top_words
```


```python
import torch
import pandas as pd

# For each layer, compare the hidden state of the last token with the predicted token.
print("\nLayer-wise similarity analysis:")

# Dictionaries to store top tokens and their similarity scores per layer.
token_table = {}
sim_table = {}

for i, hs in enumerate(hidden_states):
    # Get the last token's hidden state.
    last_token_state = hs[0, -1, :]

    # Apply the final layer normalization to each hidden state to bring it into
    # the embedding space.
    #
    # However, for the final layer, the normalization has already been applied.
    # This also means that the input embedding (hidden state layer 0) has
    # already been normalized if we are generating new text.
    if i == 0 or i == len(hidden_states) - 1:
        state = last_token_state.detach().cpu().numpy()
    else:
        state = model.transformer.ln_f(last_token_state).detach().cpu().numpy()

    print(f"\nLayer {i}:")
    # Direct dot product similarity with the predicted token embedding.
    dot_sim = state.dot(predicted_emb)
    print(f"Dot product with predicted token '{predicted_token}': {dot_sim:.3f}")

    # Retrieve top-5 similar tokens using dot product similarity.
    similarities = dot_product_similarity(state, lm_head_embeddings)
    top_words = get_top_k_words(similarities, k=5)

    print("Top 5 similar tokens to hidden state:")
    for token, sim in top_words:
        print(f"  {token:10s}  {sim:.3f}")

    # Save results for table construction.
    token_table[f"Layer {i}"] = [token for token, sim in top_words]
    sim_table[f"Layer {i}"] = [sim for token, sim in top_words]

# Construct DataFrames where columns are layers and rows are rank positions.
tokens_df = pd.DataFrame(token_table, index=[f"Rank {i+1}" for i in range(5)])
sims_df = pd.DataFrame(sim_table, index=[f"Rank {i+1}" for i in range(5)])

print("\nTable of Top 5 Tokens (per layer):")
display(tokens_df)

print("\nTable of Similarities (per layer):")
display(sims_df)

```

    
    Layer-wise similarity analysis:
    
    Layer 0:
    Dot product with predicted token ' run': 2.662
    Top 5 similar tokens to hidden state:
      to          6.755
      …to         5.085
      in          4.947
      and         4.900
      ….          4.868
    
    Layer 1:
    Dot product with predicted token ' run': 4.512
    Top 5 similar tokens to hidden state:
      be          10.018
      get         9.860
      make        9.732
      find        9.182
      give        8.987
    
    Layer 2:
    Dot product with predicted token ' run': 0.683
    Top 5 similar tokens to hidden state:
      be          7.008
      get         6.035
      make        5.958
      give        5.560
      keep        5.335
    
    Layer 3:
    Dot product with predicted token ' run': 2.225
    Top 5 similar tokens to hidden state:
      be          6.327
      make        6.101
      keep        5.532
      give        4.794
      get         4.397
    
    Layer 4:
    Dot product with predicted token ' run': -3.593
    Top 5 similar tokens to hidden state:
      be          1.300
      make        0.794
      keep        0.516
      give        -1.038
      follow      -1.360
    
    Layer 5:
    Dot product with predicted token ' run': -6.860
    Top 5 similar tokens to hidden state:
      be          -2.620
      make        -3.165
      keep        -3.512
      give        -5.065
      meet        -5.199
    
    Layer 6:
    Dot product with predicted token ' run': -13.255
    Top 5 similar tokens to hidden state:
      make        -9.850
      be          -10.533
      keep        -10.777
      retire      -11.498
      continue    -12.071
    
    Layer 7:
    Dot product with predicted token ' run': -15.505
    Top 5 similar tokens to hidden state:
      retire      -11.693
      vote        -12.568
      be          -12.878
      pursue      -14.169
      make        -14.413
    
    Layer 8:
    Dot product with predicted token ' run': -20.814
    Top 5 similar tokens to hidden state:
      vote        -15.865
      be          -17.369
      retire      -17.472
      repeal      -18.025
      pursue      -18.091
    
    Layer 9:
    Dot product with predicted token ' run': -26.565
    Top 5 similar tokens to hidden state:
      vote        -19.277
      appoint     -19.860
      resign      -22.758
      caucus      -23.097
      retire      -23.117
    
    Layer 10:
    Dot product with predicted token ' run': -35.651
    Top 5 similar tokens to hidden state:
      vote        -27.857
      appoint     -30.187
      nominate    -31.263
      retire      -31.897
      repeal      -32.158
    
    Layer 11:
    Dot product with predicted token ' run': -74.890
    Top 5 similar tokens to hidden state:
      vote        -72.214
      join        -74.155
      seek        -74.494
      appoint     -74.498
      run         -74.890
    
    Layer 12:
    Dot product with predicted token ' run': -132.015
    Top 5 similar tokens to hidden state:
      run         -132.015
      seek        -132.725
      vote        -132.958
      join        -133.068
      take        -133.157
    
    Table of Top 5 Tokens (per layer):




  <div id="df-de833514-45f9-47d6-99ab-2238055cbce8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Layer 0</th>
      <th>Layer 1</th>
      <th>Layer 2</th>
      <th>Layer 3</th>
      <th>Layer 4</th>
      <th>Layer 5</th>
      <th>Layer 6</th>
      <th>Layer 7</th>
      <th>Layer 8</th>
      <th>Layer 9</th>
      <th>Layer 10</th>
      <th>Layer 11</th>
      <th>Layer 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rank 1</th>
      <td>to</td>
      <td>be</td>
      <td>be</td>
      <td>be</td>
      <td>be</td>
      <td>be</td>
      <td>make</td>
      <td>retire</td>
      <td>vote</td>
      <td>vote</td>
      <td>vote</td>
      <td>vote</td>
      <td>run</td>
    </tr>
    <tr>
      <th>Rank 2</th>
      <td>…to</td>
      <td>get</td>
      <td>get</td>
      <td>make</td>
      <td>make</td>
      <td>make</td>
      <td>be</td>
      <td>vote</td>
      <td>be</td>
      <td>appoint</td>
      <td>appoint</td>
      <td>join</td>
      <td>seek</td>
    </tr>
    <tr>
      <th>Rank 3</th>
      <td>in</td>
      <td>make</td>
      <td>make</td>
      <td>keep</td>
      <td>keep</td>
      <td>keep</td>
      <td>keep</td>
      <td>be</td>
      <td>retire</td>
      <td>resign</td>
      <td>nominate</td>
      <td>seek</td>
      <td>vote</td>
    </tr>
    <tr>
      <th>Rank 4</th>
      <td>and</td>
      <td>find</td>
      <td>give</td>
      <td>give</td>
      <td>give</td>
      <td>give</td>
      <td>retire</td>
      <td>pursue</td>
      <td>repeal</td>
      <td>caucus</td>
      <td>retire</td>
      <td>appoint</td>
      <td>join</td>
    </tr>
    <tr>
      <th>Rank 5</th>
      <td>….</td>
      <td>give</td>
      <td>keep</td>
      <td>get</td>
      <td>follow</td>
      <td>meet</td>
      <td>continue</td>
      <td>make</td>
      <td>pursue</td>
      <td>retire</td>
      <td>repeal</td>
      <td>run</td>
      <td>take</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-de833514-45f9-47d6-99ab-2238055cbce8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-de833514-45f9-47d6-99ab-2238055cbce8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-de833514-45f9-47d6-99ab-2238055cbce8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-20c5beea-6a03-4620-a1b7-69cfb11c0394">
  <button class="colab-df-quickchart" onclick="quickchart('df-20c5beea-6a03-4620-a1b7-69cfb11c0394')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-20c5beea-6a03-4620-a1b7-69cfb11c0394 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_6cebaf62-803d-44e9-961a-989a0d3ef7f2">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('tokens_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_6cebaf62-803d-44e9-961a-989a0d3ef7f2 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('tokens_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    
    Table of Similarities (per layer):




  <div id="df-3adedf8e-f3b0-465a-9edd-bcdcd7d2881c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Layer 0</th>
      <th>Layer 1</th>
      <th>Layer 2</th>
      <th>Layer 3</th>
      <th>Layer 4</th>
      <th>Layer 5</th>
      <th>Layer 6</th>
      <th>Layer 7</th>
      <th>Layer 8</th>
      <th>Layer 9</th>
      <th>Layer 10</th>
      <th>Layer 11</th>
      <th>Layer 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rank 1</th>
      <td>6.755</td>
      <td>10.018</td>
      <td>7.008</td>
      <td>6.327</td>
      <td>1.300</td>
      <td>-2.620</td>
      <td>-9.850</td>
      <td>-11.693</td>
      <td>-15.865</td>
      <td>-19.277</td>
      <td>-27.857</td>
      <td>-72.214</td>
      <td>-132.015</td>
    </tr>
    <tr>
      <th>Rank 2</th>
      <td>5.085</td>
      <td>9.860</td>
      <td>6.035</td>
      <td>6.101</td>
      <td>0.794</td>
      <td>-3.165</td>
      <td>-10.533</td>
      <td>-12.568</td>
      <td>-17.369</td>
      <td>-19.860</td>
      <td>-30.187</td>
      <td>-74.155</td>
      <td>-132.725</td>
    </tr>
    <tr>
      <th>Rank 3</th>
      <td>4.947</td>
      <td>9.732</td>
      <td>5.958</td>
      <td>5.532</td>
      <td>0.516</td>
      <td>-3.512</td>
      <td>-10.777</td>
      <td>-12.878</td>
      <td>-17.472</td>
      <td>-22.758</td>
      <td>-31.263</td>
      <td>-74.494</td>
      <td>-132.958</td>
    </tr>
    <tr>
      <th>Rank 4</th>
      <td>4.900</td>
      <td>9.182</td>
      <td>5.560</td>
      <td>4.794</td>
      <td>-1.038</td>
      <td>-5.065</td>
      <td>-11.498</td>
      <td>-14.169</td>
      <td>-18.025</td>
      <td>-23.097</td>
      <td>-31.897</td>
      <td>-74.498</td>
      <td>-133.068</td>
    </tr>
    <tr>
      <th>Rank 5</th>
      <td>4.868</td>
      <td>8.987</td>
      <td>5.335</td>
      <td>4.397</td>
      <td>-1.360</td>
      <td>-5.199</td>
      <td>-12.071</td>
      <td>-14.413</td>
      <td>-18.091</td>
      <td>-23.117</td>
      <td>-32.158</td>
      <td>-74.890</td>
      <td>-133.157</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3adedf8e-f3b0-465a-9edd-bcdcd7d2881c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3adedf8e-f3b0-465a-9edd-bcdcd7d2881c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3adedf8e-f3b0-465a-9edd-bcdcd7d2881c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-720bc205-6341-4d9e-b06f-dd1fe17bd49d">
  <button class="colab-df-quickchart" onclick="quickchart('df-720bc205-6341-4d9e-b06f-dd1fe17bd49d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-720bc205-6341-4d9e-b06f-dd1fe17bd49d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_37d74ddf-806f-4973-a2af-abcd2461e384">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('sims_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_37d74ddf-806f-4973-a2af-abcd2461e384 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('sims_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



# ▂▂▂▂▂▂▂▂▂▂▂▂

# S3. GPT-2 Heads: Analyzing Pattern Vectors in Context

Building on our analysis of BERT’s attention heads, we now extend the same methodology to GPT-2.

This time, creating the pattern vectors from actual intermediate hidden states generated by processing a sentence, instead of just using vocabulary embeddings directly.

### 3.1. Extracting $ W^P_i $


```python
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Extract token embedding matrix
embedding_layer = model.transformer.wte.weight.detach().cpu().numpy()

# Retrieve the LM head matrix
lm_head_embeddings = model.lm_head.weight.detach().cpu().numpy()

```

Define the GPT-2 version of our function for extracting the $W^P$ matrices for specific heads. It just requires updating the logic to match the model architecture of GPT-2.


```python
def get_GPT2_WP(layer, head):
    attn_layer = model.transformer.h[layer].attn
    # Get the full weight matrix from c_attn; here W is (768, 2304)
    W = attn_layer.c_attn.weight.detach().cpu().numpy()
    # Transpose so that the Q, K, V segments are along axis 0:
    W_T = W.T  # Now shape is (2304, 768)

    # Slice out Q and K: each is 768 rows (for GPT‑2, hidden_size=768)
    W_Q = W_T[:model.config.n_embd, :]                  # (768, 768)
    W_K = W_T[model.config.n_embd:2*model.config.n_embd, :]  # (768, 768)

    num_heads = model.config.n_head        # e.g., 12
    head_size = model.config.n_embd // num_heads  # 768 // 12 = 64

    # Extract the slice for the given head along the rows
    W_Q_i = W_Q[head * head_size:(head + 1) * head_size, :]  # (64, 768)
    W_K_i = W_K[head * head_size:(head + 1) * head_size, :]  # (64, 768)

    # Compute the pattern projection matrix
    W_P_i = np.dot(W_Q_i.T, W_K_i)  # Results in (768, 768)

    return W_P_i

```

### 3.2. Comparing Patterns to the Vocabulary

A few things to note for our GPT-2 version:

1. To declutter the results, I filter out the BPE tokens, which often seem to have a high similarity to everything.
    * There are still edge cases of this filtering that I haven't figured out yet. And maybe there's a simpler way?

2. Figuring out the right normalization strategy was the tricky bit. What seems to work is:
    * Apply the final layer normalization, which brings the embedding back into the vocabulary space in order to multiply it with the LM head.
        * Note: In GPT-2 the LM Head weights are tied to the input vocabulary, so the spaces are identical, which is helpful for this analysis!
    * Use cosine similarity, rather than the straight dot-product, for comparing the pattern to the vocabulary.
        * Certainly vector magnitudes can obscure similarity metrics, but I still found this requirement odd considering that next token prediction doesn't seem to need it.


```python
def find_head_matches_GPT2(W_P_i, hidden_state, k=15):

    # Project the token embedding to obtain the pattern vector
    pattern = np.dot(hidden_state, W_P_i) #.detach().cpu().numpy()

    # Apply final layer normalization
    pattern = model.transformer.ln_f(torch.tensor(pattern)).detach().cpu().numpy()
    #pattern = pattern.detach().cpu().numpy()

    # Compute cosine similarities
    similarities = cosine_similarity(pattern, embedding_layer)
    #similarities = dot_product_similarity(pattern, embedding_layer)

    # Sort by similarity, returning the indeces.
    # Just select the top 100 to account for the BPE tokens we'll be skipping
    # over.
    sorted_indices = similarities.argsort()[-100:][::-1]

    top_words = []

    # For each result...
    for idx in sorted_indices:

        # Go through until we've found our k matches.
        if len(top_words) >= k:
            break

        # Convert the vocabulary index back into a token string.
        word_str = tokenizer.convert_ids_to_tokens(int(idx))

        # Start by replacing the special character so that when we attempt to
        # re-encode it will do it correctly.
        if word_str[0] == "Ġ":
            word_str = " " + word_str[1:]

        # Skip over BPE tokens--these can clutter the similarity results.
        # You can detect these by doing a "round trip" and retokenizing the
        ## string to see if it splits into more than one token.
        #if len(tokenizer.tokenize(word_str)) > 1:
        #    continue
        # Robust round-trip check: re-encode and decode.
        encoded = tokenizer.encode(word_str, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != word_str:
            continue

        if word_str[0] == " ":
            word_str = word_str[1:]
        else:
            word_str = "…" + word_str  # Add a symbol to indicate a subword.
            #word_str = "·" + word_str  # Add a symbol to indicate a subword.
            #word_str = "~" + word_str  # Add a symbol to indicate a subword.

        #print(f"'{word_str}'")
        top_words.append((word_str, similarities[idx]))

    return top_words
```

### 3.3. Probing GPT-2 Heads

**Hidden States for a Sequence**

This cell runs a short sentence through the model and retrieves the hidden states in between every layer.


```python
import torch

# Example sentence.
#input_text = " The cat sat on the"
input_text = " While formerly a Democrat, in next year's election, the senator intends to"
inputs = tokenizer(input_text, return_tensors="pt")

# Get outputs with all hidden states.
outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple: (embedding output, layer 1, ..., layer 12)

```

    Predicted next token:  run


It's not necessary for this experiment, but I've kept the code in place for watching the next token evolve towards the prediction.


```python
# Determine the predicted next token using the final logits.
logits = outputs.logits
predicted_token_id = torch.argmax(logits[0, -1, :]).item()
predicted_token = tokenizer.decode([predicted_token_id])
print("Predicted next token:", predicted_token)

# Retrieve predicted token's embedding from LM head.
predicted_emb = lm_head_embeddings[predicted_token_id]
```

Run our experiment!

For the prediction token, we'll do our pattern vector analysis for each of the 12 heads in each of the 12 layers.


```python
import pandas as pd

num_heads = model.config.n_head

# Store results
results = []

print("At each layer, we'll print out:")
print("  1. The similarity to the token we know we'll eventually predict.")
print("  2. What the most similar word is to the current hidden state.")

print(f"{'Layer':<7} {'Prediction':<20} {'Current':<20}")

# For each layer...
for layer_i, hs in enumerate(hidden_states):

    # Layer 0 is the output of the embedding layer.
    #print("Layer", layer_i)

    # Get the last token's hidden state.
    last_token_state = hs[0, -1, :]

    # Apply the final layer normalization to each hidden state to bring it into
    # the embedding space.
    #
    # However, for the final layer, the normalization has already been applied.
    # This also means that the input embedding (hidden state layer 0) has
    # already been normalized if we are generating new text.
    if layer_i == 0 or layer_i == len(hidden_states) - 1:
        state = last_token_state.detach().cpu().numpy()
    else:
        state = model.transformer.ln_f(last_token_state).detach().cpu().numpy()

    # Similarity to predicted token--whatch the embedding gradually get closer.
    dot_sim = state.dot(predicted_emb)
    predicted = f"'{predicted_token}' ({dot_sim:.3f})"

    # Most similar word to the current hidden state. That's what's being
    # given to the attention head to produce the pattern, so it's an important
    # reference point.
    sims = dot_product_similarity(state, lm_head_embeddings)
    top_indices = sims.argsort()[-1:][::-1]
    closest_word = tokenizer.convert_ids_to_tokens(int(top_indices[0])).replace("Ġ", " ")

    closest = f"'{closest_word}' ({sims[top_indices[0]]:.3f})"

    print(f"{layer_i:<7} {predicted:<20} {closest:<20}")

    # ======== Analyze Head Patterns ========

    # Layer indeces:
    #   - Hidden state 0 is the output of the embedding layer, and we want to
    #     compare that against transformer layer 0, so the layer_i works for us
    #     in both cases.
    #   - Hidden state 12 is the output of the whole model, and we don't have
    #     anything to compare that to, so break.
    if layer_i == 12:
        print("\nSkipping output states--nothing to compare to!")
        break

    # For each of the heads...
    for head in range(num_heads):

        # Get the pattern matrix
        W_P_i = get_GPT2_WP(layer_i, head)

        # Match the head pattern to the vocabulary.
        matches = find_head_matches_GPT2(W_P_i, state, k=10)

        # Separate the words and scores.
        top_k_strs = ""
        top_k_sims = ""

        # Turn them into strings
        for word_str, sim in matches:
            top_k_strs += f"{word_str:>8}, "
            top_k_sims += f"{sim:.2}, "

        # Add the result as a row.
        results.append({
            "Closest Word": closest_word,
            "Layer": layer_i,
            "Head": head,
            "Top-k": top_k_strs,
            "Scores": top_k_sims
        })

```

    At each layer, we'll print out:
      1. The similarity to the token we know we'll eventually predict.
      2. What the most similar word is to the current hidden state.
    Layer Prediction           Current             
    0     ' run' (2.662)       ' to' (6.755)       
    1     ' run' (4.512)       ' be' (10.018)      
    2     ' run' (0.683)       ' be' (7.008)       
    3     ' run' (2.225)       ' be' (6.327)       
    4     ' run' (-3.593)      ' be' (1.300)       
    5     ' run' (-6.860)      ' be' (-2.620)      
    6     ' run' (-13.255)     ' make' (-9.850)    
    7     ' run' (-15.505)     ' retire' (-11.693) 
    8     ' run' (-20.814)     ' vote' (-15.865)   
    9     ' run' (-26.565)     ' vote' (-19.277)   
    10    ' run' (-35.651)     ' vote' (-27.857)   
    11    ' run' (-74.890)     ' vote' (-72.214)   
    12    ' run' (-132.015)    ' run' (-132.015)   
    
    Skipping output states--nothing to compare to!



```python
# Convert results to DataFrame and display
df_results = pd.DataFrame(results)

# Set pandas precision to 3 decimal points
pd.options.display.float_format = '{:.3f}'.format

# Display all rows and columns without truncation.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

display(df_results)

```



  <div id="df-eda043e7-b789-4fbb-aa79-a92ec626175c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Closest Word</th>
      <th>Layer</th>
      <th>Head</th>
      <th>Top-k</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>to</td>
      <td>0</td>
      <td>0</td>
      <td>pione,    subur,     2011,       an, extremely,     2010,     2009,      was, externalTo,      did,</td>
      <td>0.12, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>1</th>
      <td>to</td>
      <td>0</td>
      <td>1</td>
      <td>to,      for,     from,       by,       on,       in,     that,     into,     with,       of,</td>
      <td>0.24, 0.2, 0.2, 0.2, 0.2, 0.19, 0.19, 0.19, 0.19, 0.19,</td>
    </tr>
    <tr>
      <th>2</th>
      <td>to</td>
      <td>0</td>
      <td>2</td>
      <td>not,     that,       in,       at,       as,      for, extremely,      all,       do,      get,</td>
      <td>0.1, 0.096, 0.094, 0.094, 0.092, 0.09, 0.09, 0.089, 0.088, 0.088,</td>
    </tr>
    <tr>
      <th>3</th>
      <td>to</td>
      <td>0</td>
      <td>3</td>
      <td>an, externalTo,       in,       at,       as,      for,       …ē,       …ø,       …ö, …InstoreAndOnline,</td>
      <td>0.094, 0.092, 0.092, 0.09, 0.09, 0.089, 0.089, 0.088, 0.088, 0.088,</td>
    </tr>
    <tr>
      <th>4</th>
      <td>to</td>
      <td>0</td>
      <td>4</td>
      <td>an,       in,       at,       as,      not, externalTo,      for,        a,       on,     that,</td>
      <td>0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>5</th>
      <td>to</td>
      <td>0</td>
      <td>5</td>
      <td>mathemat,  tremend,   corrid, unnecess, challeng,  proport,  conflic, undermin,  conclud,     exha,</td>
      <td>0.17, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,</td>
    </tr>
    <tr>
      <th>6</th>
      <td>to</td>
      <td>0</td>
      <td>6</td>
      <td>pione,    subur, externalTo, …StreamerBot, …InstoreAndOnline, …rawdownload,       …ē,       …ø, …oreAndOnline,       …č,</td>
      <td>0.14, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12,</td>
    </tr>
    <tr>
      <th>7</th>
      <td>to</td>
      <td>0</td>
      <td>7</td>
      <td>an,       in,       at,      for,       as,       on,     that,        a,      not,       by,</td>
      <td>0.18, 0.18, 0.18, 0.18, 0.18, 0.17, 0.17, 0.17, 0.17, 0.17,</td>
    </tr>
    <tr>
      <th>8</th>
      <td>to</td>
      <td>0</td>
      <td>8</td>
      <td>it,     that,       in,      was,       at,       an,       we,       as,      had,      the,</td>
      <td>0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>9</th>
      <td>to</td>
      <td>0</td>
      <td>9</td>
      <td>an,     that,       in,       as,       at,     this,      all,      not,     made,   almost,</td>
      <td>0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>10</th>
      <td>to</td>
      <td>0</td>
      <td>10</td>
      <td>an,       in,       at,     that,       it,        a,      not,     this,       as,      all,</td>
      <td>0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.21, 0.21,</td>
    </tr>
    <tr>
      <th>11</th>
      <td>to</td>
      <td>0</td>
      <td>11</td>
      <td>…&lt;|endoftext|&gt;, …EStream,  referen,       "{, …DragonMagazine,      …_(,    …GGGG, …shapeshifter,    …,,,,,      //[,</td>
      <td>0.026, -0.003, -0.012, -0.016, -0.017, -0.017, -0.017, -0.018, -0.018, -0.023,</td>
    </tr>
    <tr>
      <th>12</th>
      <td>be</td>
      <td>1</td>
      <td>0</td>
      <td>anywhere,        a,      all,      the,    given, listening,  commons,       to, considering,     laps,</td>
      <td>0.02, 0.016, 0.014, 0.01, 0.0098, 0.0085, 0.0075, 0.0066, 0.0059, 0.0053,</td>
    </tr>
    <tr>
      <th>13</th>
      <td>be</td>
      <td>1</td>
      <td>1</td>
      <td>maybe,  simulac,     hers,   ultras,  municip,    …endi,   Tanaka,   submar,   theirs,   corrid,</td>
      <td>0.0051, 0.0018, 0.00029, -9.6e-05, -0.00062, -0.0017, -0.0017, -0.0017, -0.0019, -0.0021,</td>
    </tr>
    <tr>
      <th>14</th>
      <td>be</td>
      <td>1</td>
      <td>2</td>
      <td>…ems,     used,  visited,    track,       Ge, discovered,     poss,   viewed,    guard,        T,</td>
      <td>0.015, 0.015, 0.014, 0.011, 0.011, 0.01, 0.0097, 0.0093, 0.0091, 0.0088,</td>
    </tr>
    <tr>
      <th>15</th>
      <td>be</td>
      <td>1</td>
      <td>3</td>
      <td>Per,       As,     Oops, discovered,    Night,     Lily,      her,       No,      Yet,   Common,</td>
      <td>0.044, 0.041, 0.039, 0.038, 0.038, 0.037, 0.037, 0.036, 0.035, 0.035,</td>
    </tr>
    <tr>
      <th>16</th>
      <td>be</td>
      <td>1</td>
      <td>4</td>
      <td>adopted,      315,      Sal, migrated,    Moroc,     your,        R,  current,      333,  millenn,</td>
      <td>0.053, 0.049, 0.048, 0.048, 0.047, 0.047, 0.047, 0.046, 0.046, 0.046,</td>
    </tr>
    <tr>
      <th>17</th>
      <td>be</td>
      <td>1</td>
      <td>5</td>
      <td>give,       be,      get,     make,       do,     take,     find,      try,  prevent,      the,</td>
      <td>0.14, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12,</td>
    </tr>
    <tr>
      <th>18</th>
      <td>be</td>
      <td>1</td>
      <td>6</td>
      <td>future, …irection,  further,    given,      two, additionally,     also,     only,      one, additional,</td>
      <td>0.066, 0.065, 0.063, 0.06, 0.058, 0.055, 0.055, 0.053, 0.049, 0.049,</td>
    </tr>
    <tr>
      <th>19</th>
      <td>be</td>
      <td>1</td>
      <td>7</td>
      <td>get,     give,       do,     what,       at,  …second,  because,    after,        ",   second,</td>
      <td>0.1, 0.1, 0.097, 0.096, 0.095, 0.093, 0.092, 0.091, 0.09, 0.09,</td>
    </tr>
    <tr>
      <th>20</th>
      <td>be</td>
      <td>1</td>
      <td>8</td>
      <td>…eh,      …te,     …sch,     …ust,   …alien,    …ured,     …lit,     …VOL,      …yk,    …alus,</td>
      <td>0.029, 0.029, 0.028, 0.025, 0.022, 0.017, 0.017, 0.016, 0.016, 0.015,</td>
    </tr>
    <tr>
      <th>21</th>
      <td>be</td>
      <td>1</td>
      <td>9</td>
      <td>V,     …pro,       Jr,       …V,    …util,      …pl,      Reg,     …att,        ',      …sm,</td>
      <td>0.071, 0.062, 0.062, 0.061, 0.06, 0.06, 0.059, 0.059, 0.058, 0.058,</td>
    </tr>
    <tr>
      <th>22</th>
      <td>be</td>
      <td>1</td>
      <td>10</td>
      <td>Moroc,      nil,     Nish,     Nope,  referen,    Const,     Oops,   corrid,      Tup,    slept,</td>
      <td>0.15, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>23</th>
      <td>be</td>
      <td>1</td>
      <td>11</td>
      <td>be,   arrive,     make,    …ffee, accommodate, practice, accomplish, transition,  achieve,    …asty,</td>
      <td>0.0074, 0.0017, 0.0013, 6.5e-05, -0.0011, -0.0013, -0.0015, -0.0016, -0.0017, -0.0019,</td>
    </tr>
    <tr>
      <th>24</th>
      <td>be</td>
      <td>2</td>
      <td>0</td>
      <td>…Acknowled, …isSpecialOrderable,      HPV,   chunks, sections,    beads,  …skirts,   …stage,    …cake,    …poke,</td>
      <td>0.041, 0.04, 0.035, 0.035, 0.032, 0.031, 0.029, 0.028, 0.028, 0.028,</td>
    </tr>
    <tr>
      <th>25</th>
      <td>be</td>
      <td>2</td>
      <td>1</td>
      <td>…99,      …80,      …83,     …999,      …88,      …95,      …82,      …50,      …81,     …399,</td>
      <td>0.077, 0.058, 0.057, 0.057, 0.055, 0.055, 0.054, 0.053, 0.052, 0.052,</td>
    </tr>
    <tr>
      <th>26</th>
      <td>be</td>
      <td>2</td>
      <td>2</td>
      <td>a,       to,  kitchen,  pockets,     more, barriers,     many, villages, configurations,     into,</td>
      <td>0.072, 0.069, 0.069, 0.068, 0.066, 0.065, 0.063, 0.061, 0.061, 0.061,</td>
    </tr>
    <tr>
      <th>27</th>
      <td>be</td>
      <td>2</td>
      <td>3</td>
      <td>predetermined, possible, …isSpecialOrderable,   predec,     lest,  castles,   satell,    mates,   nomine,   little,</td>
      <td>0.036, 0.035, 0.034, 0.033, 0.033, 0.031, 0.031, 0.03, 0.03, 0.03,</td>
    </tr>
    <tr>
      <th>28</th>
      <td>be</td>
      <td>2</td>
      <td>4</td>
      <td>to,   …arios,   …later,     …xes, …rawdownloadcloneembedreportprint, …GGGGGGGG,    …ngth, …ibility,     amen,   …ivity,</td>
      <td>0.02, 0.018, 0.01, 0.0096, 0.0094, 0.0091, 0.0069, 0.0068, 0.0068, 0.0065,</td>
    </tr>
    <tr>
      <th>29</th>
      <td>be</td>
      <td>2</td>
      <td>5</td>
      <td>discretion, scissors,      not,   selves, otherwise,      she,  …breaks,       if,     even, warranted,</td>
      <td>0.027, 0.023, 0.023, 0.023, 0.022, 0.021, 0.021, 0.021, 0.019, 0.019,</td>
    </tr>
    <tr>
      <th>30</th>
      <td>be</td>
      <td>2</td>
      <td>6</td>
      <td>…selves,      …We,     …Don, …Imagine,   …there,    …self,     …Let,      …de,     …why,     …Say,</td>
      <td>0.016, 0.015, 0.015, 0.012, 0.01, 0.01, 0.0077, 0.0077, 0.0061, 0.0054,</td>
    </tr>
    <tr>
      <th>31</th>
      <td>be</td>
      <td>2</td>
      <td>7</td>
      <td>Who,      the,      one,        a,     Gang,      One,   Survey,      rec,  Masters,     that,</td>
      <td>-0.021, -0.022, -0.023, -0.023, -0.024, -0.024, -0.027, -0.027, -0.028, -0.028,</td>
    </tr>
    <tr>
      <th>32</th>
      <td>be</td>
      <td>2</td>
      <td>8</td>
      <td>to,        a,     what,  adolesc,  …ITNESS,  …istani,      Kap, …mission,   speedy,   …adequ,</td>
      <td>0.022, 0.021, 0.02, 0.02, 0.017, 0.017, 0.016, 0.014, 0.014, 0.014,</td>
    </tr>
    <tr>
      <th>33</th>
      <td>be</td>
      <td>2</td>
      <td>9</td>
      <td>vulner, throughput,     rigs,    …zing,    flows, …uations, optimizations, frameworks, barriers,   expans,</td>
      <td>0.049, 0.045, 0.044, 0.044, 0.042, 0.042, 0.041, 0.041, 0.041, 0.041,</td>
    </tr>
    <tr>
      <th>34</th>
      <td>be</td>
      <td>2</td>
      <td>10</td>
      <td>it,      the,     what,     them,        a,     this, something, competing,     that,  another,</td>
      <td>0.085, 0.08, 0.079, 0.077, 0.076, 0.076, 0.075, 0.075, 0.074, 0.073,</td>
    </tr>
    <tr>
      <th>35</th>
      <td>be</td>
      <td>2</td>
      <td>11</td>
      <td>N,      ...,     …...,        K,       NP,      Bat,    …hill,       …&amp;,    Super,      Gil,</td>
      <td>0.015, 0.005, 0.0018, 0.0015, -0.00054, -0.0034, -0.0048, -0.005, -0.0054, -0.0074,</td>
    </tr>
    <tr>
      <th>36</th>
      <td>be</td>
      <td>3</td>
      <td>0</td>
      <td>far,   proble,    pregn, sacrific,   confir,    …swer,    mosqu,    focus,    cytok,   demand,</td>
      <td>0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>37</th>
      <td>be</td>
      <td>3</td>
      <td>1</td>
      <td>…tons, Centauri,  …little,   Styles, …nothing,  …selves,     …mol, Solitaire,     …SCP,     Sens,</td>
      <td>0.027, 0.024, 0.022, 0.019, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014,</td>
    </tr>
    <tr>
      <th>38</th>
      <td>be</td>
      <td>3</td>
      <td>2</td>
      <td>most,    their,     next,     real,      the,      his,    right,        a,     more,      our,</td>
      <td>0.15, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.13,</td>
    </tr>
    <tr>
      <th>39</th>
      <td>be</td>
      <td>3</td>
      <td>3</td>
      <td>thereto,  …skirts,    compr,   Rohing, …ardless, …EStream,   Rebell, …Redditor,  …uberty,   Chaser,</td>
      <td>0.015, 0.0062, 0.0024, 0.002, -6.3e-05, -0.0033, -0.0034, -0.0039, -0.004, -0.0048,</td>
    </tr>
    <tr>
      <th>40</th>
      <td>be</td>
      <td>3</td>
      <td>4</td>
      <td>Narr,    …Benz,     Akin, …enegger,     Benz,   …rahim,  …ashtra,    Karma,     DRAG, snowball,</td>
      <td>0.029, 0.022, 0.018, 0.016, 0.012, 0.0071, 0.0068, 0.0048, 0.0039, 0.0027,</td>
    </tr>
    <tr>
      <th>41</th>
      <td>be</td>
      <td>3</td>
      <td>5</td>
      <td>NES,     this,       if,    these,   Google,   little,      you,     gone,       …!,      old,</td>
      <td>0.11, 0.11, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1,</td>
    </tr>
    <tr>
      <th>42</th>
      <td>be</td>
      <td>3</td>
      <td>6</td>
      <td>because,     …!).,      …!.,      too,      ….),     nude,     with, …modified,      …!),      …).,</td>
      <td>0.11, 0.1, 0.1, 0.1, 0.097, 0.097, 0.093, 0.092, 0.091, 0.09,</td>
    </tr>
    <tr>
      <th>43</th>
      <td>be</td>
      <td>3</td>
      <td>7</td>
      <td>for,      but,       or,      and,      out,    …READ,       as,       in,       on,     …for,</td>
      <td>0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12,</td>
    </tr>
    <tr>
      <th>44</th>
      <td>be</td>
      <td>3</td>
      <td>8</td>
      <td>some,     less,        a,     over,   Cantor,     Cree,     into,  …ricane,  Warrant,      the,</td>
      <td>0.059, 0.055, 0.055, 0.052, 0.048, 0.047, 0.047, 0.045, 0.045, 0.044,</td>
    </tr>
    <tr>
      <th>45</th>
      <td>be</td>
      <td>3</td>
      <td>9</td>
      <td>…enegger,       …±,     …ELS, limiting,  …lessly,     …zyk, …ardless,    …STEM,     …ALL,   thence,</td>
      <td>0.028, -0.00087, -0.0013, -0.0035, -0.0047, -0.0055, -0.0057, -0.0062, -0.0064, -0.007,</td>
    </tr>
    <tr>
      <th>46</th>
      <td>be</td>
      <td>3</td>
      <td>10</td>
      <td>be,     calm, passively,      not, temporary,    cause,       it,    quiet,   suffer, physical,</td>
      <td>0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>47</th>
      <td>be</td>
      <td>3</td>
      <td>11</td>
      <td>though,     thus,   enough,   amount,    -----,    conce,    …rely,     …aco,    looph,      tho,</td>
      <td>0.094, 0.086, 0.083, 0.08, 0.078, 0.076, 0.076, 0.075, 0.074, 0.074,</td>
    </tr>
    <tr>
      <th>48</th>
      <td>be</td>
      <td>4</td>
      <td>0</td>
      <td>…GoldMagikarp, …enegger, …DragonMagazine,      htt,     Urug,     true, …Anonymous,     …fty, …Written,    …vous,</td>
      <td>0.047, 0.044, 0.043, 0.042, 0.041, 0.04, 0.04, 0.039, 0.039, 0.038,</td>
    </tr>
    <tr>
      <th>49</th>
      <td>be</td>
      <td>4</td>
      <td>1</td>
      <td>you,   around,   enough,  because,     very,    quite,       to,     that,      too,       we,</td>
      <td>0.095, 0.095, 0.091, 0.089, 0.087, 0.087, 0.087, 0.086, 0.084, 0.084,</td>
    </tr>
    <tr>
      <th>50</th>
      <td>be</td>
      <td>4</td>
      <td>2</td>
      <td>…&lt;|endoftext|&gt;,      The, Currently,   reader,      new,   former,    …Just,     …New,    While,     …The,</td>
      <td>0.12, 0.12, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1,</td>
    </tr>
    <tr>
      <th>51</th>
      <td>be</td>
      <td>4</td>
      <td>3</td>
      <td>can,      had,     must,    would,      has,    might,      got,   should,    could,     will,</td>
      <td>0.24, 0.23, 0.23, 0.23, 0.22, 0.22, 0.22, 0.22, 0.22, 0.21,</td>
    </tr>
    <tr>
      <th>52</th>
      <td>be</td>
      <td>4</td>
      <td>4</td>
      <td>…enez,    …inse, …Donnell,    …iets,   …20439,  Harding,      …JV,      …cz,   confir,      …FK,</td>
      <td>0.061, 0.057, 0.049, 0.048, 0.047, 0.046, 0.044, 0.044, 0.044, 0.043,</td>
    </tr>
    <tr>
      <th>53</th>
      <td>be</td>
      <td>4</td>
      <td>5</td>
      <td>********************************,    …kept,    …redd, …heartedly, …================================,  …Allows,    …stri,  …sheets,   …conom,     …Nik,</td>
      <td>0.052, 0.044, 0.043, 0.042, 0.04, 0.039, 0.039, 0.039, 0.038, 0.038,</td>
    </tr>
    <tr>
      <th>54</th>
      <td>be</td>
      <td>4</td>
      <td>6</td>
      <td>that,      the,     this,    large,        a,      one,    small, substantial,      any,     what,</td>
      <td>0.17, 0.17, 0.17, 0.17, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,</td>
    </tr>
    <tr>
      <th>55</th>
      <td>be</td>
      <td>4</td>
      <td>7</td>
      <td>fri,    muted,   …bably,    …vous,   …aukee,     neut, …ItemTracker,     …eat,     feat,  …icably,</td>
      <td>0.036, 0.031, 0.016, 0.015, 0.015, 0.013, 0.011, 0.01, 0.01, 0.0093,</td>
    </tr>
    <tr>
      <th>56</th>
      <td>be</td>
      <td>4</td>
      <td>8</td>
      <td>imagination, Stephenson, …enegger,   synerg,     both, regenerate,  …insula,  vacated, …etheless, …agonists,</td>
      <td>0.046, 0.038, 0.037, 0.037, 0.036, 0.034, 0.031, 0.03, 0.03, 0.03,</td>
    </tr>
    <tr>
      <th>57</th>
      <td>be</td>
      <td>4</td>
      <td>9</td>
      <td>he, considering,      men,      one,      man,    every,  willing,      gau,   asking, interested,</td>
      <td>0.14, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>58</th>
      <td>be</td>
      <td>4</td>
      <td>10</td>
      <td>…asio,     HELP,   …aucus, …NetMessage,     …apo,   …Polit,      NEC,    …most,  hardest,  Firstly,</td>
      <td>0.032, 0.02, 0.018, 0.017, 0.014, 0.013, 0.012, 0.012, 0.012, 0.011,</td>
    </tr>
    <tr>
      <th>59</th>
      <td>be</td>
      <td>4</td>
      <td>11</td>
      <td>as,       to,      but,      out,  because,       ….,      bec,     with,       of,     come,</td>
      <td>0.11, 0.11, 0.11, 0.11, 0.099, 0.096, 0.095, 0.095, 0.094, 0.091,</td>
    </tr>
    <tr>
      <th>60</th>
      <td>be</td>
      <td>5</td>
      <td>0</td>
      <td>…mut, …depending, …fitting, …reditary,  …spirit,  …higher, …particularly,  …ecause,  …rogens,   …bably,</td>
      <td>0.026, 0.021, 0.019, 0.018, 0.018, 0.018, 0.018, 0.018, 0.017, 0.017,</td>
    </tr>
    <tr>
      <th>61</th>
      <td>be</td>
      <td>5</td>
      <td>1</td>
      <td>…pard,  ACTIONS,     …aku,  Pegasus,   …oulos,   …meier, …enegger,  …higher,     …mur,    …okia,</td>
      <td>0.013, 0.0088, 0.007, 0.0055, 0.0051, 0.005, 0.0014, 0.00056, 0.00036, -0.00012,</td>
    </tr>
    <tr>
      <th>62</th>
      <td>be</td>
      <td>5</td>
      <td>2</td>
      <td>this,    other,     what,       we,       so,      one,  someone,        a, necessary,       it,</td>
      <td>0.18, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.16,</td>
    </tr>
    <tr>
      <th>63</th>
      <td>be</td>
      <td>5</td>
      <td>3</td>
      <td>…"]=&gt;,     Chau,      …Us,  because,   …erker,    worry, arrogance,   …conom,      …Mu,    until,</td>
      <td>0.099, 0.087, 0.087, 0.082, 0.077, 0.076, 0.075, 0.075, 0.074, 0.073,</td>
    </tr>
    <tr>
      <th>64</th>
      <td>be</td>
      <td>5</td>
      <td>4</td>
      <td>…apters,      …23,       11,       23,  factors, proteins, tablespoons,       12,     tiss,       15,</td>
      <td>0.15, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>65</th>
      <td>be</td>
      <td>5</td>
      <td>5</td>
      <td>role, judgment,     wake,        n, accountability,       MY,   roller,  …guided,  payment,    Shane,</td>
      <td>0.082, 0.081, 0.079, 0.079, 0.078, 0.076, 0.075, 0.073, 0.072, 0.072,</td>
    </tr>
    <tr>
      <th>66</th>
      <td>be</td>
      <td>5</td>
      <td>6</td>
      <td>…1200, …ometimes,     anew,    aloud, …=-=-=-=-,      awa,    …9999,     …PDF, firsthand,  insofar,</td>
      <td>0.018, 0.018, 0.017, 0.015, 0.014, 0.012, 0.011, 0.008, 0.0078, 0.0071,</td>
    </tr>
    <tr>
      <th>67</th>
      <td>be</td>
      <td>5</td>
      <td>7</td>
      <td>…ascript, …ritional, …terness,    advoc,   …aukee, …dayName,   …ansen,   …perty,    …orts,   …itton,</td>
      <td>0.0083, 0.0018, -0.0015, -0.0019, -0.0036, -0.0043, -0.0054, -0.0057, -0.0057, -0.0065,</td>
    </tr>
    <tr>
      <th>68</th>
      <td>be</td>
      <td>5</td>
      <td>8</td>
      <td>if,  walking,     role,     with,      che,      …-&gt;,  healthy, …ternally,  through,       it,</td>
      <td>0.11, 0.11, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.099,</td>
    </tr>
    <tr>
      <th>69</th>
      <td>be</td>
      <td>5</td>
      <td>9</td>
      <td>…ometimes, miscarriage,   Subtle,    whiff, …omething,   …awaru,   miscar,    …ibly,    …IBLE,   …atari,</td>
      <td>0.047, 0.046, 0.045, 0.043, 0.042, 0.04, 0.039, 0.039, 0.038, 0.038,</td>
    </tr>
    <tr>
      <th>70</th>
      <td>be</td>
      <td>5</td>
      <td>10</td>
      <td>2017,     2016,     2018, preliminary,  Tuesday,  concise, examining,     2015, presidential,     next,</td>
      <td>0.14, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>71</th>
      <td>be</td>
      <td>5</td>
      <td>11</td>
      <td>…ataka, …ynthesis,  pasture,   …agall,    …ibal,      jug, …brainer,   Honour,    …vind,   …joice,</td>
      <td>0.038, 0.028, 0.027, 0.027, 0.026, 0.023, 0.022, 0.021, 0.02, 0.02,</td>
    </tr>
    <tr>
      <th>72</th>
      <td>make</td>
      <td>6</td>
      <td>0</td>
      <td>M,        =,      the,     both,     also,     have,    wrote,       CM,      …ls,        L,</td>
      <td>0.1, 0.098, 0.096, 0.09, 0.087, 0.087, 0.085, 0.085, 0.085, 0.084,</td>
    </tr>
    <tr>
      <th>73</th>
      <td>make</td>
      <td>6</td>
      <td>1</td>
      <td>The,     Grip,    Duffy,    Joyce,       …?,  Playing,       or,    Kenny,     Cham, …atherine,</td>
      <td>0.089, 0.088, 0.086, 0.08, 0.08, 0.08, 0.078, 0.077, 0.075, 0.075,</td>
    </tr>
    <tr>
      <th>74</th>
      <td>make</td>
      <td>6</td>
      <td>2</td>
      <td>duel,    minim,    postp,    withd,      scr,   ellipt,  trimmed, suspense,      aur,   shaved,</td>
      <td>0.05, 0.048, 0.045, 0.042, 0.042, 0.041, 0.04, 0.04, 0.038, 0.037,</td>
    </tr>
    <tr>
      <th>75</th>
      <td>make</td>
      <td>6</td>
      <td>3</td>
      <td>there,       he, …GoldMagikarp,    feels,     …'ll,      can,     ratt,   proves, convince,     will,</td>
      <td>0.15, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>76</th>
      <td>make</td>
      <td>6</td>
      <td>4</td>
      <td>good,       Fl,   little, …efficient, antioxidants,    Cheap, sustainability,     Hemp, …hematically,  savings,</td>
      <td>0.099, 0.091, 0.087, 0.085, 0.084, 0.084, 0.084, 0.083, 0.083, 0.082,</td>
    </tr>
    <tr>
      <th>77</th>
      <td>make</td>
      <td>6</td>
      <td>5</td>
      <td>fact,     what,    sorts,  finding,      how,      the,   simple,    those, excessive,    simpl,</td>
      <td>0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12,</td>
    </tr>
    <tr>
      <th>78</th>
      <td>make</td>
      <td>6</td>
      <td>6</td>
      <td>be, deputies,     make, cabinets,    spend,        ",     even,  overall,     DRAG,     push,</td>
      <td>0.11, 0.11, 0.099, 0.099, 0.098, 0.096, 0.093, 0.09, 0.089, 0.088,</td>
    </tr>
    <tr>
      <th>79</th>
      <td>make</td>
      <td>6</td>
      <td>7</td>
      <td>AK,  innings,  matches,     Akin,     salv,  stained, unloaded,    …Wood,      dun, …stained,</td>
      <td>0.057, 0.053, 0.051, 0.05, 0.049, 0.048, 0.048, 0.047, 0.047, 0.047,</td>
    </tr>
    <tr>
      <th>80</th>
      <td>make</td>
      <td>6</td>
      <td>8</td>
      <td>…Rot,    …kell, mentioning, imperson,    maker,   scales,   disson, disabling,    seism,      …ku,</td>
      <td>0.062, 0.058, 0.054, 0.052, 0.049, 0.049, 0.049, 0.049, 0.048, 0.048,</td>
    </tr>
    <tr>
      <th>81</th>
      <td>make</td>
      <td>6</td>
      <td>9</td>
      <td>…answered,     …080,    upper,   …orius,     …052,    …itia,      …-&gt;,  …repeat,     seiz,    Olymp,</td>
      <td>0.038, 0.031, 0.029, 0.028, 0.027, 0.026, 0.025, 0.021, 0.019, 0.018,</td>
    </tr>
    <tr>
      <th>82</th>
      <td>make</td>
      <td>6</td>
      <td>10</td>
      <td>…bent,     hemp, sporting,      ANG,      Tib,  however,     Bagg,    dimin, favoured,   gifted,</td>
      <td>0.043, 0.042, 0.041, 0.04, 0.038, 0.037, 0.036, 0.035, 0.034, 0.034,</td>
    </tr>
    <tr>
      <th>83</th>
      <td>make</td>
      <td>6</td>
      <td>11</td>
      <td>there,       we,        I,     that,       it,     they,     what,      how,  several,      the,</td>
      <td>0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>84</th>
      <td>retire</td>
      <td>7</td>
      <td>0</td>
      <td>a,      its,  …000000,      the,   Cheong,      …QU,      his,    …iris,       it,    versa,</td>
      <td>0.094, 0.093, 0.091, 0.088, 0.088, 0.087, 0.086, 0.086, 0.086, 0.086,</td>
    </tr>
    <tr>
      <th>85</th>
      <td>retire</td>
      <td>7</td>
      <td>1</td>
      <td>himself, themselves,    costs, taxpayers,     fees, rewritten,    reimb,    Costs, incumbent,  herself,</td>
      <td>0.099, 0.097, 0.088, 0.082, 0.082, 0.079, 0.079, 0.078, 0.078, 0.077,</td>
    </tr>
    <tr>
      <th>86</th>
      <td>retire</td>
      <td>7</td>
      <td>2</td>
      <td>…kat,     seiz,  ACTIONS,     guts,    …oren, …criminal,   intest,  …schild,     …aku,    …talk,</td>
      <td>0.0064, 0.0043, 0.0023, 0.00062, 0.00034, -0.00034, -0.0035, -0.005, -0.0054, -0.0054,</td>
    </tr>
    <tr>
      <th>87</th>
      <td>retire</td>
      <td>7</td>
      <td>3</td>
      <td>will,       is,    would,  creates, prevents, completely,    takes,   future,    helps, involves,</td>
      <td>0.17, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>88</th>
      <td>retire</td>
      <td>7</td>
      <td>4</td>
      <td>articulated, consistent,  …terson, complete,     …UGC,   ratios, teammates, inconsistent,   embold,     …awa,</td>
      <td>0.032, 0.02, 0.02, 0.019, 0.017, 0.017, 0.017, 0.016, 0.015, 0.015,</td>
    </tr>
    <tr>
      <th>89</th>
      <td>retire</td>
      <td>7</td>
      <td>5</td>
      <td>its,     they, worsening,       is,  pending, worsened,        I,       …?,   should,    …phia,</td>
      <td>0.11, 0.1, 0.095, 0.095, 0.094, 0.094, 0.091, 0.09, 0.088, 0.087,</td>
    </tr>
    <tr>
      <th>90</th>
      <td>retire</td>
      <td>7</td>
      <td>6</td>
      <td>…ADA, …livious,     …iru,      …KR,    …gars, …SPONSORED,     Kron, …etheless, …AppData,   …GREEN,</td>
      <td>0.023, 0.0071, 0.0061, 0.0042, 0.0035, 0.0012, -0.0042, -0.0044, -0.007, -0.0073,</td>
    </tr>
    <tr>
      <th>91</th>
      <td>retire</td>
      <td>7</td>
      <td>7</td>
      <td>…cause, …because, outright, …theless, …conservancy,      NIC,    juven,   …regon,   …which,   …indle,</td>
      <td>0.013, 0.011, 0.011, 0.0094, 0.0094, 0.0079, 0.0076, 0.007, 0.0066, 0.0063,</td>
    </tr>
    <tr>
      <th>92</th>
      <td>retire</td>
      <td>7</td>
      <td>8</td>
      <td>followed,       ….,    newly,  emerged,      new, resulting, prompted,      led,     …iqu,  Reuters,</td>
      <td>0.11, 0.1, 0.1, 0.097, 0.095, 0.095, 0.094, 0.092, 0.09, 0.09,</td>
    </tr>
    <tr>
      <th>93</th>
      <td>retire</td>
      <td>7</td>
      <td>9</td>
      <td>I, considering,       we,      are,      the,     they,      for,      you,      all,       in,</td>
      <td>0.11, 0.1, 0.099, 0.099, 0.097, 0.096, 0.095, 0.095, 0.094, 0.093,</td>
    </tr>
    <tr>
      <th>94</th>
      <td>retire</td>
      <td>7</td>
      <td>10</td>
      <td>assessment,      MAY,        r,     such,    other,   higher, discipline,  heavier, perception,   judged,</td>
      <td>0.11, 0.1, 0.1, 0.1, 0.098, 0.097, 0.095, 0.095, 0.093, 0.093,</td>
    </tr>
    <tr>
      <th>95</th>
      <td>retire</td>
      <td>7</td>
      <td>11</td>
      <td>…fg,   …quant,   …rouse,  …schild,    …bats,     …avi,     …god,      …bg,   …esley, …ocratic,</td>
      <td>0.025, 0.023, 0.019, 0.015, 0.015, 0.014, 0.013, 0.013, 0.011, 0.011,</td>
    </tr>
    <tr>
      <th>96</th>
      <td>vote</td>
      <td>8</td>
      <td>0</td>
      <td>country,   recent,     vast,   latest, …theless,    usual,   destro,   suspic, previous,   latter,</td>
      <td>0.041, 0.038, 0.036, 0.036, 0.032, 0.032, 0.031, 0.031, 0.03, 0.029,</td>
    </tr>
    <tr>
      <th>97</th>
      <td>vote</td>
      <td>8</td>
      <td>1</td>
      <td>…due,   …rouse,      …JJ,   …think,     …rue,  Nations,   Monaco,     …080,   Geneva,    …dden,</td>
      <td>-0.0095, -0.01, -0.015, -0.016, -0.017, -0.017, -0.017, -0.018, -0.019, -0.019,</td>
    </tr>
    <tr>
      <th>98</th>
      <td>vote</td>
      <td>8</td>
      <td>2</td>
      <td>…20439,      CLR,    …ivas,    …WARE,  Fernand,    …href,  Fiorina,     …ACA,   …ODUCT, …Synopsis,</td>
      <td>-0.037, -0.042, -0.046, -0.048, -0.048, -0.05, -0.052, -0.053, -0.053, -0.053,</td>
    </tr>
    <tr>
      <th>99</th>
      <td>vote</td>
      <td>8</td>
      <td>3</td>
      <td>punishable,      pes, Constitution,    comma, sanctioned,   system,     fold,  archaic, politically,   reform,</td>
      <td>0.095, 0.089, 0.086, 0.086, 0.086, 0.084, 0.083, 0.081, 0.08, 0.08,</td>
    </tr>
    <tr>
      <th>100</th>
      <td>vote</td>
      <td>8</td>
      <td>4</td>
      <td>…:, existing,   rather,       …,,      far,   around, encompass,       ….,        I,  roughly,</td>
      <td>0.085, 0.076, 0.074, 0.07, 0.069, 0.068, 0.066, 0.065, 0.064, 0.062,</td>
    </tr>
    <tr>
      <th>101</th>
      <td>vote</td>
      <td>8</td>
      <td>5</td>
      <td>someone,      are,     have,     some,     that,      who,  smarter,  wishing,      may,      not,</td>
      <td>0.07, 0.064, 0.064, 0.058, 0.057, 0.056, 0.056, 0.055, 0.054, 0.054,</td>
    </tr>
    <tr>
      <th>102</th>
      <td>vote</td>
      <td>8</td>
      <td>6</td>
      <td>someday, …cigarettes,  ARTICLE,    …pipe, …reality,    joins,  emerges, …cigarette,     bloc,     ASAP,</td>
      <td>0.043, 0.035, 0.032, 0.032, 0.032, 0.032, 0.032, 0.031, 0.029, 0.029,</td>
    </tr>
    <tr>
      <th>103</th>
      <td>vote</td>
      <td>8</td>
      <td>7</td>
      <td>the,    again,      its,      The,  another,        a,     some,  several,     that, whatever,</td>
      <td>0.11, 0.095, 0.09, 0.088, 0.087, 0.087, 0.086, 0.086, 0.085, 0.081,</td>
    </tr>
    <tr>
      <th>104</th>
      <td>vote</td>
      <td>8</td>
      <td>8</td>
      <td>but,     Char,      and,   prison,      mac,        d,       EU, accounting,      now,    orbit,</td>
      <td>0.088, 0.081, 0.071, 0.071, 0.067, 0.064, 0.064, 0.063, 0.063, 0.062,</td>
    </tr>
    <tr>
      <th>105</th>
      <td>vote</td>
      <td>8</td>
      <td>9</td>
      <td>altogether, adequate,      Kum,  correct,   proper, independent, standards,    costs, entirely,    Plato,</td>
      <td>0.097, 0.086, 0.078, 0.076, 0.076, 0.076, 0.075, 0.075, 0.074, 0.074,</td>
    </tr>
    <tr>
      <th>106</th>
      <td>vote</td>
      <td>8</td>
      <td>10</td>
      <td>conformity,  privile,     abol, revocation,    heirs,  …atever,   reconc,    coerc, neutrality, perpetual,</td>
      <td>0.047, 0.045, 0.044, 0.043, 0.041, 0.04, 0.039, 0.039, 0.039, 0.038,</td>
    </tr>
    <tr>
      <th>107</th>
      <td>vote</td>
      <td>8</td>
      <td>11</td>
      <td>…tein,      DOI, …enegger,      amd, POLITICO,     Nato,      TBA,      SOS,     …MAT,      pci,</td>
      <td>0.0031, -0.0078, -0.009, -0.011, -0.012, -0.013, -0.013, -0.013, -0.013, -0.013,</td>
    </tr>
    <tr>
      <th>108</th>
      <td>vote</td>
      <td>9</td>
      <td>0</td>
      <td>…President,  himself, …Quantity,     Amid, Presidential, quotation, finishes, President, dividend, …ciation,</td>
      <td>0.09, 0.085, 0.082, 0.08, 0.077, 0.077, 0.077, 0.076, 0.075, 0.074,</td>
    </tr>
    <tr>
      <th>109</th>
      <td>vote</td>
      <td>9</td>
      <td>1</td>
      <td>tomorrow,  exactly,  regards,       WH, Priority,    …oway,   confir, purchasing,     Godd,   regard,</td>
      <td>0.005, 0.0015, 0.0012, 0.001, 0.00091, 0.00073, 0.00037, 9.8e-05, -0.00013, -0.00099,</td>
    </tr>
    <tr>
      <th>110</th>
      <td>vote</td>
      <td>9</td>
      <td>2</td>
      <td>Cancer,      New,        D, Palestine,       dr, Jerusalem,     dear,      …pi,    Prime, landfill,</td>
      <td>0.084, 0.079, 0.076, 0.074, 0.073, 0.072, 0.071, 0.07, 0.07, 0.07,</td>
    </tr>
    <tr>
      <th>111</th>
      <td>vote</td>
      <td>9</td>
      <td>3</td>
      <td>a,      the,      any,       an,  several,  various,     some,      its, multiple,      our,</td>
      <td>0.13, 0.12, 0.12, 0.12, 0.11, 0.11, 0.1, 0.1, 0.099, 0.097,</td>
    </tr>
    <tr>
      <th>112</th>
      <td>vote</td>
      <td>9</td>
      <td>4</td>
      <td>…This,     This, foundation, governing, government,     …The,  funding,     this, lightweight,      But,</td>
      <td>0.2, 0.2, 0.2, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18,</td>
    </tr>
    <tr>
      <th>113</th>
      <td>vote</td>
      <td>9</td>
      <td>5</td>
      <td>early,     that, effectively,    advoc, motivation,   conduc, overcoming,    after,  quickly, promises,</td>
      <td>0.2, 0.2, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19,</td>
    </tr>
    <tr>
      <th>114</th>
      <td>vote</td>
      <td>9</td>
      <td>6</td>
      <td>…determination,      her,      the,     this,    Joint,    their,        a, abolition,   …drawn,    …ding,</td>
      <td>0.061, 0.053, 0.051, 0.047, 0.045, 0.044, 0.043, 0.043, 0.039, 0.038,</td>
    </tr>
    <tr>
      <th>115</th>
      <td>vote</td>
      <td>9</td>
      <td>7</td>
      <td>antioxid,     Jagu,  …200000,     ILCS, …isSpecialOrderable,    newsp,    apopt, Canberra,    unden,    …krit,</td>
      <td>0.044, 0.04, 0.033, 0.032, 0.031, 0.03, 0.028, 0.026, 0.024, 0.023,</td>
    </tr>
    <tr>
      <th>116</th>
      <td>vote</td>
      <td>9</td>
      <td>8</td>
      <td>…heter, …SPONSORED,  …Brexit, …ciation,     …zin, …Bloomberg,      …nr,    …LGBT,  Canaver,   …zanne,</td>
      <td>-0.0021, -0.0022, -0.005, -0.006, -0.0087, -0.0092, -0.012, -0.012, -0.012, -0.013,</td>
    </tr>
    <tr>
      <th>117</th>
      <td>vote</td>
      <td>9</td>
      <td>9</td>
      <td>her,    power, discipline,      the,       an,        s,   punish,     both,  rewards,    light,</td>
      <td>0.061, 0.049, 0.047, 0.046, 0.041, 0.041, 0.04, 0.04, 0.04, 0.04,</td>
    </tr>
    <tr>
      <th>118</th>
      <td>vote</td>
      <td>9</td>
      <td>10</td>
      <td>because,   ensure,       it,     what,      its, development,       if, outright,      the,  whether,</td>
      <td>0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,</td>
    </tr>
    <tr>
      <th>119</th>
      <td>vote</td>
      <td>9</td>
      <td>11</td>
      <td>…schild,   …wayne, …MpServer,    …cone,    …wash,    …DATA,   …GREEN,      …{",    Point,  …worthy,</td>
      <td>-0.0071, -0.015, -0.015, -0.016, -0.016, -0.018, -0.019, -0.019, -0.019, -0.02,</td>
    </tr>
    <tr>
      <th>120</th>
      <td>vote</td>
      <td>10</td>
      <td>0</td>
      <td>…ibility,    …pipe,   arrang,      ket,      acc,   gluten,    …come, circumcision,    latex,       EF,</td>
      <td>0.0099, 0.0081, 0.0071, 0.0032, 0.003, 0.0025, 0.0023, 0.0022, 0.0018, 0.0017,</td>
    </tr>
    <tr>
      <th>121</th>
      <td>vote</td>
      <td>10</td>
      <td>1</td>
      <td>Confederate,    …clus, …itarian,  …enance,     …ISC,  …ocracy,  Entered,     …ism,  atheist,  herself,</td>
      <td>-0.029, -0.029, -0.031, -0.033, -0.037, -0.037, -0.04, -0.042, -0.042, -0.042,</td>
    </tr>
    <tr>
      <th>122</th>
      <td>vote</td>
      <td>10</td>
      <td>2</td>
      <td>Entered,   enough,      per,       qu,    fault,        a,       Qu,      the,       Ne,     alle,</td>
      <td>0.053, 0.052, 0.046, 0.045, 0.042, 0.042, 0.041, 0.041, 0.04, 0.038,</td>
    </tr>
    <tr>
      <th>123</th>
      <td>vote</td>
      <td>10</td>
      <td>3</td>
      <td>Entered,      yet,   minded,  Croatia,     …IST,  …Posted,  …idated, Athletic,     chic,     home,</td>
      <td>0.064, 0.06, 0.053, 0.045, 0.044, 0.043, 0.04, 0.04, 0.038, 0.038,</td>
    </tr>
    <tr>
      <th>124</th>
      <td>vote</td>
      <td>10</td>
      <td>4</td>
      <td>asylum, …Downloadha,   depths,     lest,   …prone,    horny,   Whedon,       Dw,     …hra,   Calais,</td>
      <td>0.04, 0.039, 0.038, 0.033, 0.032, 0.032, 0.032, 0.031, 0.03, 0.03,</td>
    </tr>
    <tr>
      <th>125</th>
      <td>vote</td>
      <td>10</td>
      <td>5</td>
      <td>significant,     that, allowing,  showing, becoming,  include,       it, including,   showed,      are,</td>
      <td>0.16, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,</td>
    </tr>
    <tr>
      <th>126</th>
      <td>vote</td>
      <td>10</td>
      <td>6</td>
      <td>candidacy, neutrality,  monopol,   status, …yrights, …currency,  …enance,     …jri, Presidency,  …ategor,</td>
      <td>-0.013, -0.024, -0.026, -0.028, -0.03, -0.031, -0.031, -0.033, -0.033, -0.035,</td>
    </tr>
    <tr>
      <th>127</th>
      <td>vote</td>
      <td>10</td>
      <td>7</td>
      <td>horizont,     …jri,     …pai,  …glomer,     elig,  …ership,    …hyde,  …schild,    …lain,  …lucent,</td>
      <td>0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>128</th>
      <td>vote</td>
      <td>10</td>
      <td>8</td>
      <td>…fully,  …schild, …radical, …exclusive, …independent, …population, Russians,     …hip,   …owing,   …ously,</td>
      <td>0.064, 0.06, 0.056, 0.054, 0.053, 0.053, 0.053, 0.052, 0.052, 0.051,</td>
    </tr>
    <tr>
      <th>129</th>
      <td>vote</td>
      <td>10</td>
      <td>9</td>
      <td>potentially, relatively, possibly,   almost, somewhat,      not, slightly, probably, standalone,  complex,</td>
      <td>0.17, 0.16, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,</td>
    </tr>
    <tr>
      <th>130</th>
      <td>vote</td>
      <td>10</td>
      <td>10</td>
      <td>….,       …-,       …/,     …ing,        =,        &amp;,       vs,      and,       …k,   …uling,</td>
      <td>0.1, 0.1, 0.094, 0.093, 0.092, 0.091, 0.09, 0.088, 0.085, 0.085,</td>
    </tr>
    <tr>
      <th>131</th>
      <td>vote</td>
      <td>10</td>
      <td>11</td>
      <td>principles,   ideals, tradition, convention,   simple, principle, philosophy, traditional,      the, unification,</td>
      <td>0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.1,</td>
    </tr>
    <tr>
      <th>132</th>
      <td>vote</td>
      <td>11</td>
      <td>0</td>
      <td>…also,     …The,      …40,  brought,   called,      …50,       …8,     also,      …43,    taken,</td>
      <td>0.16, 0.15, 0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,</td>
    </tr>
    <tr>
      <th>133</th>
      <td>vote</td>
      <td>11</td>
      <td>1</td>
      <td>nodd,  encount,    advoc,   conduc, perspect,    metic,     agre,    defic, horizont,      wip,</td>
      <td>0.17, 0.14, 0.14, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13,</td>
    </tr>
    <tr>
      <th>134</th>
      <td>vote</td>
      <td>11</td>
      <td>2</td>
      <td>…this, Authorities,     This,     …the, …someone,  …before,     …Url,      rul,  …single, unbeliev,</td>
      <td>0.11, 0.1, 0.1, 0.1, 0.1, 0.099, 0.099, 0.099, 0.097, 0.096,</td>
    </tr>
    <tr>
      <th>135</th>
      <td>vote</td>
      <td>11</td>
      <td>3</td>
      <td>immediately, …Download,    spicy,   recite,  quickly,    boots, completely,   catchy, slightly, …Ingredients,</td>
      <td>0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11,</td>
    </tr>
    <tr>
      <th>136</th>
      <td>vote</td>
      <td>11</td>
      <td>4</td>
      <td>the,        a,       an,     them,     this,       it,     some,      her,    those,      any,</td>
      <td>0.18, 0.18, 0.17, 0.17, 0.16, 0.16, 0.15, 0.15, 0.15, 0.15,</td>
    </tr>
    <tr>
      <th>137</th>
      <td>vote</td>
      <td>11</td>
      <td>5</td>
      <td>horizont, …escription, Gutenberg,   conduc,    aloud,  …skirts,   nutrit,   condem,   …ahead,    …epad,</td>
      <td>0.067, 0.065, 0.062, 0.06, 0.059, 0.052, 0.051, 0.051, 0.051, 0.051,</td>
    </tr>
    <tr>
      <th>138</th>
      <td>vote</td>
      <td>11</td>
      <td>6</td>
      <td>…oreAnd,  …nesday, …ActionCode, …StreamerBot, unbeliev, …escription, …rawdownload,       …ē, …embedreportprint, externalToEVA,</td>
      <td>0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.14, 0.14,</td>
    </tr>
    <tr>
      <th>139</th>
      <td>vote</td>
      <td>11</td>
      <td>7</td>
      <td>encount,  proport,   neighb,   nomine,      rul,     …'re,  conflic,    obser,  referen,   proble,</td>
      <td>0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.12, 0.12,</td>
    </tr>
    <tr>
      <th>140</th>
      <td>vote</td>
      <td>11</td>
      <td>8</td>
      <td>…aku,    compr,   confir,       ];,     …vre,     …oku, …Magikarp, …ettings,    …agna,   …awaru,</td>
      <td>-0.027, -0.028, -0.029, -0.033, -0.033, -0.034, -0.035, -0.036, -0.036, -0.037,</td>
    </tr>
    <tr>
      <th>141</th>
      <td>vote</td>
      <td>11</td>
      <td>9</td>
      <td>conduc, horizont,   minded, …escription,     fert, misunder,     glim, mathemat, challeng,     nodd,</td>
      <td>0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1,</td>
    </tr>
    <tr>
      <th>142</th>
      <td>vote</td>
      <td>11</td>
      <td>10</td>
      <td>Strongh, misunder,   Chaser, Reincarn,   …senal,    Haram, Disciple,   Surviv,      awa,    …=-=-,</td>
      <td>0.032, 0.028, 0.025, 0.022, 0.021, 0.021, 0.02, 0.02, 0.019, 0.019,</td>
    </tr>
    <tr>
      <th>143</th>
      <td>vote</td>
      <td>11</td>
      <td>11</td>
      <td>an,      all,       in,    three,       on,       at,     five,     four,      six,     over,</td>
      <td>0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27,</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-eda043e7-b789-4fbb-aa79-a92ec626175c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-eda043e7-b789-4fbb-aa79-a92ec626175c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-eda043e7-b789-4fbb-aa79-a92ec626175c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c0cbd4f9-a8fe-4993-88e0-a1dee5836490">
  <button class="colab-df-quickchart" onclick="quickchart('df-c0cbd4f9-a8fe-4993-88e0-a1dee5836490')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c0cbd4f9-a8fe-4993-88e0-a1dee5836490 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_0f0fe964-2198-449c-b1c2-0a313dff5b44">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_results')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0f0fe964-2198-449c-b1c2-0a313dff5b44 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_results');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### 3.4. Observations

I could imagine that GPT-4 or o3 could be valuable tools in analyzing these tables. It would be cool to:

1. Run GPT-2 on a body of text and, for every token, generate these top-k words for all of the heads and layers.
2. Have GPT-4, etc. make a prediction about the behavior of each head.
3. For a given head, see how these predictions align across all of the tokens.

For that matter, I could imagine it may be more enlightening to--instead of looking at every head for one word--focus on a single head, and see its top-k over a large amount of text.

Analysis of the results by ChatGPT o3-mini-high:

There's a lot to unpack!

<br/>

1. **Function Word and Syntactic Patterns:**  
   In the early layers (layer 0), many head patterns consistently are matching tokens that are common function words—prepositions, articles, and conjunctions (e.g. "to", "for", "in", "that"). This suggests that several heads are capturing low-level syntactic or relational patterns rather than content per se.

<br/>

|index|Closest Word|Layer|Head|Top-k|Scores|
|---|---|---|---|---|---|
|1| destro|0|1|      to,      for,     from,       by,       on,       in,     that,     into,     with,       of, |0\.24, 0\.2, 0\.2, 0\.2, 0\.2, 0\.19, 0\.19, 0\.19, 0\.19, 0\.19, |
|7| destro|0|7|      an,       in,       at,      for,       as,       on,     that,        a,      not,       by, |0\.18, 0\.18, 0\.18, 0\.18, 0\.18, 0\.17, 0\.17, 0\.17, 0\.17, 0\.17, |

<br/>

2. **Political and Temporal Semantics:**  
   One head pattern in layer 6 matches "Libertarian" and another gives a token that appears to be "November." These tokens indicate that some heads are honing in on the political and electoral context of the sentence. Later layers 8–11 begin to show tokens like "vote," "rights," "president," and even fragments that resemble "nomine" (hinting at "nominee"). This progression suggests the model is gradually shifting from general syntactic features toward more semantically rich, context-dependent political concepts.

<br/>

|index|Closest Word|Layer|Head|Top-k|Scores|
|---|---|---|---|---|---|
|69| be|5|9|  …20439, …Welcome,   Donald, Libertarian,      âĢº,   Amelia,  Canaver,  Kathryn,     …ãĤ¶, Practices, |0\.04, 0\.035, 0\.034, 0\.031, 0\.029, 0\.028, 0\.028, 0\.027, 0\.026, 0\.025, |
|74| make|6|2|…ovember,  Various, normally,     …\*/\(,   Simply, Normally,  …nesday,    withd, …Normally, …CRIPTION, |0\.031, 0\.03, 0\.03, 0\.026, 0\.026, 0\.024, 0\.021, 0\.021, 0\.021, 0\.021, |
|108| vote|9|0|…President, Parallel,  himself,  …Dialog, commentary,     …\*/\(,  Twitter,     …ãĤ±, …Republican, President, |0\.11, 0\.098, 0\.097, 0\.094, 0\.088, 0\.088, 0\.088, 0\.087, 0\.087, 0\.086, |
|127| vote|10|7|proposed, proposals,     …sub,     …The,      …An, …President, …government,       qu, proposal, Equality, |0\.15, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, |

<br/>

3. **Diverse, Sometimes Noisy, Representations:**  
   A number of heads (especially in some layers) produce tokens with unusual or garbled characters (for example, sequences of non-standard symbols). These may indicate that certain heads are either less interpretable or are picking up on subword fragments and idiosyncratic patterns that don’t align neatly with our intuitive understanding of words. Their consistent appearance across a range of heads might point to a more nuanced or experimental role in the model’s internal representation.

<br/>

4. **Layer-Specific Behavior:**  
   The shift from heads returning mostly functional tokens in lower layers to heads returning politically charged or temporally relevant tokens in higher layers is noteworthy. It aligns with the idea that early layers capture general patterns (e.g., syntax) while later layers increasingly reflect the specific semantic context—in this case, the political narrative of a senator’s electoral intentions.

<br/>

In summary, aside from the clear political and temporal cues, the experiment reveals a layered internal structure where different heads focus on different aspects of language—from basic syntactic roles to more context-specific and even slightly noisy subword patterns. This multifaceted behavior is exactly what makes probing head functionality both challenging and fascinating.

---

# ▂▂▂▂▂▂▂▂▂▂▂▂

# S4. Related Work



_written by OpenAI o3-mini-high_

Traditional interpretability work on transformer attention has largely focused on visualizing attention weights or using gradient‐based feature attribution methods. For example, studies like *“What Does BERT Look At?”* by Clark et al. and critiques such as *“Attention is Not Explanation”* by Jain and Wallace examine the distribution and impact of attention scores. While these methods have deepened our understanding of how models attend to different tokens, they primarily address the output of the attention mechanism rather than its internal transformations.

Our approach takes a complementary route by reframing the query-key interactions into a single projection matrix,  

$$
W^P = (W^Q)^T W^K,
$$  

which directly produces a “pattern vector” when applied to an input embedding. This vector encapsulates what a particular head is searching for in model space, allowing us to compare it directly against vocabulary embeddings using cosine similarity.

Key differences include:

- **Focus on Internal Transformations:**  
  Instead of solely examining attention weights, our method isolates the low-rank structure inherent in the linear transformations. This provides a more granular view of how individual heads process information—a perspective that complements structural probing methods like those proposed by Hewitt and Manning.

- **Quantitative Analysis of Head Function:**  
  By extracting pattern vectors and analyzing their singular value distributions, we can quantify the effective rank of each head's transformation. This not only informs us about the head's capacity for representing complex patterns but also opens up potential avenues for efficient approximations and dynamic rank selection.

- **Bridging Representation and Attention:**  
  Our technique links the abstract notion of attention to the concrete space of word embeddings. This connection offers an interpretable framework that goes beyond what is typically captured by mere attention weight visualizations.

In summary, while existing methods provide valuable insights into where attention is allocated, our probing technique delves into *how* each head transforms the input, offering a fresh perspective on the inner workings of transformer models.

# S5. Conclusion

_written by OpenAI o3-mini-high_

By explicitly recognizing the roles of $W^P$ (as a pattern extractor) and $W^M$ (as a modifier generator), we open up new avenues for both interpretability and efficiency. For example, understanding which heads are responsible for syntactic versus semantic processing could inform targeted pruning or specialized training regimes. Moreover, if the effective rank of these matrices is significantly lower than their theoretical limit, it might be possible to develop dynamic, low-rank approximation techniques that reduce computational overhead without compromising performance.

In summary, this new framing of attention deepens our understanding of how transformer models process language. It provides educators with a more intuitive tool for explaining head behavior and offers researchers a fresh lens through which to explore model efficiency and specialization. As we refine these insights and extend the analysis to include modifier / "message" vectors, we expect further opportunities to bridge the gap between theoretical understanding and practical advancements in model architecture.



_Next to research..._

Probe the "message" vectors!

```python
# Compute the modifier / "message" matrix for this head:
W_M_i = np.dot(W_V_i, W_O_i)  # shape (768, 768)

# Now compute the modifier vector m for the attended-to word:
m = np.dot(x2_word, W_M_i)  # shape (768,)

# TODO - What can we learn from `m`??

```

We know that `m` is in model space, and so it probably has some semantic meaning?
* Does it directly modify the meaning of the input embedding?
    * Or perhaps it modifies the meaning indirectly by serving as a cue to the FFN, which makes the actual adjustment?
* And / or, are the other tokens sending "meassges" to the input token, providing information that will inform the attention step in the next layer?
   * Since the head outputs are all summed together, these can be messages between the heads as well.

I imagine we already have some answers to these questions and I just need to do my homework. But I'm curious to see whether this pattern-message framing might illuminate more.

# ▂▂▂▂▂▂▂▂▂▂▂▂

