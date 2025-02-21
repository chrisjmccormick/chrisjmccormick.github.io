---
layout: post
title:  "Inspecting Head Behavior through Pattern Projections"
date:   2025-02-20 17:00:00 -0800
comments: true
image:
tags: Machine Learning, Transformers, Attention Mechanism, NLP, LLMs, Rank Factorization, Low-Rank Attention, Multi-Head Attention, Optimization
---

# Inspecting Head Behavior through Pattern Projections



In this post, I explore a novel way to interpret attention head functionality in transformer models by reformulating the Query and Key projections into a single matrix, $W^P = (W^Q)^T W^K$. This formulation allows us to view each head not just as a black-box attention unit but as a distinct pattern extractor operating in model space. By projecting an input embedding onto $W^P$, we obtain a “pattern vector” that can be directly compared to vocabulary embeddings using cosine similarity. This method opens up a new avenue for understanding what each head is “searching for” in the input sequence.

Rather than attempting to provide a definitive explanation of attention, the aim here is to demonstrate the potential of this approach through some initial experiments. I’ll start by applying the method to BERT’s early layers—where patterns tend to be more syntactic and less confounded by positional or deeper semantic factors—and then extend the analysis to GPT‑2. The following code shows how to extract these pattern projection matrices and leverage them to probe head behavior.

By offering an alternative, low-rank perspective on head behavior, I hope to provide both an accessible teaching tool and a stepping stone toward more advanced interpretability research. Let’s dive into the code and see what insights we can uncover!

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


Next, a function for constructing the Pattern projection matrix for a given head in a given layer of BERT.

Get the $W^P_i$ matrix for the specified layer and head in BERT.


```python
def get_BERT_WP(layer, head):
    # Extract W^Q and W^K for the chosen head
    W_Q = model.encoder.layer[layer].attention.self.query.weight.detach().cpu().numpy()
    W_K = model.encoder.layer[layer].attention.self.key.weight.detach().cpu().numpy()

    # Extract just the slice for this head
    head_size = W_Q.shape[0] // num_heads
    W_Q_i = W_Q[head * head_size:(head + 1) * head_size, :]
    W_K_i = W_K[head * head_size:(head + 1) * head_size, :]

    # Compute W^P for this head (Transposing W^Q_i first)
    W_P_i = np.dot(W_Q_i.T, W_K_i)  # Shape (768, 768)

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

Skimming through, here were some interesting examples:


**Disambiguating Heads**

These all seem to be examples where the head is looking for a context word to clarify the right meaning of the input word.


* For example, when given the word "run", these two heads appear to look for context for it:
    * Layer 0, head 1:
        * election, innings, theatre, mayor, sales, reelected, theaters, selling, wickets, theater, commercials, electoral, elections, gallons, elected,
    * Layer 1, head 3:
        * pitcher, home, inning, goalkeeper, wickets, schumacher, pitchers, bowler, baseball, wicket, shortstop, nfl, mlb, ##holder, drivers,

* I noticed a number of examples of this behavior in Layer 0, head 2:
    * "dog" --> hot, sent, watch, guard, radio, guide, send, hound, unsuccessfully, voice, neck, sends, success, feel, mas,
    * "bed" --> truck      river    playoff      creek      speed       flow    vehicle    lecture       fish     stream     flower    thunder      drain     narrow        dry
    * "drive" --> disk       disc       leaf      flash       club   magnetic      wheel       gene    reverse        rip       data      blood commercially    serpent    captive


**Unsure...**

_"happy"_
* Layer 0, head 3:
    * make, making, made, makes, people, not, ##made, women, ##ria, something, felt, city, dee, men, paper,
* Layer 3, head 8:
    * picture, faces, genesis, aftermath, emotional, expression, concurrently, pictures, expressions, emotion, hearts, account, jasper, mental, disorders,


**Special Tokens**

Many of the results were a pattern closely resembling the special tokens. (Matching the finding in "What Does BERT Look At?", [pdf](https://arxiv.org/pdf/1906.04341))

For example, layer 1 head 1,

"couch" --> [CLS], [MASK], [SEP], ##⁄, ##rricular, ##fully, ##vances, ##ostal, pmid, ##⋅, ##atable, ##tained, ##lessly, ##genase, ##ingly,
            
With cosine similarities 0.55, 0.29, 0.23, 0.17, 0.14, ...


**Self-Attending**

Some patterns matched the input word and its synonyms, implying the head is attending to the input token.

Layer 2, head 6,

"couch" --> couch, sofa, lagoon, ##ppel,

With cosine similarities: 0.23, 0.2, 0.17, 0.17


```python
# Select a sample set of words
words = ["couch", "dog", "run", "happy"]

# Layers to process (0 and 1)
layers = [0, 1, 2, 3]
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

display(df_results)

df_results.to_csv("bert_head_results.csv")
```



  <div id="df-c3c34e7c-bba8-4b20-9925-f46e89336eb2" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-c3c34e7c-bba8-4b20-9925-f46e89336eb2')"
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
        document.querySelector('#df-c3c34e7c-bba8-4b20-9925-f46e89336eb2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c3c34e7c-bba8-4b20-9925-f46e89336eb2');
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


<div id="df-a8e01806-e854-415d-9021-8210d48599b6">
  <button class="colab-df-quickchart" onclick="quickchart('df-a8e01806-e854-415d-9021-8210d48599b6')"
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
        document.querySelector('#df-a8e01806-e854-415d-9021-8210d48599b6 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_8fd1cec7-c8e2-4c3a-b0f0-ac002dee2b1c">
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
        document.querySelector('#id_8fd1cec7-c8e2-4c3a-b0f0-ac002dee2b1c button.colab-df-generate');
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
    
    


# ▂▂▂▂▂▂▂▂▂▂▂▂

# S2. GPT-2: Layer-Wise Evolution of an Embedding


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

    # Apply the final layer normalization to each hidden state to bring it into the embedding space.
    if i == len(hidden_states) - 1:
        # For the final layer, the normalization has already been applied.
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
    Dot product with predicted token ' floor': 48.647
    Top 5 similar tokens to hidden state:
      destro      81.759
      mathemat    80.826
      livest      79.613
      challeng    78.476
      …theless    78.156
    
    Layer 1:
    Dot product with predicted token ' floor': 0.726
    Top 5 similar tokens to hidden state:
      same        11.226
      latter      8.338
      first       8.058
      world       7.621
      last        7.551
    
    Layer 2:
    Dot product with predicted token ' floor': 1.161
    Top 5 similar tokens to hidden state:
      same        13.096
      latter      9.439
      first       8.846
      world       8.409
      last        8.355
    
    Layer 3:
    Dot product with predicted token ' floor': 0.513
    Top 5 similar tokens to hidden state:
      same        11.183
      last        5.757
      world       5.405
      first       5.294
      latter      5.180
    
    Layer 4:
    Dot product with predicted token ' floor': 1.848
    Top 5 similar tokens to hidden state:
      same        10.353
      opposite    5.604
      last        4.869
      first       4.859
      next        4.608
    
    Layer 5:
    Dot product with predicted token ' floor': -0.502
    Top 5 similar tokens to hidden state:
      same        5.646
      opposite    1.171
      next        0.955
      table       0.894
      very        0.596
    
    Layer 6:
    Dot product with predicted token ' floor': -1.299
    Top 5 similar tokens to hidden state:
      same        2.140
      opposite    -0.933
      table       -1.194
      floor       -1.299
      board       -2.418
    
    Layer 7:
    Dot product with predicted token ' floor': -1.397
    Top 5 similar tokens to hidden state:
      same        1.074
      shoulders   -1.286
      table       -1.347
      floor       -1.397
      opposite    -1.717
    
    Layer 8:
    Dot product with predicted token ' floor': -6.399
    Top 5 similar tokens to hidden state:
      floor       -6.399
      edge        -7.088
      same        -7.228
      ground      -7.307
      table       -7.308
    
    Layer 9:
    Dot product with predicted token ' floor': -7.291
    Top 5 similar tokens to hidden state:
      ground      -7.132
      floor       -7.291
      table       -7.300
      edge        -7.660
      bottom      -8.409
    
    Layer 10:
    Dot product with predicted token ' floor': -9.594
    Top 5 similar tokens to hidden state:
      floor       -9.594
      bed         -11.009
      sofa        -11.136
      table       -11.299
      ground      -11.632
    
    Layer 11:
    Dot product with predicted token ' floor': -43.615
    Top 5 similar tokens to hidden state:
      floor       -43.615
      sofa        -44.381
      bed         -44.545
      couch       -44.738
      table       -45.060
    
    Layer 12:
    Dot product with predicted token ' floor': -80.597
    Top 5 similar tokens to hidden state:
      floor       -80.597
      bed         -80.720
      couch       -80.899
      ground      -81.091
      edge        -81.102
    
    Table of Top 5 Tokens (per layer):




  <div id="df-d6224b98-13ef-47e8-9a60-8af4731f4d31" class="colab-df-container">
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
      <td>destro</td>
      <td>same</td>
      <td>same</td>
      <td>same</td>
      <td>same</td>
      <td>same</td>
      <td>same</td>
      <td>same</td>
      <td>floor</td>
      <td>ground</td>
      <td>floor</td>
      <td>floor</td>
      <td>floor</td>
    </tr>
    <tr>
      <th>Rank 2</th>
      <td>mathemat</td>
      <td>latter</td>
      <td>latter</td>
      <td>last</td>
      <td>opposite</td>
      <td>opposite</td>
      <td>opposite</td>
      <td>shoulders</td>
      <td>edge</td>
      <td>floor</td>
      <td>bed</td>
      <td>sofa</td>
      <td>bed</td>
    </tr>
    <tr>
      <th>Rank 3</th>
      <td>livest</td>
      <td>first</td>
      <td>first</td>
      <td>world</td>
      <td>last</td>
      <td>next</td>
      <td>table</td>
      <td>table</td>
      <td>same</td>
      <td>table</td>
      <td>sofa</td>
      <td>bed</td>
      <td>couch</td>
    </tr>
    <tr>
      <th>Rank 4</th>
      <td>challeng</td>
      <td>world</td>
      <td>world</td>
      <td>first</td>
      <td>first</td>
      <td>table</td>
      <td>floor</td>
      <td>floor</td>
      <td>ground</td>
      <td>edge</td>
      <td>table</td>
      <td>couch</td>
      <td>ground</td>
    </tr>
    <tr>
      <th>Rank 5</th>
      <td>…theless</td>
      <td>last</td>
      <td>last</td>
      <td>latter</td>
      <td>next</td>
      <td>very</td>
      <td>board</td>
      <td>opposite</td>
      <td>table</td>
      <td>bottom</td>
      <td>ground</td>
      <td>table</td>
      <td>edge</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d6224b98-13ef-47e8-9a60-8af4731f4d31')"
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
        document.querySelector('#df-d6224b98-13ef-47e8-9a60-8af4731f4d31 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d6224b98-13ef-47e8-9a60-8af4731f4d31');
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


<div id="df-01e8e4ee-8f85-4a76-a2e3-248d0fa7bbbd">
  <button class="colab-df-quickchart" onclick="quickchart('df-01e8e4ee-8f85-4a76-a2e3-248d0fa7bbbd')"
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
        document.querySelector('#df-01e8e4ee-8f85-4a76-a2e3-248d0fa7bbbd button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_f465ee21-a001-49dc-946b-94d3b8e6e467">
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
        document.querySelector('#id_f465ee21-a001-49dc-946b-94d3b8e6e467 button.colab-df-generate');
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




  <div id="df-aa848645-e36e-46c3-b5bb-fe74251f6db4" class="colab-df-container">
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
      <td>81.759</td>
      <td>11.226</td>
      <td>13.096</td>
      <td>11.183</td>
      <td>10.353</td>
      <td>5.646</td>
      <td>2.140</td>
      <td>1.074</td>
      <td>-6.399</td>
      <td>-7.132</td>
      <td>-9.594</td>
      <td>-43.615</td>
      <td>-80.597</td>
    </tr>
    <tr>
      <th>Rank 2</th>
      <td>80.826</td>
      <td>8.338</td>
      <td>9.439</td>
      <td>5.757</td>
      <td>5.604</td>
      <td>1.171</td>
      <td>-0.933</td>
      <td>-1.286</td>
      <td>-7.088</td>
      <td>-7.291</td>
      <td>-11.009</td>
      <td>-44.381</td>
      <td>-80.720</td>
    </tr>
    <tr>
      <th>Rank 3</th>
      <td>79.613</td>
      <td>8.058</td>
      <td>8.846</td>
      <td>5.405</td>
      <td>4.869</td>
      <td>0.955</td>
      <td>-1.194</td>
      <td>-1.347</td>
      <td>-7.228</td>
      <td>-7.300</td>
      <td>-11.136</td>
      <td>-44.545</td>
      <td>-80.899</td>
    </tr>
    <tr>
      <th>Rank 4</th>
      <td>78.476</td>
      <td>7.621</td>
      <td>8.409</td>
      <td>5.294</td>
      <td>4.859</td>
      <td>0.894</td>
      <td>-1.299</td>
      <td>-1.397</td>
      <td>-7.307</td>
      <td>-7.660</td>
      <td>-11.299</td>
      <td>-44.738</td>
      <td>-81.091</td>
    </tr>
    <tr>
      <th>Rank 5</th>
      <td>78.156</td>
      <td>7.551</td>
      <td>8.355</td>
      <td>5.180</td>
      <td>4.608</td>
      <td>0.596</td>
      <td>-2.418</td>
      <td>-1.717</td>
      <td>-7.308</td>
      <td>-8.409</td>
      <td>-11.632</td>
      <td>-45.060</td>
      <td>-81.102</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-aa848645-e36e-46c3-b5bb-fe74251f6db4')"
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
        document.querySelector('#df-aa848645-e36e-46c3-b5bb-fe74251f6db4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-aa848645-e36e-46c3-b5bb-fe74251f6db4');
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


<div id="df-672eee7e-e3b4-4de0-a62c-63a5b00cf4d8">
  <button class="colab-df-quickchart" onclick="quickchart('df-672eee7e-e3b4-4de0-a62c-63a5b00cf4d8')"
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
        document.querySelector('#df-672eee7e-e3b4-4de0-a62c-63a5b00cf4d8 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_638458c1-32b7-493b-9cc1-ccee16c6ed12">
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
        document.querySelector('#id_638458c1-32b7-493b-9cc1-ccee16c6ed12 button.colab-df-generate');
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

### 3.1. Extracting Pattern Projection Matrices for GPT-2

We start by defining a function to extract the $W^P$ matrices for specific heads in GPT-2. Since GPT-2 is a decoder model, it uses causal self-attention, but the core extraction process is similar.


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

### 3.2. Comparing GPT-2 Patterns to Vocabulary Embeddings

Now, we extend our prior method by applying layer normalization to the pattern vectors before computing their similarity to the vocabulary embeddings.


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

    # Retrieve top-k matches
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

### 3.3. Run Sequence


```python
import torch

# Example sentence.
#input_text = " The cat sat on the"
input_text = " While formerly a Democrat, in next year's election, the senator intends to"
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

    Predicted next token:  run


### 3.3. Probing GPT-2 Heads

Compare the pattern vector for a hidden state to the vocabulary.


```python
num_heads = model.config.n_head

# Store results
results = []

# For each layer...
for layer_i, hs in enumerate(hidden_states):

    # Get the last token's hidden state.
    last_token_state = hs[0, -1, :]

    print("Layer", layer_i)

    # As a general sanity check, ensure this dot product yields the same
    # as the prior example.
    # Direct dot product similarity with the predicted token embedding.
    state = model.transformer.ln_f(last_token_state).detach().cpu().numpy()
    dot_sim = state.dot(predicted_emb)
    print(f"   Dot product with predicted token '{predicted_token}': {dot_sim:.3f}")

    # Find the current most similar word to the hidden state. It will gradually
    # become more like the predicted word, as we saw in section 2.
    sims = dot_product_similarity(state, lm_head_embeddings)
    top_indices = sims.argsort()[-1:][::-1]
    closest_word = tokenizer.convert_ids_to_tokens(int(top_indices[0])).replace("Ġ", " ")
    print(f"   Closest word:", closest_word, f"({sims[top_indices[0]]:.3f})")
    print()

    # ======== Analyze Head Patterns ========

    # Hidden state of token to predict...
    last_token_state = last_token_state.detach().cpu().numpy()

    # For each of the heads...
    for head in range(num_heads):

        # Get the pattern matrix
        W_P_i = get_GPT2_WP(layer_i, head)

        # Match the head pattern to the vocabulary.
        matches = find_head_matches_GPT2(W_P_i, last_token_state, k=10)

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

    # TODO - Not sure what's up here.
    if layer_i == 11:
        break

# Convert results to DataFrame and display
df_results = pd.DataFrame(results)

# Set pandas precision to 3 decimal points
pd.options.display.float_format = '{:.3f}'.format

display(df_results)

```

    Layer 0
       Dot product with predicted token ' run': 73.287
       Closest word:  destro (112.097)
    
    Layer 1
       Dot product with predicted token ' run': 4.512
       Closest word:  be (10.018)
    
    Layer 2
       Dot product with predicted token ' run': 0.683
       Closest word:  be (7.008)
    
    Layer 3
       Dot product with predicted token ' run': 2.225
       Closest word:  be (6.327)
    
    Layer 4
       Dot product with predicted token ' run': -3.593
       Closest word:  be (1.300)
    
    Layer 5
       Dot product with predicted token ' run': -6.860
       Closest word:  be (-2.620)
    
    Layer 6
       Dot product with predicted token ' run': -13.255
       Closest word:  make (-9.850)
    
    Layer 7
       Dot product with predicted token ' run': -15.505
       Closest word:  retire (-11.693)
    
    Layer 8
       Dot product with predicted token ' run': -20.814
       Closest word:  vote (-15.865)
    
    Layer 9
       Dot product with predicted token ' run': -26.565
       Closest word:  vote (-19.277)
    
    Layer 10
       Dot product with predicted token ' run': -35.651
       Closest word:  vote (-27.857)
    
    Layer 11
       Dot product with predicted token ' run': -74.890
       Closest word:  vote (-72.214)
    




  <div id="df-f0c429d4-a42a-460b-b8b7-da1c0c9e2f04" class="colab-df-container">
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
      <td>destro</td>
      <td>0</td>
      <td>0</td>
      <td>…ò,    pione, …ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃ...</td>
      <td>0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.11...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>destro</td>
      <td>0</td>
      <td>1</td>
      <td>to,      for,     from,       by,       ...</td>
      <td>0.24, 0.2, 0.2, 0.2, 0.2, 0.19, 0.19, 0.19, 0....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>destro</td>
      <td>0</td>
      <td>2</td>
      <td>not,     that,       in,       at,       ...</td>
      <td>0.1, 0.096, 0.094, 0.094, 0.092, 0.09, 0.09, 0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>destro</td>
      <td>0</td>
      <td>3</td>
      <td>…ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃ...</td>
      <td>0.094, 0.094, 0.092, 0.092, 0.09, 0.09, 0.09, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>destro</td>
      <td>0</td>
      <td>4</td>
      <td>an,       in,       at, …ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃ...</td>
      <td>0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11, 0.11...</td>
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
      <th>139</th>
      <td>vote</td>
      <td>11</td>
      <td>7</td>
      <td>neighb,   …PDATE,       …Þ,    eleph,   nomi...</td>
      <td>0.12, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11...</td>
    </tr>
    <tr>
      <th>140</th>
      <td>vote</td>
      <td>11</td>
      <td>8</td>
      <td>proble,       '',     …oda, recently, report...</td>
      <td>0.0049, 0.0019, -0.0015, -0.004, -0.006, -0.00...</td>
    </tr>
    <tr>
      <th>141</th>
      <td>vote</td>
      <td>11</td>
      <td>9</td>
      <td>…escription, horizont, mathemat,     …ãĤ¯,   c...</td>
      <td>0.18, 0.17, 0.17, 0.16, 0.16, 0.16, 0.15, 0.15...</td>
    </tr>
    <tr>
      <th>142</th>
      <td>vote</td>
      <td>11</td>
      <td>10</td>
      <td>Chaser,   Surviv, …Firstly, …Interested, …Pr...</td>
      <td>0.03, 0.029, 0.029, 0.027, 0.026, 0.026, 0.024...</td>
    </tr>
    <tr>
      <th>143</th>
      <td>vote</td>
      <td>11</td>
      <td>11</td>
      <td>up,       on,       in,      out,     fo...</td>
      <td>0.28, 0.28, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27...</td>
    </tr>
  </tbody>
</table>
<p>144 rows × 5 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f0c429d4-a42a-460b-b8b7-da1c0c9e2f04')"
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
        document.querySelector('#df-f0c429d4-a42a-460b-b8b7-da1c0c9e2f04 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f0c429d4-a42a-460b-b8b7-da1c0c9e2f04');
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


<div id="df-07d3ad67-7803-485b-a9d8-c3f61f8c3450">
  <button class="colab-df-quickchart" onclick="quickchart('df-07d3ad67-7803-485b-a9d8-c3f61f8c3450')"
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
        document.querySelector('#df-07d3ad67-7803-485b-a9d8-c3f61f8c3450 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_0c43ea10-eaf3-4613-9010-313638bf8abc">
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
        document.querySelector('#id_0c43ea10-eaf3-4613-9010-313638bf8abc button.colab-df-generate');
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

Analysis of the results by ChatGPT o3-mini-high:

There's a lot to unpack!

1. **Function Word and Syntactic Patterns:**  
   In the early layers (layer 0), many head patterns consistently are matching tokens that are common function words—prepositions, articles, and conjunctions (e.g. "to", "for", "in", "that"). This suggests that several heads are capturing low-level syntactic or relational patterns rather than content per se.

|index|Closest Word|Layer|Head|Top-k|Scores|
|---|---|---|---|---|---|
|1| destro|0|1|      to,      for,     from,       by,       on,       in,     that,     into,     with,       of, |0\.24, 0\.2, 0\.2, 0\.2, 0\.2, 0\.19, 0\.19, 0\.19, 0\.19, 0\.19, |
|7| destro|0|7|      an,       in,       at,      for,       as,       on,     that,        a,      not,       by, |0\.18, 0\.18, 0\.18, 0\.18, 0\.18, 0\.17, 0\.17, 0\.17, 0\.17, 0\.17, |

2. **Political and Temporal Semantics:**  
   One head pattern in layer 6 matches "Libertarian" and another gives a token that appears to be "November." These tokens indicate that some heads are honing in on the political and electoral context of the sentence. Later layers 8–11 begin to show tokens like "vote," "rights," "president," and even fragments that resemble "nomine" (hinting at "nominee"). This progression suggests the model is gradually shifting from general syntactic features toward more semantically rich, context-dependent political concepts.

|index|Closest Word|Layer|Head|Top-k|Scores|
|---|---|---|---|---|---|
|69| be|5|9|  …20439, …Welcome,   Donald, Libertarian,      âĢº,   Amelia,  Canaver,  Kathryn,     …ãĤ¶, Practices, |0\.04, 0\.035, 0\.034, 0\.031, 0\.029, 0\.028, 0\.028, 0\.027, 0\.026, 0\.025, |
|74| make|6|2|…ovember,  Various, normally,     …\*/\(,   Simply, Normally,  …nesday,    withd, …Normally, …CRIPTION, |0\.031, 0\.03, 0\.03, 0\.026, 0\.026, 0\.024, 0\.021, 0\.021, 0\.021, 0\.021, |
|108| vote|9|0|…President, Parallel,  himself,  …Dialog, commentary,     …\*/\(,  Twitter,     …ãĤ±, …Republican, President, |0\.11, 0\.098, 0\.097, 0\.094, 0\.088, 0\.088, 0\.088, 0\.087, 0\.087, 0\.086, |
|127| vote|10|7|proposed, proposals,     …sub,     …The,      …An, …President, …government,       qu, proposal, Equality, |0\.15, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, 0\.14, |


3. **Diverse, Sometimes Noisy, Representations:**  
   A number of heads (especially in some layers) produce tokens with unusual or garbled characters (for example, sequences of non-standard symbols). These may indicate that certain heads are either less interpretable or are picking up on subword fragments and idiosyncratic patterns that don’t align neatly with our intuitive understanding of words. Their consistent appearance across a range of heads might point to a more nuanced or experimental role in the model’s internal representation.

4. **Layer-Specific Behavior:**  
   The shift from heads returning mostly functional tokens in lower layers to heads returning politically charged or temporally relevant tokens in higher layers is noteworthy. It aligns with the idea that early layers capture general patterns (e.g., syntax) while later layers increasingly reflect the specific semantic context—in this case, the political narrative of a senator’s electoral intentions.

In summary, aside from the clear political and temporal cues, the experiment reveals a layered internal structure where different heads focus on different aspects of language—from basic syntactic roles to more context-specific and even slightly noisy subword patterns. This multifaceted behavior is exactly what makes probing head functionality both challenging and fascinating.

---

## S4. Related Work



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

# S4. Conclusion

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

# ▂▂▂▂▂▂▂▂▂▂▂▂
