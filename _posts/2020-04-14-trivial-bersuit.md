---
layout: post
title:  "Trivial BERsuiT - How much trivia does BERT know?"
date:   2020-04-14 8:00:00 -0800
comments: true
image: /assets/BERT/bert_trivia_correct.png
tags: BERT, Transformers, Trivia, World Knowledge, Masked Language Model, BertForMaskedLM, Natural Language Processing, NLP
---

*by Chris McCormick*

As I've been doing all of this research into BERT, I've been really curious--just how much *trivia* does BERT know? We use BERT for it's impressive knowledge of language, but how many *factoids* are encoded in there along with all of the language understanding?

It turns out, kind of a lot! We're going to look at some fun examples in this post. 

Now, BERT can't generate text, so we can't actually ask it a question and have it generate a natural response. *But,* we can still test its knowledge by formatting our questions as "fill-in-the-blank"s.

# Contents
 
* TOC
{:toc}

# Part 1 - Let's Quiz BERT!

The code for having BERT answer questions is down in [Part 2](https://colab.research.google.com/drive/14YZpquVhOo78dFdbH8Fva9CNwLanqyBV#scrollTo=457VPa20fZzY) of this post/notebook, but let's start by looking at some examples!



1. "In ____, Christopher Columbus sailed across the ocean to discover the Americas."
    * **BERT**: "1492" - CORRECT
2. "The Second Punic War broke out in ___ after Hannibal's attack on Saguntum."
    * **BERT**: "218" - CORRECT
3. "The ______ Mountains divided Greece into isolated valleys."
    * **BERT**: "pindus" - CORRECT
4. "The Greek gods were said to reside atop _____________ in Greece."
    * **BERT**: "the olympus" - *WRONG*
        * It should be "mount olympus" -- pretty close, though!
5. "During the rise of Greek city-states, ____ replaced bronze."
    * **BERT**: "iron" - CORRECT
6. "___________ is called the "Father of Medicine"."
    * **BERT**: "hippocrates" - CORRECT
7. "During the Second Punic War, Hannibal famously led an army of war _________ across the Alps, although many of them perished in the harsh conditions."
    * **BERT**: "elephants" - CORRECT
8. "On December 21, 1864, General Sherman’s famous “March to the Sea” concluded with the capture of ________."
    * **BERT**: "atlanta" - *WRONG*
        * It should be "Savannah", but at least BERT predicted a southern city.
        * Seems like BERT has a pretty strong grasp of world history--let's try some other topics...
9. "On dress shirts, button-down, spread and tab are popular types of _______."
    * **BERT**: "button buttons" - *WRONG*
        *  Correct answer is "collars".
10. "1 + 1 = _"
    * **BERT**: "2" - CORRECT
11. "5 + 5 = __"
    * **BERT**: "5" - *WRONG*
        * Ok, so BERT's reading comprehension doesn't include the ability to perform basic math :)
12. "If you are having trouble with your computer, you should probably try _________ it."
    * **BERT**: "to to with" - *WRONG*
        * Correct answer is "rebooting". Apparently BERT doesn't know the first thing about providing IT support... 
        * BERT gets it right if you give it more help--"If you are having trouble with your computer, you should try turning ______ and back on again.", BERT correctly predicts "it off".
13. "The correct spelling of 'mispelled' is '__________'."
   * **BERT**: "mis -led" - *WRONG*
       * BERT came very close; it predicted 2 out of 3 of the tokens correctly: `['mis', '-', '##led']`. The middle token should be `'#spel'`.
14. "Super Bowl 50 was an American football game in which the ______________ defeated the Carolina Panthers 24–10 to earn their third Super Bowl title."
    * **BERT**: "dallas steelers" - *WRONG*
        * It was the Denver Broncos. Apparently BERT knows world history better than sports history.
15. "The Greek religion was ____________, meaning that they believed in many gods."
    * **BERT**: "polytheistic" - CORRECT
        * That's better! :)



I took my initial examples from this [set of flash cards](https://quizlet.com/295489702/world-history-fill-in-the-blank-flash-cards/) on world history, so that's why there's a disproportionate number about ancient Greece. I'm hoping you guys will try some of your own and share them!

In the rest of this post, I'll share some thoughts on why BERT is so good at this task, as well as the details of the implementation.

But before we do that, let's see if we can start some flame wars by asking BERT its opinion on a few very important matters.

1. "_________ has the best video game console."
    * **BERT**: "japan"
2. "Episode _ is the best of the original Star Wars Trilogy."
    * **BERT**: "iii"
3. "I prefer the ________ over the PlayStation 3."
    * **BERT**: "xbox 2"
        * I think BERT meant the 2nd generation Xbox, the "Xbox 360". Of course, it's a very leading question...
4. "James Cameron has made many great films, but his best is ____________."
    * **BERT**: "titanic"
        * Really BERT? You're picking the chick-flick over the one where an AI becomes sentient and subdues humanity?!
5. "I don't always drink beer, but when I do, I prefer _________."
    * **BERT**: "a and ofs".
        * I don't think BERT knows anything about beer, guys...
6. "Chris McCormick creates helpful illustrations and clear explanations of difficult subjects in ________________ and natural language processing."
    * **BERT**: "computer linguistics"
        * Well, thank you, BERT--that's very kind.


## Why it Works



*The Masked Language Model*

BERT is most exciting because of how well it learns to comprehend language, but clearly it has learned a lot of factoids or "world knowledge" as well!

This isn't surprising, though, given that "fill-in-blank" was exactly what BERT was trained on! 

For BERT's "Masked Language Model" (MLM) pre-training task, all of Wikipedia was fed through BERT (in chunks), and roughly *one in every six words* was replaced with the special `[MASK]` token. For each chunk of text, BERT's job was to predict the missing words.

And because Wikipedia was the source for the text, sometimes the masked words would be things like dates, names of people and places, or domain-specific terms. In those cases, to predict the right answer, general language understanding isn't enough. You need to have an education in history, or whichever subject the text is coming from.

> *Side Note:* BERT was trained on both Wikipedia (800M words) and the "BookCorpus" (2,500M words). I assumed the latter meant Google's collection of scanned books, but it's actually a collection of *self-published eBooks* taken from this [site](https://www.smashwords.com/)! I've shared more on this in the Appendix [here](https://colab.research.google.com/drive/14YZpquVhOo78dFdbH8Fva9CNwLanqyBV#scrollTo=bf_jvfYu8MG-).


It does seem like a waste for BERT to learn all of this *knowledge*, much of which probably has no relevance to your specific application. It's important to recognize, though, that a critical part of BERT's pre-training is the size of the corpus--it was trained on a corpus with over 3 billion words. 

Sure, it might be better to pre-train BERT on text from your own application area, but only if you have a dataset that's larger than all of Wikipedia! 

## The Token Count Problem

There are a couple caveats here which might limit BERT's usefulness for *actually competing* in a trivia game.

The first we've already mentioned--the question has to be posed as a fill-in-the-blank. Most quiz games instead pose a full question, and then you have to either state the answer or choose it from a list ("multiple choice").

The second issue is that, in order for BERT to accurately fill in the blank, it needs to know *how many tokens are in the answer*. And not just the number of words--*the number of tokens*--because the BERT tokenizer will break any out-of-vocabulary words into multiple subwords. 

For example, for the blank in the question, "The correct spelling of 'mispelled' is '__________'.", what's actually passed to BERT is '`[MASK] [MASK] [MASK]`' because the BERT tokenizer breaks "misspelled" into three subwords: `['mis', '#spel', '##led']`.

In general, though, my goal was not to create a trivia-solving bot, but rather to demonstrate that BERT does know a lot of trivia. For that purpose, telling it how many tokens to predict seems like a small enough concession.


# Part 2 - Source Code


In this section I've included my code for implementing the quiz questions. You can try it out on your own questions, and maybe experiment with different models!

## Setup

### Install 'transformers'

This example uses the `transformers` [library](https://github.com/huggingface/transformers/) by huggingface to interact with BERT, so we'll start by installing the package.


```python
!pip install transformers
```

    Collecting transformers
    [?25l  Downloading https://files.pythonhosted.org/packages/a3/78/92cedda05552398352ed9784908b834ee32a0bd071a9b32de287327370b7/transformers-2.8.0-py3-none-any.whl (563kB)
    [K     |████████████████████████████████| 573kB 8.6MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from transformers) (1.18.2)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers) (0.7)
    Collecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/99/50/93509f906a40bffd7d175f97fd75ea328ad9bd91f48f59c4bd084c94a25e/sacremoses-0.0.41.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 20.6MB/s 
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers) (3.0.12)
    Requirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from transformers) (1.12.38)
    Collecting tokenizers==0.5.2
    [?25l  Downloading https://files.pythonhosted.org/packages/d1/3f/73c881ea4723e43c1e9acf317cf407fab3a278daab3a69c98dcac511c04f/tokenizers-0.5.2-cp36-cp36m-manylinux1_x86_64.whl (3.7MB)
    [K     |████████████████████████████████| 3.7MB 48.2MB/s 
    [?25hRequirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers) (2019.12.20)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.6/dist-packages (from transformers) (4.38.0)
    Collecting sentencepiece
    [?25l  Downloading https://files.pythonhosted.org/packages/74/f4/2d5214cbf13d06e7cb2c20d84115ca25b53ea76fa1f0ade0e3c9749de214/sentencepiece-0.1.85-cp36-cp36m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 29.8MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers) (2.21.0)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.12.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (7.1.1)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (0.14.1)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers) (0.9.5)
    Requirement already satisfied: botocore<1.16.0,>=1.15.38 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers) (1.15.38)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers) (0.3.3)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2020.4.5.1)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2.8)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.6/dist-packages (from botocore<1.16.0,>=1.15.38->boto3->transformers) (2.8.1)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.16.0,>=1.15.38->boto3->transformers) (0.15.2)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.41-cp36-none-any.whl size=893334 sha256=64446adffae7aea1d1757b295469693ad074d411f4ae3fb50a869ea069e028ac
      Stored in directory: /root/.cache/pip/wheels/22/5a/d4/b020a81249de7dc63758a34222feaa668dbe8ebfe9170cc9b1
    Successfully built sacremoses
    Installing collected packages: sacremoses, tokenizers, sentencepiece, transformers
    Successfully installed sacremoses-0.0.41 sentencepiece-0.1.85 tokenizers-0.5.2 transformers-2.8.0


### 2. Load Pre-Trained BERT

I decided to use `BERT-large` for this Notebook--it's a *huge* model (24-layers and an embedding size of 1,024), but we won't need to perform any fine-tuning on it for this example, so we might as well use the large variant! 

To work with this model, we'll use the [BertForMaskedLM](https://huggingface.co/transformers/model_doc/bert.html?#bertformaskedlm) class from the `transformers` library. This "Masked Language Model" is what Google used to perform "pre-training" on BERT-large, so it's already been fine-tuned for us!

I'm also using the `whole-word-masking` variant of BERT. The original BERT masked individual tokens, which meant that sometimes the masked token was a subword within a word. More recently, the authors modified this task to ensure that all parts of any masked word are selected; this is a more difficult task and it improves the quality of the pre-trained model. 


```python
import torch
from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

```


    HBox(children=(IntProgress(value=0, description='Downloading', max=362, style=ProgressStyle(description_width=…


    



    HBox(children=(IntProgress(value=0, description='Downloading', max=1345000548, style=ProgressStyle(description…


    



    HBox(children=(IntProgress(value=0, description='Downloading', max=231508, style=ProgressStyle(description_wid…


    


I also tried out `ALBERT-xxlarge`. Compared to BERT-large, it got some answers wrong and some others right--so it didn't seem to me to be substantially better than BERT-large for this task. I don't have a formal benchmark here, though... 

If you decide to try ALBERT, note that ALBERT also uses whole-word masking, along with "n-gram masking", meaning it would pick multiple sequential words to mask out.


```python
#from transformers import AlbertForMaskedLM, AlbertTokenizer

#model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v1')
#tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v1')

#model = AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2')
#tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
```


```python
# Have the model run on the GPU.
desc = model.to('cuda')
```

## Retrieve Questions

I've defined a number of questions in a Google Spreadsheet [here](https://docs.google.com/spreadsheets/d/1zN4P-O6sNpATbEy7suAKhwyziWCxQ_XCnypxMzHZFR0/edit#gid=537013301)--currently there are about 50. The [Trivia Question Sources](https://colab.research.google.com/drive/14YZpquVhOo78dFdbH8Fva9CNwLanqyBV#scrollTo=XfgHLMExXyIE) section in the appendix lists some places that I've pulled from. 

The Google sheet is publicly viewable, but not editable--if you have more questions to add, send me a link to your own copy of the sheet and I'll pull them in. 



```python
import pandas as pd
import gspread
```


```python
from google.colab import auth

# Even though my sheet is publicly viewable, it seems that you still have to
# authorize `gspread`. 
auth.authenticate_user() 

from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default()) 
```


```python
# Open the spreadsheet by file ID.
spreadsheet = gc.open_by_key('1zN4P-O6sNpATbEy7suAKhwyziWCxQ_XCnypxMzHZFR0')

# Open the spreadsheet by name--only works if the file is already in your drive.
#spreadsheet = gc.open("Trivia Questions")

# Grab the first (and only) sheet.
sheet =  spreadsheet.get_worksheet(0)

# Parse into a pandas DataFrame!
df2 = pd.DataFrame(sheet.get_all_records())

# The 'ID' column is there to be used as an index.
df2 = df2.set_index('ID')

# Show the first few rows...
df2.head()
```




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
      <th>Question</th>
      <th>Answer</th>
      <th>Category</th>
      <th>Source</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>In 1492, Christopher Columbus sailed across th...</td>
      <td>1492</td>
      <td>World History</td>
      <td>Chris</td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>The Second Punic War broke out in 218 after Ha...</td>
      <td>218</td>
      <td>World History</td>
      <td>quizlet</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Pindus Mountains divided Greece into isola...</td>
      <td>Pindus</td>
      <td>World History</td>
      <td>quizlet</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Greek gods were said to reside atop Mount ...</td>
      <td>Mount Olympus</td>
      <td>World History</td>
      <td>quizlet</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>During the rise of Greek city-states, iron rep...</td>
      <td>iron</td>
      <td>World History</td>
      <td>quizlet</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



## Functions

This section defines the code for answering the questions, and for formatting the questions and answers in a fun way :)

### print_question

This function prints out the question, with the answer replaced by a "blank" (underscores).


```python
import textwrap

# Create a text-wrapper to constrain the question text to 80 characters.
wrapper = textwrap.TextWrapper(initial_indent="    ", 
                               subsequent_indent="    ", width = 80)

def print_question(q_orig, answer, show_mask = False):
    '''
    Prints out a question `q_orig` with the `answer` replaced by underscores.
    '''
    
    # Verify the answer is actually in the question string!
    if not answer in q_orig:
        print('Error -- answer not found in question!')
        return

    # Tokenize the answer--it may be broken into multiple words and/or subwords.
    answer_tokens = tokenizer.tokenize(answer)

    # Create the version of the sentence to display (with the answer removed).
    # Note: This is slightly different from the similar code in 
    # `answer_question` because we don't need to convert to lowercase here.
    if show_mask:
        # Replace the answer with the correct number of '[MASK]' tokens.
        hide_str = ' '.join(['[MASK]']*len(answer_tokens))
    else:
        # Replace the answer with underscores.
        hide_str = '_'*len(answer)

    # Replace the answer (with either underscores or mask tokens).
    q_disp = q_orig.replace(answer, hide_str)

    print('==== Question ====\n')

    # Print the question, with the answer removed.
    print(wrapper.fill(q_disp))

    print('')
```

### predict_answer


This function uses the BERT MLM model to try and "fill-in-the-blank".

I was glad to see that the MLM model *does* include the weights for the output classifier (which predicts the token). 





```python
import numpy as np

def predict_answer(q_orig, answer):
    '''
    Apply the BERT Masked LM to the question text to predict the answer tokens.
    Parameters:
      `q_orig` - The unmodified question text (as a string), with the answer 
                 still in place.
      `answer` - String containing the portion of the sentence to be masked out.
    '''
    # Tokenize the answer--it may be broken into multiple subwords.
    answer_tokens = tokenizer.tokenize(answer)

    # Create a sequence of `[MASK]` tokens to put in place of the answer.
    masks_str = ' '.join(['[MASK]']*len(answer_tokens))

    # Replace the answer with mask tokens.
    q_masked = q_orig.replace(answer, masks_str)

    # `encode` performs multiple functions:
    #   1. Tokenizes the text
    #   2. Maps the tokens to their IDs
    #   3. Adds the special [CLS] and [SEP] tokens.
    input_ids = tokenizer.encode(q_masked)

    # Find all indeces of the [MASK] token.
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    # ======== Choose Answer(s) ========
    model.eval()

    # List of tokens predicted by BERT.
    pred_tokens = []

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([input_ids])

    # Copy the input to the GPU.
    tokens_tensor = tokens_tensor.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        # Evaluate the model on the sentence.
        outputs = model(tokens_tensor)

        # Predictions will have shape:
        #  [1  x  sentence_length  x   vocab_size]
        #
        # e.g., torch.Size([1, 18, 30522])
        #
        # For a given word in the input text, the model produces a score for
        # every word in the vocabulary, and the word with the highest score 
        # is what we take as the predicted token. Note that the model does 
        # this for every word in the input text, not just the [MASK] token...
        predictions = outputs[0]

    # For each of the mask tokens...
    for masked_i in mask_token_indeces:

        # Get the scores corresponding to the word at psotion `masked_i` in the 
        # input text.
        vocab_scores = predictions[0, masked_i]

        # Use `argmax` to get the index of the highest score. `vocab_scores` has
        # the same length as the vocabulary, so this index is also the token ID
        # of the highest scoring word. 
        predicted_token_id = torch.argmax(vocab_scores).item()

        # Convert the token ID back to a string.
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
        
        # Add the token string to the list.
        pred_tokens.append(predicted_token)

    # ======== Recombine Tokens ========
    
    # Use the tokenizer to recombine tokens into words.
    combined = tokenizer.convert_tokens_to_string(pred_tokens)

    # Return both the list of token strings and the recombined answer string.
    return (pred_tokens, combined)
```

### print_answer

Prints BERT's answer and whether it's right or wrong. If BERT's answer is wrong, then this prints the correct answer, and the list of tokens predicted by BERT.


```python
def print_answer(answer, pred_answer, pred_tokens):
    
    print('==== BERT\'s Answer ====\n')
    
    # If the predicted answer is correct...
    # Note: The predicted answer will be lowercase...
    if (answer.lower() == pred_answer):
        print('    "' + pred_answer + '"  -  CORRECT!')
    
    # If it's wrong...
    else:
        # 
        print('    "' + pred_answer + '"  -  WRONG.\n')
        print('    Correct:    "' + answer + '"\n')
        print('    Tokens:     ', pred_tokens, '\n')
        
```

## Examples

### Single Question

*Ask a question by providing question and answer strings.*


```python
# Specify the question as a complete sentence (don't put in the blanks 
# yourself), and specify the "answer", the portion of the question which you 
# want to be masked out.
text = "Winston Churchill was the Prime Minister of the United Kingdom from 1940 to 1945, when he led Britain to victory in the Second World War."
answer = "Winston Churchill"

# Print the question.
print_question(text, answer)

# Predict the answer.
(tokens, pred_answer) = predict_answer(text, answer)

# Print and score the answer.
print_answer(answer, pred_answer, tokens)

```

    ==== Question ====
    
        _________________ was the Prime Minister of the United Kingdom from 1940 to
        1945, when he led Britain to victory in the Second World War.
    
    ==== BERT's Answer ====
    
        "winston churchill"  -  CORRECT!


*Ask a question from the Google spreadsheet by specifying its ID number.*


```python
# Retrieve a question using its ID.
q = df2.loc[9]

text = q['Question']
answer = str(q['Answer']) # Cast to string in case it's a number.

# Print the question.
print_question(text, answer)

# Predict the answer.
(tokens, pred_answer) = predict_answer(text, answer)

# Print and score the answer.
print_answer(answer, pred_answer, tokens)

```

    ==== Question ====
    
        ___________ is called the "Father of Medicine".
    
    ==== BERT's Answer ====
    
        "hippocrates"  -  CORRECT!


### Interactive Loop

I created this section for my YouTube video. It lets you iterate through all of the questions in the spreadsheet, answering them one at a time, using two cells.


```python
# Create an iterator to go through the questions.
# Run the next 2 cells repeatedly to iterate.
iter = df2.iterrows()
```

*Here's the question...*


```python
# Get the next question.
(i, q) = next(iter)

text = q['Question']
answer = str(q['Answer']) # Cast to string in case it's a number.

# Print out the question.
print_question(text, answer)

```

    ==== Question ====
    
        In ____, Christopher Columbus sailed across the ocean to discover the
        Americas
    


*And here's BERT's answer!*


```python
# Have BERT predict the answer.
(tokens, pred_answer) = predict_answer(text, answer)

# Print BERT's answer, and whether it got it right!
print_answer(answer, pred_answer, tokens)

```

    ==== BERT's Answer ====
    
        "1492"  -  CORRECT!


### BERT's Opinions

To try and fabricate BERT's opinions, I ran it with some opinionated statements. 

The token count problem is an issue here--the number of tokens in the answer might force BERT to pick a particular answer. 

To combat this, I ran each statement multiple times with several possible answers to see if the token count changed BERT's answer. BERT seemed to be pretty consistent in its choices, though :)


```python
# List of (question, answer) pairs.
pairs = [
    ("Microsoft has the best video game console.", "Microsoft"),
    ("Sony has the best video game console.", "Sony"),
    ("Nintendo has the best video game console.", "Nintendo"),    

    ("I prefer the Xbox One over the PS4.", "Xbox One"),
    ("I prefer the Xbox 360 over the PlayStation 3.", "Xbox 360"),

    ("James Cameron has made many great films, but his best is Terminator 2.", "Terminator 2"),
    ("James Cameron has made many great films, but his best is Avatar.", "Avatar"),
    ("James Cameron has made many great films, but his best is Titanic.", "Titanic"),

    ("I don't always drink beer, but when I do, I prefer Dos Equis.", "Dos Equis"),
    ("I don't always drink beer, but when I do, I prefer Stella Artois.", "Stella Artois"),

    ("Episode V is the best of the original Star Wars Trilogy.", 'V'),
    ("Episode IV is the best of the original Star Wars Trilogy.", 'IV'),
    ("Episode VI is the best of the original Star Wars Trilogy.", 'VI'),

    ("The acronymn 'GIF', which stands for Graphics Interchange Format, should be pronounced “jif”, like the brand of peanut butter.", "jif"),
    
    ("Chris McCormick creaties helpful illustrations and clear explanations of difficult subjects in machine learning and natural language processing.", "machine learning"),
]

# For each question...
for p in pairs:
    text = p[0]
    answer = p[1]

    # Print out the question.
    print_question(text, answer)

    # Predict the answer.
    (tokens, pred_answer) = predict_answer(text, answer)

    # Print and score the answer.
    print_answer(answer, pred_answer, tokens)
```

    ==== Question ====
    
        _________ has the best video game console.
    
    ==== BERT's Answer ====
    
        "japan"  -  WRONG.
    
        Correct:    "Microsoft"
    
        Tokens:      ['japan'] 
    
    ==== Question ====
    
        ____ has the best video game console.
    
    ==== BERT's Answer ====
    
        "japan"  -  WRONG.
    
        Correct:    "Sony"
    
        Tokens:      ['japan'] 
    
    ==== Question ====
    
        ________ has the best video game console.
    
    ==== BERT's Answer ====
    
        "japan"  -  WRONG.
    
        Correct:    "Nintendo"
    
        Tokens:      ['japan'] 
    
    ==== Question ====
    
        I prefer the ________ over the PS4.
    
    ==== BERT's Answer ====
    
        "ipod4"  -  WRONG.
    
        Correct:    "Xbox One"
    
        Tokens:      ['ipod', '##4'] 
    
    ==== Question ====
    
        I prefer the ________ over the PlayStation 3.
    
    ==== BERT's Answer ====
    
        "xbox 2"  -  WRONG.
    
        Correct:    "Xbox 360"
    
        Tokens:      ['xbox', '2'] 
    
    ==== Question ====
    
        James Cameron has made many great films, but his best is ____________.
    
    ==== BERT's Answer ====
    
        "the of titanic"  -  WRONG.
    
        Correct:    "Terminator 2"
    
        Tokens:      ['the', 'of', 'titanic'] 
    
    ==== Question ====
    
        James Cameron has made many great films, but his best is ______.
    
    ==== BERT's Answer ====
    
        "titanic"  -  WRONG.
    
        Correct:    "Avatar"
    
        Tokens:      ['titanic'] 
    
    ==== Question ====
    
        James Cameron has made many great films, but his best is _______.
    
    ==== BERT's Answer ====
    
        "titanic"  -  CORRECT!
    ==== Question ====
    
        I don't always drink beer, but when I do, I prefer _________.
    
    ==== BERT's Answer ====
    
        "a and ofs"  -  WRONG.
    
        Correct:    "Dos Equis"
    
        Tokens:      ['a', 'and', 'of', '##s'] 
    
    ==== Question ====
    
        I don't always drink beer, but when I do, I prefer _____________.
    
    ==== BERT's Answer ====
    
        "a ands"  -  WRONG.
    
        Correct:    "Stella Artois"
    
        Tokens:      ['a', 'and', '##s'] 
    
    ==== Question ====
    
        Episode _ is the best of the original Star Wars Trilogy.
    
    ==== BERT's Answer ====
    
        "iii"  -  WRONG.
    
        Correct:    "V"
    
        Tokens:      ['iii'] 
    
    ==== Question ====
    
        Episode __ is the best of the original Star Wars Trilogy.
    
    ==== BERT's Answer ====
    
        "iii"  -  WRONG.
    
        Correct:    "IV"
    
        Tokens:      ['iii'] 
    
    ==== Question ====
    
        Episode __ is the best of the original Star Wars Trilogy.
    
    ==== BERT's Answer ====
    
        "iii"  -  WRONG.
    
        Correct:    "VI"
    
        Tokens:      ['iii'] 
    
    ==== Question ====
    
        The acronymn 'GIF', which stands for Graphics Interchange Format, should be
        pronounced “___”, like the brand of peanut butter.
    
    ==== BERT's Answer ====
    
        "gif"  -  WRONG.
    
        Correct:    "jif"
    
        Tokens:      ['gi', '##f'] 
    
    ==== Question ====
    
        Chris McCormick creaties helpful illustrations and clear explanations of
        difficult subjects in ________________ and natural language processing.
    
    ==== BERT's Answer ====
    
        "computer linguistics"  -  WRONG.
    
        Correct:    "machine learning"
    
        Tokens:      ['computer', 'linguistics'] 
    


# Part 3 - Appendix

## Trivia Question Sources


I had a hard time finding free trivia questions in an easily downloadable format. On top of that, almost all of the questions I've come across would require re-wording to put them in "fill-in-the-blank" format. 

Here are some interesting sources that I looked at, though, if you want to help expand the dataset!

**Reddit Post & Spreadsheets**

* I found this [reddit post](https://www.reddit.com/r/trivia/comments/3wzpvt/free_database_of_50000_trivia_questions/), complaining about the difficulty of finding free trivia questions.
* The author compiled a Google spreadsheet totalling 50k trivia questions [here](https://docs.google.com/spreadsheets/d/0Bzs-xvR-5hQ3SGdxNXpWVHFNWG8/edit#gid=878197345).
    * This spreadsheet includes questions from the shows *Who Wants to be a Millionaire?* and *Are You Smarter Than a Fifth Grader?*. 
    * It also includes a sheet named 'Trivia' which I think is a compilation of the other sources.

**Jeopardy**

* This site has an [archive](http://www.j-archive.com/showgame.php?game_id=3447) of all of the Jeoprady boards from the television show. The Jeopardy questions would require careful re-wording, and generally look to be very difficult!

**Quizlet**

* This site has free quiz questions, though not in the form that you could download easily. I took my initial examples from this set of fill-in-the-blank [flash cards](https://quizlet.com/295489702/world-history-fill-in-the-blank-flash-cards/) on world history.

**Wikipedia**

* Since BERT was trained on Wikipedia, taking text directly from Wikipedia seems like cheating, but maybe it's still valid to see how much knowledge BERT retained.
* I found out there's a keyboard shortcut on Wikipedia for walking to a random article... While on Wikipedia, press `Alt + Shift + X`. You'll end up with some pretty obscure trivia this way!

## BookCorpus


From the Appendix of the original [BERT paper](https://arxiv.org/pdf/1810.04805.pdf): 
> "BERT is trained on the BooksCorpus (800M words) and Wikipedia (2,500M
words)". 

With BERT coming from Google, I always just assumed that "BookCorpus" referred to training on Google's massive "Google Books" library (which you can browse from https://books.google.com).

Turns out that's completely wrong. **BookCorpus** (not BooksCorpus) comes from the following paper:

* *Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books* ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zhu_Aligning_Books_and_ICCV_2015_paper.pdf))
     * First Author: Yukun Zhu, University of Toronto
     * Published ~2015

Here's the description of the dataset in the paper (emphasis added):

> **BookCorpus**. In order to train our sentence similarity
model we collected a corpus of 11,038 books from the web.
These are **free books written by yet unpublished authors.**
We only included books that had more than 20K words
in order to filter out perhaps noisier shorter stories. The
dataset has books in 16 different genres, e.g., Romance
(2,865 books), Fantasy (1,479), Science fiction (786), etc.
Table 2 highlights the summary statistics of our corpus.

Table 2, re-created from the paper.

| Property                       | Value       |
|--------------------------------|-------------|
| # of books                     | 11,038      |
| # of sentences                 | 74,004,228  |
| # of words                     | 984,846,357 |
| # of unique words mean         | 1,316,420   |
| # of words per sentence median | 13          |
| # of words per sentence        | 11          |

There is a parallel paper by the same authors, *Skip-Thought Vectors* ([pdf](https://arxiv.org/pdf/1506.06726.pdf)). It contains a couple small extra details:
* They offer one more category: "Teen (430)"
* "Along with narratives, books contain dialogue, emotion and a wide range of interaction between characters".

The website for the BookCorpus project is [here](https://yknzhu.wixsite.com/mbweb), but they no longer host or distribute this dataset. 

Instead, they say that the text was gathered from this site: https://www.smashwords.com/, and suggest that you gather your own dataset from there. I found a GitHub repo for doing just that [here](https://github.com/soskek/bookcorpus)--not a lot of activity, though.

