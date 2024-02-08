---
layout: post
title:  "Existing Tools for Named Entity Recognition"
date:   2020-05-19 8:00:00 -0800
comments: true
image: https://lh3.googleusercontent.com/d/1SfkyBN53SAgfr3yjMLTeu_l1e6IgvJxs
tags: machine learning, deep learning, NER, named entity recognition, NLP, natural language processing, python, BERT, spacy, stanza, flair, pytorch, fine-tune, tutorial, example code, NLP library
---

In conjunction with our tutorial for fine-tuning BERT on Named Entity Recognition (NER) tasks [here](https://www.chrismccormick.ai/offers/sFfjji7i), we wanted to provide some practical guidance and resources for building your own NER application since fine-tuning BERT may not be the best solution for every NER application. 

In this post, we will:

1. Discuss when it might be appropriate to use an off-the-shelf library vs. training / fine-tuning your own model.
2. Point you to some popular libraries for performing NER tagging and share some quick-start examples.
3. Share some resources we've found comparing and benchmarking different NER tools.

You can also find the Colab Notebook version of this post [here](https://colab.research.google.com/drive/16mNNJRLs0FEHEA8PhQl4IzeK9GvJ27OJ).

*by Nick Ryan and Chris McCormick*

## Contents
 
* TOC
{:toc}

## Testing Existing Tools



Before developing and training your own NER model, it's worth your time to first consider the requirements of your project and try out some of the preexisting off-the-shelf NER models to see if they can do the job for you. Preexisting NER models have the advantage of being ready to test in a few lines of code and are in some cases designed around being fast and robust in a production setting. 

If your project requires you to *identify basic NER types like people, organizations, locations, etc.* then I encourage you to first test your project with the existing NER models from spaCy, Stanford, and Flair. 


The following code cell shows how to retrieve entity tags for some text using spaCy, which comes pre-installed on Colab.


```python
import spacy

# Download a spacy model for processing English
nlp = spacy.load("en_core_web_sm")

# Process a sentence using the spacy model
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Display the entities found by the model, and the type of each.
print('{:<12}  {:}\n'.format('Entity', 'Type'))

# For each entity found...
for ent in doc.ents:
    
    # Print the entity text `ent.text` and its label `ent.label_`.
    print('{:<12}  {:}'.format(ent.text, ent.label_))
```

    Entity        Type
    
    Apple         ORG
    U.K.          GPE
    $1 billion    MONEY


(We've also implemented this same simple example using Flair and Stanza in the [Appendix](https://colab.research.google.com/drive/16mNNJRLs0FEHEA8PhQl4IzeK9GvJ27OJ#scrollTo=E0a-G2IqjxEA&line=1&uniqifier=1)!)



spaCy currently supports 18 different entity types, listed [here](https://spacy.io/api/annotation#named-entities). In the above example, "ORG" is used for companies and institutions, and "GPE" (Geo-Political Entity) is used for countries.

NER is covered in the spaCy getting started guide [here](https://spacy.io/usage/linguistic-features#named-entities).


These three libraries and most other off-the-shelf NLP libraries have an interface for you to train your own NER model using your dataset and their predetermined model architecture if you wish. (spaCy's documentation includes an example of this [here](https://spacy.io/usage/training#ner)).



## When to Fine-Tune



In some cases, these off-the-shelf libraries won't be the best solution for your project. You might have:
- Specific entity types that are not included in the off-the-shelf versions
- A different kind of text corpus from what the off-the-shelf models are trained on
- Very high accuracy or recall requirements 

In general, fine-tuning BERT (or variants of BERT) on your dataset will yield a *highly accurate* tagger, and with *less training data* required than training a custom model from scratch. 

The biggest caveat, however, is that BERT models are large, and typically warrant GPU acceleration. Working with GPUs can be expensive, and BERT will be slower to run on text than tools like spaCy.

So consider your production requirements for speed, accuracy, and cost before going straight to BERT!


## Resources & Experiments


There is, of course, a lot more that can be said about these different NLP toolkits. Our goal for this post was to simply make sure that you were aware of them, and the reasons you might use them over BERT.

The following are some articles which we found informative, containing experiments, summaries, benchmarks, and other comparisons of different NER tools.  They're worth looking through if you'd like to get a sense of NER pipelines and the power of existing NER tools. Below each article, we've highlighted the main points.

https://towardsdatascience.com/benchmark-ner-algorithm-d4ab01b2d4c3
- French legal dataset, focus on multilingual performance and irregular text domain
- Compares flair, camenBERT, mBERT, and Spacy 
- Authors conclude Flair is best for their application

https://primer.ai/blog/a-new-state-of-the-art-for-named-entity-recognition/
- NLP company Primer creates their own NER to benchmark on the CONLL2003 dataset
- BERT provides a large performance jump on Spacy
- XLNet provides a small performance jump on BERT
- The biggest key improvement comes from adding a diverse range of documents to the training aimed at improving performance on CONLL2003 specifically

http://nlpprogress.com/english/named_entity_recognition.html
- A mostly up-to-date collection of top models on a few of the most popular NER datasets for benchmarking (including CONLL2003).
- Compares research algorithms rather than tools like Spacy, NLTK, etc.

https://medium.com/@b.terryjack/nlp-pretrained-named-entity-recognition-7caa5cd28d7b#:~:text=
- Contains a great google colab notebook that provides code for quickly downloading and implementing ~10 different off-the-shelf NER models including Spacy, Stanford, NLTK, Flair, etc: https://colab.research.google.com/github/mohammedterry/NLP_for_ML/blob/master/NER.ipynb
- Some benchmarks on speed and accuracy, but limited to a very small (one sentence) test.

https://medium.com/@sapphireduffy/is-flair-a-suitable-alternative-to-spacy-6f55192bfb01
- Another comparison of different NER tools focused mainly on Flair vs. Spacy. 
- Both have strengths: Spacy is well documented but not as accurate, Flair is more accurate but not too well documented and not necessarily built for performance yet



Some NER datasets for testing/benchmarking tools:
- https://lionbridge.ai/datasets/15-free-datasets-and-corpora-for-named-entity-recognition-ner/
- https://github.com/juand-r/entity-recognition-datasets

## Appendix

Below are quick examples of performing NER using two other popular libraries (besides spaCy).

### Flair


[Flair](https://github.com/flairNLP/flair) is an open-source library developed by Humboldt University of Berlin.

See [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md) for a list of different pre-trained NER models available from flair, and [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md) is a tutorial on training your own flair model.

We'll first need to install the library from GitHub.


```python
!pip install --upgrade git+https://github.com/flairNLP/flair.git
```

The next cell will identify the entities in our example sentence. 

Note that Flair will need to download the `ner-ontonotes` model to run this cell, and this model appears to be around 1.5GB. If you're running this here in Colab, though, then it's not using your own bandwidth or disk space.


```python
from flair.data import Sentence
from flair.models import SequenceTagger

# Make a sentence
sentence = Sentence("Apple is looking at buying U.K. startup for $1 billion")

# Load the NER tagger
# This file is around 1.5 GB so will take a little while to load.
tagger = SequenceTagger.load('ner-ontonotes')

# Run NER over sentence
tagger.predict(sentence)

# Retrieve the entities found by the tagger.
entity_dict = sentence.to_dict(tag_type='ner')

# Display the entities, and the type(s) of each.
print('\n{:<12}  {:}\n'.format('Entity', 'Type(s)'))

# For each entity...
for entity in entity_dict['entities']:
    
    # Print the entity text and its labels. Flair supports multiple labels
    # per entity, and includes a confidence score.
    print('{:<12}  {:}'.format(entity["text"], str(entity["labels"])))

```

    2020-05-19 15:57:16,085 loading file /root/.flair/models/en-ner-ontonotes-v0.4.pt


    /pytorch/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program.


    
    Entity        Type(s)
    
    Apple         [ORG (0.9999)]
    U.K.          [GPE (0.9996)]
    $1 billion    [MONEY (0.9829)]


### Stanza (from Stanford)


Stanza [about page](https://stanfordnlp.github.io/stanza/index.html#about).

For information on how to train your own NER model, see Stanza's documentation [here](https://stanfordnlp.github.io/stanza/training.html).


```python
!pip install stanza
```


```python
import stanza

# This downloads the English models for the neural pipeline
stanza.download('en')     

# This sets up a default neural pipeline in English
nlp = stanza.Pipeline('en') 

# Process a sentence.
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Display the text and type of entities the model found
print('\n{:<12}  {:}\n'.format('Entity', 'Type'))

# For each entity...
for entity in doc.entities:

    # Print the text and its type.
    print('{:<12}  {:}'.format(entity.text, entity.type))
```

    Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/master/resources_1.0.0.json: 116kB [00:00, 9.85MB/s]                    
    2020-05-19 15:57:25 INFO: Downloading default packages for language: en (English)...
    2020-05-19 15:57:26 INFO: File exists: /root/stanza_resources/en/default.zip.
    2020-05-19 15:57:31 INFO: Finished downloading models and saved to /root/stanza_resources.
    2020-05-19 15:57:31 INFO: Loading these models for language: en (English):
    =========================
    | Processor | Package   |
    -------------------------
    | tokenize  | ewt       |
    | pos       | ewt       |
    | lemma     | ewt       |
    | depparse  | ewt       |
    | ner       | ontonotes |
    =========================
    
    2020-05-19 15:57:31 INFO: Use device: cpu
    2020-05-19 15:57:31 INFO: Loading: tokenize
    2020-05-19 15:57:31 INFO: Loading: pos
    2020-05-19 15:57:32 INFO: Loading: lemma
    2020-05-19 15:57:32 INFO: Loading: depparse
    2020-05-19 15:57:33 INFO: Loading: ner
    2020-05-19 15:57:34 INFO: Done loading processors!


    
    Entity        Type
    
    Apple         ORG
    U.K.          GPE
    $1 billion    MONEY

