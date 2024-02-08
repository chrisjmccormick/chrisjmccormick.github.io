---
layout: post
title:  "Summarizing Long PDFs with ChatGPT"
date:   2024-01-30 8:00:00 -0800
comments: true
image: /assets/chatgpt/summarizing_pdfs_thumbnail_with_dropshadow.png
tags: ChatGPT, Long Documents, Summarization, PDF, OCR
---

<a href="https://colab.research.google.com/github/chrisjmccormick/summarize-long-pdfs/blob/main/Summarizing_Long_PDFs_with_ChatGPT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ▂▂▂▂▂▂▂▂▂▂▂▂

# I. Introduction

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/chrisjmccormick/summarize-long-pdfs/)

A friend of mine was taking a college course in political science with a ton of assigned reading material, and found that ChatGPT could produce helpful summaries (and in case you're wondering, the summaries are intended as an additional learning aid, rather than a replacement for doing the reading 😜).

There were a few challenges to trying to use ChatGPT for this, though:

* The reading materials are in the form of PDFs, and there are just too many (39! 😳) to do this manually.
* Most of the readings are too long to fit into ChatGPT in a single pass.
* Some of the PDFs are scans (or even just photos!) of pages from books, and none of the text is selectable.
* Even for the PDFs which do have selectable text, copying and pasting it into ChatGPT isn't trivial.

So, I created this Notebook to automate the process and summarize all _39_ of the PDFs assigned for the class, and it sounds like they were really helpful!

This Notebook is intended both as a relatively polished tool for completing this task, and as a tutorial and example code for working on this "summarization" problem yourself. I'm sure you can improve on it by experimenting with various details of the process!

_Note: I think the biggest caveat to this Notebook as a practical tool is that it_ does _rely on OpenAI's interface, which means you'll need to do some setup work on OpenAI's website in order to fully run it. Sorry!_

## i. Text Sources

**Part 1** of this Notebook turns all of the PDFs into "**plain text**" .txt files. The `PyMuPDF` library has everything we need for this--it can extract text from the PDFs that have it, and can run "OCR" (optical character recognition) for PDFs that only contain text in the form of images. For text extraction, it uses the `tesseract` library.



**PDFs, eBooks, and Physical Books**

I think it's worth pointing out that you can use this same code to summarize portions of text from literally any source--including eBooks and paper books--you'll just have to create PDFs first.

For an **eBook**, you could paste **screenshots** (since eBook readers don't allow you to copy text) into a Google Doc and then save it as a PDF. I imagine text extraction should work great on such clean images.

For a **paper book**, you could use a scanner app on your phone (I often use the scan tool in the "Files" app on my iPhone).

If you already have the **plain text** you want summarized, then you can just place it in a `.txt` file with the suffix `* - Parsed.txt` and the summarization code will work on it.

For something on the **web**, you may just be able to copy and paste the text. But if that's tricky, then, hey, you can always save it as a PDF! 😊

(Note: I also included some starter code in the appendix for extracting text straight from the web page's HTML instead.)


## ii. Using ChatGPT

**Part 2** of the Notebook uses ChatGPT to **summarize** each of those .txt files.



**Garbled Text**

I think something great here about the way ChatGPT works is that, similar to you and I, it is pretty good at making sense of imperfect text. The text that comes out of the PDFs can be pretty messy, especially if the document contains tables and figures, and yet GPT still seems to perform great!



**Length Limit**

One of the challenges here is that ChatGPT can't consume a 25-page book chapter all at once. There's a limit to how much text you can give it.

You _can_ break that book chapter into, e.g., 4 separate chunks that are each just within the limit. There's still a problem, though... When you feed ChatGPT a big chunk of text like this, it actually has no memory of the previous chunks you've given it!

The way ChatGPT works is that every time it replies to you, it actually _re-reads your entire chat history_, plus your latest message, in order to respond to you. This creates the _illusion_ that it remembers what you've been talking about. In reality, once your conversation goes beyond that length limit, older messages get dropped, and it has _zero_ knowledge of them.

The implication of this is that your **entire chat history**, plus your **next prompt**, and even its own **reply**, all have to fit within the text-length limit.

So we'll have to get creative in how we work around this limitation!


**Summarizing Across Chunks**

The way I chose to address this was to give GPT the summary so far along with the next chunk to summarize. Then I give it all of the chunk summaries and ask it to create a single more concise summary.

You'll see the exact process and my wording (i.e., my "prompt") down in Part 2!

There's another challenge here around figuring out _where_ exactly to split the text in order to get chunks of the appropriate size. I did this using OpenAI's "tokenizer", and I'll get into the details of that as well.


**Using OpenAI**

It's possible to use what I've created here for free by going to chat.openai.com and copying and pasting things in and out of their chat interface...

To make this all _automatic_, though, this Notebook is set up to use **OpenAI's web service**. In order to run it, you'll need to register for an OpenAI account and grab your **"API key"** to plug into this Notebook further down.

It's pretty easy to do, and I assume they still provide some **free credits** when you start out. If you're using this _heavily_ you may eventually need to pay a little bit, but the free credits should take you a long way.

Kind of a bummer, I know. Sorry!


-------

Aside: *Aren't there smaller models now that you can run yourself for free with similar performance to GPT?*

Yes! These are exciting.

But... I think the catch there is that using your own GPU, or a free one on Colab, will only work for _relatively short_ inputs.

The problem is that the amount of memory and compute required by GPT grows _exponentially_ with the length of the text!

Because of this, I don't think you're going to be able to use the maximum 4,096 tokens that GPT can handle, and I'm worried that this won't work as well if you have to break the article into too small of chunks.

But I could be wrong on both counts, honestly! Let me know if you try it. 😊

--------






# ▂▂▂▂▂▂▂▂▂▂▂▂

# Part 1 - Extracting Text from PDFs

## 1.1. Install PDF Libraries

**Tesseract**

Tesseract is a library for extracting text from images. PyMuPDF requires it for this functionality.

Note that this is not a Python library (though I'm sure some Python wrappers exist out there), so we're installing it with `apt`.


```python
!sudo apt install tesseract-ocr
```


It should install to following folder:


```python
!ls /usr/share/tesseract-ocr/4.00/tessdata
```

    configs  eng.traineddata  osd.traineddata  pdf.ttf  tessconfigs


PyMuPDF needs this environment variable set in order to locate tesseract.


```python
import os

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
```

**PyMuPDF**

The `PyMuPDF` (on GitHub [here](https://github.com/pymupdf/PyMuPDF), docs [here](https://pymupdf.readthedocs.io/en/latest/)) library is actually written in C, and this is just a Python wrapper for it. The logo for the library is written with the "mu" character, "μ", so perhaps the name is actually "Micro PDF"? 🤷‍♂️


```python
# I'm specifying this version number because, as of today (8/31/23), version
# 1.23 is telling me that OCR isn't supported.
!pip install pymupdf==1.22.5
```


Note that when importing PyMuPDF, it seems the module is actually named `fitz`, for whatever reason.

This code from their GitHub documentation verifies the library version and sets an environment variable to indicate where Tesseract can be found.


```python
import fitz

print('PyMuPDF version', fitz.VersionBind)
if tuple(map(int, fitz.VersionBind.split("."))) < (1, 19, 1):
    raise ValueError("Need at least v1.19.1 of PyMuPDF")
```

    PyMuPDF version 1.22.5


Finally, define a little helper function for formatting the elapsed time. (The OCR process can be kinda slow!)


```python
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
```

## 1.2. Path to PDFs

I put all of the PDFs onto my Google Drive under a folder named `Readings`.
Make sure to update this path to point to your folder.

If you just want to try this out and don't have your own PDFs yet, here are links to the examples I'm using: [1](https://drive.google.com/file/d/1RZ2iNDaxSxgcpqdXhAY4huT5FRD1Rpbq/view?usp=sharing), [2](https://drive.google.com/file/d/11JSMd93t4TxNjxekEwOvlIH_-pncBsMW/view?usp=sharing), [3](https://drive.google.com/file/d/1GvviH1qq95Ajx9xPkz3kzm_bj8N7N9EA/view?usp=sharing)


```python
dir = "./drive/MyDrive/Readings/"
```

This cell just gets a list of all of the PDFs in the specified folder.


```python
import os

files = os.listdir(dir)

# We'll construct a list of just paths to the PDFs.
pdf_filenames = []

# For each file in the directory...
for f in files:

    # Filter for pdfs.
    if '.pdf' in f:

        # Add to the list
        pdf_filenames.append(f)

        print(f)
```

    America and China Cooperating on Climate.pdf
    Machiavelli - The Prince - Chp 19.pdf
    Krauss - Chp 3 - Decent OCR.pdf


## 1.3. Extract the Text!

Now we'll loop through the PDFs and parse each of them!



**Output Files**

After parsing a PDF, we'll write the plain text out to a `.txt` file. It will have the same filename as the PDF, but with the extension ` - Parsed.txt`.

For example, "America and China Cooperating on Climate - Parsed.txt"

Part 2 of this Notebook, which performs the summarization, runs on any "* - Parsed.txt" files it finds in the directory.



**Text vs. OCR**

I'm using an overly simple approach (which _usually_ worked fine) to determining whether we need to run OCR or not.

I use the library's `get_text()` function (which does _not_ perform OCR) on the PDF, and if it returns anything, then I assume this is a text PDF that doesn't require OCR.  

If `get_text` returns an empty string, though, then I run OCR on it.

> Note: I did come across one PDF which was image based, but where the `get_text` function did return a tiny bit of text--I think it was returning page titles and page numbers for each page.
> I'm not sure of a good general solution to this. I just tweaked my copy of the code to handle that file specially. 🤷‍♂️

The OCR step can be slow, so I also included a check to skip over files that've already been parsed (by checking for the existence of the `'* - Parsed.txt'` file).


```python
import os

# For each of the documents...
for (pdf_num, pdf_name) in enumerate(pdf_filenames):


    # Print out which one we're on.
    print('\n======== {:} ({:} of {:}) ========\n'.format(pdf_name, pdf_num + 1, len(pdf_filenames)) )

    # Construct the full path to the file.
    pdf_path = dir + pdf_name

    # Construct the file name for the output by adding the tage " - Parsed" to
    # the end of the filename and replacing the file extension '.pdf' with
    # '.txt'.
    text_file_path = pdf_path[0:-4] + " - Parsed.txt"

    # If the output .txt file already exists, then I'm assuming we already took
    # care of it, so skip this PDF.
    if os.path.exists(text_file_path):
        print('Skipping - Already Parsed.')
        continue

    # Track the time.
    t0 = time.time()

    # ======== Open ========

    # Open the PDF file as a fitz.Document
    doc = fitz.open(pdf_path)


    # ======== Try Text ========
    all_text = ""

    # For each page in the document...
    for i, page in enumerate(doc):

        # Retrieve whatever text exists.
        all_text += page.get_text()

    # If there wasn't any text...
    if len(all_text) == 0:

        # ======== Run OCR =========
        # We'll try parsing the images.

        print('  Running OCR...')

        # For each page...
        for i, page in enumerate(doc):

            # This can be slow, so we'll print out the current page number to
            # show our progress.
            print('    Page {:} of {:}'.format(i + 1, len(doc)))

            # Make the `TextPage` object using the `get_textpage_ocr` function.
            # This is the step that does all of the OCR.
            full_tp = page.get_textpage_ocr(flags=0, dpi=300, full=True)

            # Next, we'll pass this `full_tp` to the `get_text` function to
            # pull out the text.
            #
            # By passing the "blocks" argument, we'll get a list of text blocks
            # from the page. I'm guessing this means the library first
            # identifies regions of text in the image, and then runs OCR on
            # them separately.
            #
            # Some example code from the library's GitHub repo showed this
            # approach, where we are eliminating *all* newline characters from
            # the text. I followed this example because, wihtout it, there does
            # seem to be a ton of whitespace! ChatGPT won't care, but if you
            # want to read the parsed file yourself, it helps to clean that up.

            # Get the parsed text blocks.
            blocks = page.get_text("blocks", textpage=full_tp)

            # For each block...
            for b in blocks:
                # The first four elements of 'b' appear to be something like
                # coordinates, and the fifth element, b[4], is the actual text.
                #
                # Replace *all* of the newline characters with a single space,
                # but then add back in a single newline character at the end
                # to separate the blocks.
                all_text += b[4].replace("\n", " ") + '\n'

            # Alternatively, if you ommit the "blocks" argument, you'll just
            # get a single string with all of the text.
            #all_text += page.get_text(textpage=full_tp)

    # ======== Record to Disk ========
    print('  Writing out scanned text to:\n    ', text_file_path)

    # Write all of the text to the .txt file.
    with open(text_file_path, "w") as f:
        f.write(all_text)

    print('  Done.')

    print('  Elapsed:', format_time(time.time() - t0))


```

    
    ======== America and China Cooperating on Climate.pdf (1 of 3) ========
    
      Writing out scanned text to:
         ./drive/MyDrive/Readings/America and China Cooperating on Climate - Parsed.txt
      Done.
      Elapsed: 0:00:01
    
    ======== Machiavelli - The Prince - Chp 19.pdf (2 of 3) ========
    
      Writing out scanned text to:
         ./drive/MyDrive/Readings/Machiavelli - The Prince - Chp 19 - Parsed.txt
      Done.
      Elapsed: 0:00:01
    
    ======== Krauss - Chp 3 - Decent OCR.pdf (3 of 3) ========
    
      Running OCR...
        Page 1 of 8
        Page 2 of 8
        Page 3 of 8
        Page 4 of 8
        Page 5 of 8
        Page 6 of 8
        Page 7 of 8
        Page 8 of 8
      Writing out scanned text to:
         ./drive/MyDrive/Readings/Krauss - Chp 3 - Decent OCR - Parsed.txt
      Done.
      Elapsed: 0:01:58


# ▂▂▂▂▂▂▂▂▂▂▂▂

# Part 2 - Summarizing

Now we can finally summarize the articles / chapters!

## 2.1. The Problem with Long Text

**Summarizing Long Passages**

A significant limitation with how ChatGPT is implemented is that it can only accept up to a certain amount of text.

If our article is short enough, then asking ChatGPT to summarize it is as simple as something like "Please summarize the following article: ".

But the articles and book chapters assigned in my friend's college class are (usually) too long to be summarized in one go.

Instead, we need a strategy for breaking the article into smaller chunks...

**ChatGPT's Conversation "Memory"**

When you go to chat.openai.com and have a conversation with ChatGPT, you can chat back and forth with it endlessly, and it remembers what's already been said so far in the conversation.

It's a little deceptive, though... Once your chat gets really long, it actually starts to _completely_ forget anything beyond the past few thousand words of chat history.

ChatGPT can process a maximum of 4,096 "tokens" at a time (you could think of this as roughly 3,000 words, or a four page article).

Note: This limit may have increased since the time of writing.

But this isn't the limit on just the size of the next message you send... The **combined length** of **everything** needs to be short enough. That is, all of the following together must be under this 4,096 token limit:

1. _The whole chat history_
2. Your next message
3. ChatGPT's next reply

This is because ChatGPT _doesn't_ actually have a memory of your conversation--it just **re-reads** the **whole conversation** every time it replies!

Once you go past the 4,096 token limit, OpenAI starts to **drop** the **oldest parts** of your conversation (behind the scenes) in order to make room for new dialogue.


**Aside: Why is it limited?**

One of the biggest limitations with how ChatGPT is implemented is that the amount of GPU power required to run ChatGPT grows **exponentially** with how long your text is.

This leads to a practical problem around how much **time** and **money** it costs to **train** ChatGPT. OpenAI had a big budget to work with, and of course wanted this token limit to be as large as possible, but they still had to pick a cut-off point somewhere.

At 4,096 tokens, ChatGPT cost them over a **million dollars** worth of GPU compute power to train. When you take into account that training these models required a lot of **trial-and-error** experimentation by the researchers (i.e., they trained it **many times** in different ways to try and improve its performance!), I imagine it was more like 10s or even 100s of millions of dollars before they arrived at the final version. Bigger models also require the researchers to **wait longer** before they get to see the results of their experiment.

Finally, **4,096** seems a little odd... Why not just 4,000? Us humans use the decimal system and tend to like numbers that are powers of 10 (10, 100, 1000, ...). But computers use the "binary" system, and so they like numbers that are powers of 2. (2, 4, 8, 16, 32, ...). The number 4,096 is 2 raised to the 12th power.


**Working Around the Limit**

To give an example, one of the reading materials was a 25-page book chapter which needed to be split into 4 chunks in order for each chunk to fit in this limit.

But each time I feed it a new big chunk, it has _no memory_ of the previous chunk(s). So that's where we have to get a little creative!



## 2.2. My Approach

**"Prompt Design"**

"Prompt Design" or "Prompt Engineering" refers to figuring out how to phrase and format your request to ChatGPT that will give you the best result.

I tried a few different approaches to this summarization task, and also tried optimizing my approach by making tweaks to the phrasing and layout of the prompt.  

What I landed on seems to work well, but I bet there's still room for improvement if you want to experiment more!

**My Prompt**

The trick I'm using to solve this length problem is to provide ChatGPT with a **summary of the previous chunks** of text to give it context as it works on summarizing the current chunk.

For each new chunk, I actually break it into two steps:



**Step 1:** Provide a summary of the past chunks, plus the new chunk. Ask ChatGPT to create a summary of the new chunk.

Here's the actual prompt. I'm using curly brackets to denote the variables.



>>
I am working on creating a summary of a long article or book chapter
which I have broken into **{total chunks}** segments.
>>
For context, here is a summary of the first **{# of chunks so far}** segments:
>>
**{Summary of all chunks so far}**
>>
Please write a bullet point summary of this next segment in 250 words
or less:
>>
**{next chunk}**



**Step 2:** Update the summary by providing it with all of the separate chunk summaries, and asking it to produce a single, more concise version.

>>
I am working on creating a summary of a long article or book chapter which I
have broken into **{total chunks}** segments.
>>
Please consolidate the following summaries of the first **{# of chunks so far}** segments down into a single bullet point summary that is 250 words or less:
>>
Summary of Segment 1:
>>
**{chunk 1 summary}**
>>
Summary of Segment 2:
>>
**{chunk 2 summary}**
>>
...

The code in this Notebook will also create a .txt log file (one log file per article) with all of the actual messages and replies sent and received, so you can read through one of those as an example as well.

Now let's move on to the code!

## 2.3. Setup OpenAI

First we have a bit of setup work to do.

**OpenAI**

This library will allow us to interact with ChatGPT programmatically.


```python
!pip install openai
```


**API Key**

You'll need to register for an OpenAI account, and provide your API key here in order to use this library.

You could modify my code to use more of a copy-and-paste method to work with the free chat interface at chat.openai.com, but that would be pretty slow-going!

I used the `secrets` feature in Colab (see the key-shaped icon in the sidebar) to allow me to run this Notebook easily without running the risk of accidentally sharing my API key with you. 😜


```python
from google.colab import userdata

api_key = userdata.get('Chris_API_key')
```

Pass your API key to the `openai` library.


```python
import openai

# Set up the OpenAI API client
openai.api_key = api_key
```

**Example Interaction**

This cell demonstrates the most basic way to send a prompt to ChatGPT and retrieve its reply.

The code is based on an example in the documentation, [here](https://github.com/openai/openai-python?tab=readme-ov-file#module-level-client).

The interface is clearly much more feature-rich than just what I'm doing here, but I won't be getting into the details of the API.


```python
import openai

prompt = "Tell me a joke."

# Send our message to ChatGPT
completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": prompt,
        },
    ],
)

# Extract the reply text.
reply = completion.choices[0].message.content

# Print the reply
print(reply)

```

    Why don't scientists trust atoms?
    
    Because they make up everything!


## 2.4. Load Tokenizer

We need to break the article down into separate chunks that are less than 4,096 tokens, but how do we know where to make those breaks in the text?

Luckily, OpenAI provides the `tiktoken` library which makes this task pretty easy! It will break the article down for us into a list of tokens, and then we can break that list down into chunks.

First, install the library.


```python
%pip install --upgrade tiktoken
```

**Load the Tokenizer**

ChatGPT isn't the only GPT "model" that OpenAI has, and sometimes different models use a different "vocabulary" of tokens.

To tokenize our text, we need to load the tokenizer for ChatGPT, and we can use the function `encoding_for_model` to have it pick the correct one for us based on our model name. ChatGPT is actually built on version 3.5 of GPT, so that's what you'll see in the code below.

Finally, you'll see that we use the word 'encoder' rather than 'tokenizer'. This is a little technicality that you don't need to concern yourself with.

But if you're curious... Each token in the vocabulary is represented by an integer, and the `encode` function breaks a text string into a list of token integers (rather than a list of token strings).


```python
import tiktoken

# Use tiktoken.encoding_for_model() to automatically load the correct tokenizer
# for a given model name.
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
```

Here's a quick example:


```python
text = "tiktoken is great!"

# Turn text into tokens with encoding.encode(), which converts the string into a
# list of integers.
token_ids = encoder.encode(text)

print(text, "-->")
print(token_ids)

# It will print: [83, 1609, 5963, 374, 2294, 0]

# The length of the list tells us how many tokens are required to represent our
# text.
num_tokens = len(token_ids)

print("\nThe text string breaks into", num_tokens, "tokens.")
```

    tiktoken is great! -->
    [83, 1609, 5963, 374, 2294, 0]
    
    The text string breaks into 6 tokens.


Turning the full text into a list of tokens will allow us to break the list  down into separate segments.

But what we need to actually give ChatGPT is a block of text, not a list of numbers!

For this, we'll use the `decode` function of the tokenizer.


```python
# Convert back from token numbers into a string.
# Note that the 'decode' function is able to exactly reproduce the
# input string--nothing is lost from encoding and decoding.
str_chunk = encoder.decode(token_ids)

print("The reconstructed string:")
print(str_chunk)
```

    The reconstructed string:
    tiktoken is great!


## 2.5. Summarize!

This section contains the code to actually perform the summarization.


**Output Files**

For each article, it will produce two .txt files:

1. "`{article name} - Summary.txt`" - This contains the final summary of the article. I also added two other things that I found useful:
    1. I asked ChatGPT to generate a glossary of key terms (with definitions).
    2. I included the list of separate chunk summaries, which provides a longer, more detailed overview of the article.

2. "`{article name} - Chat Log.txt`" - This includes the full text of all messages sent and received. If ChatGPT produces a particularly bad summary for an article, you can read through this log to see if you can figure out why.


**Restarting or Resuming**

Note that the code will check to see if the "- Summary.txt" file already exists, and skip that article if it does. This way, if something goes wrong, you can run the code again and it won't re-do articles it's already finished.

If you _do_ want to re-run the code for a particular article (or all of them), just delete the "- Summary.txt" and "- Chat Log.txt" files for the ones you want to re-run.


**Chunk Size**

We can't just break the article into 4,096 token chunks--we need to leave room for the prompt, summary of past chunks, and for ChatGPT's reply.

There's no way to predict how much room we'll need for those, but breaking the article down into 3,000 token chunks seems to be a safe bet. You can try adjusting this number if you want.



```python
# How many tokens in each chunk.
chunk_size = 3000
```

**List of Articles**

Get the list of all of the parsed articles that we are going to summarize.


```python
import os

txt_filenames = []

#dir = "./drive/MyDrive/Readings/"

files = os.listdir(dir)

for f in files:

    # Filter for pdfs.
    if ' - Parsed.txt' in f:

        # Add to the list
        txt_filenames.append(f)

        print(f)

```

    America and China Cooperating on Climate - Parsed.txt
    Machiavelli - The Prince - Chp 19 - Parsed.txt
    Krauss - Chp 3 - Decent OCR - Parsed.txt


**Summarization Loop**

Let's goooo!


```python
import os
import numpy as np
import textwrap

# This tiktoken 'Encoding' object can tokenize and de-tokenize our text.
encoding = tiktoken.get_encoding("cl100k_base")

# When we print out the final summary, we'll wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80)

# For each of the documents...
for (txt_num, txt_name) in enumerate(txt_filenames):

    # We already filtered for this, but just to sanity check, ensure we're only
    # running this on the "Parsed" files, and not the summaries. :-P
    assert(" - Parsed.txt" in txt_name)

    # Get the base filename.
    base_name = txt_name[0:-len(" - Parsed.txt")]

    # We'll be writing out two more files--the final summary, and a log of
    # the actual chat. (Note: I ommit the article chunks)
    summary_name = base_name + " - Summary.txt"
    log_name = base_name + " - Chat Log.txt"

    # Print the name of the article we're about to work on.
    print('\n======== {:} ({:} of {:}) ========\n'.format(base_name, txt_num + 1, len(txt_filenames)) )

    # If the summary output already exists...
    if os.path.exists(dir + summary_name):
         print('  Skipping - Already Summarized.')
         continue

    # Track the time.
    t0 = time.time()

    # Read in the full text of the article.
    with open(dir + txt_name, 'r') as f:
        lines = f.readlines()

    # Combine all of the lines into a single string.
    text = "".join(lines)

    # ======== Tokenize ========

    # Using tiktoken, we can split our string into ChatGPT tokens. Each token
    # is represented here by its index in the vocabulary.
    tokens = encoding.encode(text)

    # Determine the number of chunks we'll need to break this into.
    chunk_count = float(len(tokens)) / float(chunk_size)

    # Round up.
    chunk_count = int(np.ceil(chunk_count))

    # Print out the full token count of the article, and how many chunks we'll
    # be breaking it into.
    print("  Token count: {:,}   ({:} chunks of {:,} tokens)".format(len(tokens), chunk_count, chunk_size))

    # ======== Split Chunks ========

    str_chunks = []

    # For each chunk of tokens...
    for i in range(0, len(tokens), chunk_size):

        # Get the next cunk of tokens.
        # (Note that Python is nice enough to automatically handle the end of
        # the array for us.)
        token_chunk = tokens[i:i + chunk_size]

        # Convert back from token numbers into a string.
        # Note that the 'decode' function is able to exactly reproduce the
        # input string--nothing is lost from encoding and decoding.
        str_chunk = encoding.decode(token_chunk)

        # If the chunks line up such that the last one is tiny, then skip it.
        # I figure too small of an input might do more harm than good to the
        # summary--but I haven't tested that theory!
        if len(str_chunk) < 1000:
            print('Dropping the last chunk -- only {:} characters'.format(len(str_chunk)))
            continue

        str_chunks.append(str_chunk)

    num_chunks = len(str_chunks)

    # ======== Summarize! ========
    with open(dir + log_name, 'w') as f:

        print('\nSummarizing chunks...')

        chunk_summaries = []
        agg_summary = ""
        acc_prompt = ""

        # For each of the chunks...
        for i in range(len(str_chunks)):

            print('\n  Chunk {:} of {:}'.format(i + 1, num_chunks))

            # ======== Step 1: Summarize Next Chunk ========
            # Provide a summary of the past chunks, plus the new chunk.
            # Ask ChatGPT to create a summary of the new chunk.

            # The first chunk is a special case since we don't have context to
            # provide yet.
            if i == 0:
                # State the problem and what we want.
                prompt = \
"I am working on creating a summary of an article which I have broken into {:} \
segments. Below is the first segment; please write a bullet point summary of \
it in 250 words or less:\n\n".format(num_chunks) + str_chunks[0]

            # For all subsequent chunks, we'll provide the summary plus the new
            # chunk.
            else:
                # State the problem.
                prompt = \
"I am working on creating a summary of a long article or book chapter which I \
have broken into {:} segments.\nFor context, here is a summary of the first \
{:} segments:\n\n".format(num_chunks, i)

                # Insert the summary of the prior chunks.
                prompt += agg_summary + "\n"

                # Ask it to summarize.
                prompt += \
"\nPlease write a bullet point summary of this next segment in 250 words \
or less:\n\n"
                # Add the text for the current chunk.
                prompt += str_chunks[i]

            # Send it to ChatGPT!
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            # Extract the reply text.
            reply = completion.choices[0].message.content

            # Update the list of summaries.
            chunk_summaries.append(reply)

            # Write the response to the chat log file.
            f.write("\n\n▂▂▂▂▂▂▂▂▂▂▂▂▂ ↓ Prompt (Chunk {:} of {:}) ↓ ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂\n".format(i + 1, num_chunks))
            f.write(prompt)
            f.write("\n▂▂▂▂▂▂▂▂▂▂▂▂▂ ↓ Reply (Chunk {:} of {:}) ↓ ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂\n".format(i + 1, num_chunks))
            f.write(reply)

            # Report token counts.
            num_prompt_tokens = len(encoder.encode(prompt))
            num_reply_tokens = len(encoder.encode(reply))
            total_tokens = num_prompt_tokens + num_reply_tokens

            print('    Token counts - Prompt: {:,}    Reply: {:,}    Total: {:,}  (max is 4,096)'.format(num_prompt_tokens, num_reply_tokens, total_tokens))

            # ======== Step 2: Summarize the Summaries ========
            # Ask ChatGPT to consolidate all of the existing chunk summaries
            # down into a single condensed version.

            # For the first chunk, we don't need to ask it to consolidate
            # anything.
            if i == 0:
                agg_summary = reply

            # For subsequent chunks, provide each of the separate summaries and
            # ask it to create a single, condensed version.
            else:
                # Explain the task, and what we want.
                prompt = \
"I am working on creating a summary of a long article or book chapter which I \
have broken into {:} segments.\n Please consolidate the following summaries of \
the first {:} segments down into a single bullet point summary that is 250 \
words or less:\n".format(num_chunks, i+1)

                # Add each of the separate summaries:
                for (s_i, s) in enumerate(chunk_summaries):
                    prompt += "\nSummary of Segment {:}:\n".format(s_i + 1)
                    prompt += s

                # Send it to ChatGPT!
                completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                # Extract the reply text.
                reply = completion.choices[0].message.content

                # The reply is our new consolidated summary of the prior chunks.
                agg_summary = reply

                # Write the response to the chat log file.
                f.write("\n\n▂▂▂▂▂▂▂▂▂▂▂▂▂ ↓ Prompt (Chunk {:} of {:}) ↓ ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂\n".format(i + 1, len(str_chunks)))
                f.write(prompt)
                f.write("\n▂▂▂▂▂▂▂▂▂▂▂▂▂ ↓ Reply (Chunk {:} of {:}) ↓ ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂\n".format(i + 1, len(str_chunks)))
                f.write(reply)

            # End of the chunk loop!

    # At this point, we have summaries of all of the segments, and a single,
    # final summary.

    # Print the final summary! This uses the wrapper tool (instantiated at the
    # top of this cell) to wrap the text to 80-characters wide, for readability.
    print('\nFinal, overall summary:\n')
    print('--------')
    print(wrapper.fill(agg_summary))
    print('--------')
    print('')

    # ======== Glossary ========
    # As a final step, let's ask for a glossary of key terms!

    # Explain the task and what we want.
    prompt = \
"The following is a summary of a chapter of a book. Can you create a glossary \
for the key terms that readers might not be familiar with?\n\n" + agg_summary

    # Send it to chat gpt!
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    # Extract the reply text.
    reply = completion.choices[0].message.content

    # The reply should be a glossary of key terms.
    glossary = reply

    # Save the final summary to a text file.
    with open(dir + summary_name, 'w') as f:

        f.write("This document contains:\n")
        f.write("  1. The overall summary of the document.\n")
        f.write("  2. A glossary of key terms.\n")
        f.write("  3. The separate summaries of each chunk of text that the doc was broken into.\n")
        f.write("\n")
        f.write("▂▂▂▂▂▂▂ Overall Summary ▂▂▂▂▂▂▂\n")
        f.write(agg_summary)

        f.write("\n\n")
        f.write("▂▂▂▂▂▂▂ Glossary of Key Terms ▂▂▂▂▂▂▂\n")
        f.write(glossary)

        # For each separate chunk summary...
        for (s_i, s) in enumerate(chunk_summaries):
            f.write("\n\n")
            f.write("▂▂▂▂▂▂▂ Summary of Chunk {:} of {:} ▂▂▂▂▂▂▂\n".format(s_i + 1, num_chunks))
            f.write(s)



```

    
    ======== America and China Cooperating on Climate (1 of 3) ========
    
      Token count: 3,602   (2 chunks of 3,000 tokens)
    
    Summarizing chunks...
    
      Chunk 1 of 2
        Token counts - Prompt: 3,040    Reply: 316    Total: 3,356  (max is 4,096)
    
      Chunk 2 of 2
        Token counts - Prompt: 972    Reply: 240    Total: 1,212  (max is 4,096)
    
    Final, overall summary:
    
    --------
    US and China should establish a shared climate finance platform to overcome the
    challenges of fragmented and inadequate assistance to low-income developing
    countries facing unsustainable debt burdens. Coordination on climate finance
    would allow for complementary projects in developing countries, spreading
    financial risk and achieving more impact with limited resources. This
    collaboration would also help build trust and understanding between the two
    countries in blending public and private finance. The platform can include
    additional partners to make it more politically acceptable and bring in more
    funding. By pioneering this new approach, the US and China have the opportunity
    to encourage global green development and renew their cooperation on climate
    change.
    --------
    
    
    ======== Machiavelli - The Prince - Chp 19 (2 of 3) ========
    
      Token count: 4,953   (2 chunks of 3,000 tokens)
    
    Summarizing chunks...
    
      Chunk 1 of 2
        Token counts - Prompt: 3,040    Reply: 284    Total: 3,324  (max is 4,096)
    
      Chunk 2 of 2
        Token counts - Prompt: 2,291    Reply: 267    Total: 2,558  (max is 4,096)
    
    Final, overall summary:
    
    --------
    A ruler must avoid behaving in ways that provoke hatred or contempt from the
    subjects, including seizing property and women. They should act with greatness,
    seriousness, and strength, and preside over disputes to prevent trickery. It is
    crucial for a ruler to be well-liked by the majority to prevent conspiracies,
    and they should also guard against internal and external threats. The example of
    the Bentivogli family in Bologna highlights the power of being well-liked.
    Emperors in Rome faced challenges with the army, and those who favored the army
    over the people often faced unfortunate ends. Marcus Aurelius, Pertinax, and
    Alexander were decent rulers but lost respect and were overthrown, while cruel
    rulers such as Commodus, Severus, Antoninus Caracalla, and Maximinus met bad
    ends. Severus became emperor by defeating his opponents and gained power by
    being feared and respected. His son Antoninus was well-liked initially but his
    cruelty led to downfall. Similarly, Commodus and Maximinus were despised for
    their behavior. Heliogabalus, Macrinus, and Julian were also quickly overthrown.
    Modern rulers do not face the same pressure to satisfy the army, but leadership
    qualities and behavior still determine their success or failure.
    --------
    
    
    ======== Krauss - Chp 3 - Decent OCR (3 of 3) ========
    
      Token count: 7,819   (3 chunks of 3,000 tokens)
    
    Summarizing chunks...
    
      Chunk 1 of 3
        Token counts - Prompt: 3,040    Reply: 247    Total: 3,287  (max is 4,096)
    
      Chunk 2 of 3
        Token counts - Prompt: 3,301    Reply: 251    Total: 3,552  (max is 4,096)
    
      Chunk 3 of 3
        Token counts - Prompt: 2,070    Reply: 168    Total: 2,238  (max is 4,096)
    
    Final, overall summary:
    
    --------
    Protectionists and governments collude to deceive the public about the benefits
    of free trade. Protectionism, such as restrictions on international trade, harms
    both other countries and the US. The export of low-skill jobs to Mexico can
    actually raise US national income by redistributing jobs. Protectionism does not
    create employment, but rather redistributes it within the economy. Government
    policies impede labor mobility and wage flexibility. Subsidies are preferable to
    import restrictions. Increasing US savings is necessary to reduce the trade
    deficit, not imposing import restrictions. The terms of trade argument for
    imposing tariffs is flawed. Trade agreements should focus on beneficial
    reallocations of labor and capital. The US-Mexico trade liberalization agreement
    is expected to provoke beneficial reallocations due to differences in labor
    costs.
    --------
    


# ▂▂▂▂▂▂▂▂▂▂▂▂

# Appendix

## A.1. PDF Splitter

When trying to summarize a really long document like a book, for example, it's probably more helpful to create summaries for each of the individual chapters.

The code in this section will help you split a single large PDF into smaller ones (e.g., one PDF per chapter).

Once the PDF is split, you can run this Notebook on the splits (rather than the original PDF) in order to get separate summaries.

**Inputs**

Provide the path to the document.


```python
dir = './drive/MyDrive/Readings/Split-PDFs/'

filename = 'Machiavelli - The Prince.pdf'

# Remove the '.pdf' extension to get the base name.
base_name = filename[0:-4]
```

Fill out the list with the first page of each split.


```python
# This divide the PDF into 10 separate PDFs. The first will contain pages 1-22,
# the second will contain pages 23-51, and so on.
#split_starts = [1, 23, 52, 73, 98, 123, 139, 163, 192, 205]

# Pg 127 is the start of chapter 18, pg 130 is chapter 19, pg 140 is chapter 20.
split_starts = [1, 127, 130, 140]

final_page = -1
```

**From Table of Contents**

This little block of code (which is disabled) is helpful for splitting books or documents that have a table of contents.

1. Copy the page numbers from the table of contents into the `book_pages` list.
2. Calculate the offset. Which PDF page is page 1 of the book? Subtract one from that page number to get the offset.

The code simply adds the offset to the page numbers for you.


```python
# Map, e.g., from a book's table of contents to page numbers within the PDF.
# In the below example, Chapter 1 of the book is on page 1 of the book, but page
# 23 of the PDF. Chapter is on pg. 30 of the book, but page 52 of the PDF. And
# so on.
if False:
    # Page numbers according to the book.
    book_pages = [1, 30, 51, 76, 101, 117, 141, 170, 183]

    # Page one of the book occurs on page 23 of the PDF, so the offset is 22.
    pdf_offset = 22

    # The first split will contain pages 1-22 of the PDF.
    pdf_splits = [1]

    # Add the PDF offset to each of the book page numbers.
    for bp in book_pages:
        pdf_pages.append(bp + pdf_offset)

    split_starts = pdf_pages

    print(pdf_pages)
```

    [1, 23, 52, 73, 98, 123, 139, 163, 192, 205]


**Create the Splits**

The resulting split files will be named 'Example Book - 1.pdf', 'Example Book - 2.pdf', and so on.


```python
import sys, fitz

with fitz.open(dir + filename) as src_doc:

    print('Splitting', dir + filename)

    # Set the ending page of the last split to be the last page of the PDF, if
    # not specified.
    if final_page == -1:
        final_page = len(src_doc)
        print('Final page =', final_page)

    # For each of the splits...
    for (split_i, start_page) in enumerate(split_starts):

        # For the purposes of printing progress and naming the files, number the
        # splits starting from 1.
        split_num = split_i + 1
        print('  Split', split_num)

        # Determine what should be the last page of the current split.

        # If it's the final split, use the 'final_page' variable.
        if split_num == len(split_starts):
            end_page = final_page

        # Otherwise, use one less than the start of the next split.
        else:
            end_page = split_starts[split_i + 1] - 1

        # Create a new PDF object for this split.
        split_doc = fitz.open()

        # Add the pages from the current split.
        split_doc.insert_pdf(
            src_doc, # The original doc
            from_page = start_page - 1, # 0-indexed, so subtract 1.
            to_page = end_page - 1,
        )

        # Write out the new PDF, adding in the split number.
        split_doc.save(dir + base_name + ' - {:}.pdf'.format(split_num))

        split_doc.close()
```

    Splitting ./drive/MyDrive/Readings/Split-PDFs/Machiavelli - The Prince.pdf
      Split 1
      Split 2
      Split 3
      Split 4


## A.2. PDF Rotator

One of the assigned readings was a scan of a book chapter, but the PDF pages showed the book rotated 90 degrees. In order to parse and summarize this PDF, we first needed to fix the page rotation using the below code.


```python
import fitz  # PyMuPDF

input_pdf_path = "./drive/MyDrive/Readings/Example Article.pdf"
output_pdf_path = "./drive/MyDrive/Readings/Example Article - Rotated.pdf"

# Open the PDF file
pdf_document = fitz.open(input_pdf_path)

# Iterate through each page and rotate it 90 degrees counter-clockwise
for page_num in range(pdf_document.page_count):
    page = pdf_document[page_num]
    page.set_rotation(-90)  # Rotate 90 degrees counter-clockwise

# Save the modified PDF to a new file
pdf_document.save(output_pdf_path)
pdf_document.close()

```

## A.3. Extract a Webpage

Below is some code for running this Notebook on HTML files instead of PDFs.

The "input" to Part 2 of the Notebook (which does the summarizing) is that it just runs on any `* - Parsed.txt` files it finds in the specified directory, so really as long as you can get your documents into that format, you're good to go!

The below code extracts all of the plain text from any `.html` files in the specified folder and writes out the corresponding ` - Parsed.txt` files. So you can run this section and then go run Part 2.

**Beautiful Soup - HTML Parsing**


```python
!pip install beautifulsoup4
```


```python
import os

dir = "./drive/MyDrive/Readings/"

files = os.listdir(dir)

# We'll construct a list of just paths to the HTML files.
html_filenames = []

# For each file in the directory...
for f in files:

    # Filter for HTML.
    if '.html' in f:

        # Add to the list
        html_filenames.append(f)

        print(f)
```


```python
from bs4 import BeautifulSoup

# For each of the documents...
for (html_num, html_name) in enumerate(html_filenames):

    # Print out which one we're on.
    print('\n======== {:} ({:} of {:}) ========\n'.format(html_name, html_num + 1, len(html_filenames)) )

    # Construct the full path to the file.
    html_path = dir + html_name

    # Construct the file name for the output by adding the tage " - Parsed" to
    # the end of the filename and replacing the file extension '.html' with
    # '.txt'.
    text_file_path = html_path[0:-5] + " - Parsed.txt"

    with open(dir + html_name, 'r') as f:
        # Read the HTML file.
        html_content = f.readlines()

        # Convert from a list to a single string.
        html_content = '\n'.join(html_content)

        # Set up the HTML parser.
        soup = BeautifulSoup(html_content, 'html.parser')

        # Estract the plain text.
        plain_text = soup.get_text()

        # Write all of the text to the .txt file.
        with open(text_file_path, "w") as f:
            f.write(all_text)


```


```python

# Example HTML content
html_content = """
<html>
    <head>
        <title>Sample HTML</title>
    </head>
    <body>
        <h1>Hello, World!</h1>
        <p>This is a sample HTML document.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>
    </body>
</html>
"""

```

    
    
    
    Sample HTML
    
    
    Hello, World!
    This is a sample HTML document.
    
    Item 1
    Item 2
    Item 3
    
    
    
    


# ▂▂▂▂▂▂▂▂▂▂▂▂

# End
