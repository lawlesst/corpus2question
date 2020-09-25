"""
Move the tutorial notebook functions into a file
so that they can be imported by notebooks.
"""

import os
import sys
from typing import List, Iterable

import nltk
import torch
import pandas as pd
from tqdm.notebook import tqdm
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

print("Downloading and unzipping model.", file=sys.stderr)
os.system("wget -nc https://storage.googleapis.com/doctttttquery_git/t5-base.zip")
os.system("unzip -o t5-base.zip")

nltk.download('punkt')

# Define the target device. Use GPU if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Instantiate and load the QG model to the GPU.
qg_tokenizer = T5Tokenizer.from_pretrained('t5-base')
qg_config = T5Config.from_pretrained('t5-base')
qg_model = T5ForConditionalGeneration.from_pretrained('model.ckpt-1004000', from_tf=True, config=qg_config)

qg_model.to(device)


def preprocess(document: str, span=10, stride=5) -> List[str]:
    """
    Define your preprocessing function.

    This function should take the a corpus document and output a list of generation
    spans. This is required so we can match the expected sequence size of the
    generation model.
    """

    sentences = nltk.tokenize.sent_tokenize(document)
    chunks = [" ".join(sentences[i:i+span]) for i in range(0, len(sentences), stride)]

    return chunks


def generate_questions(text: str) -> List[str]:
    """
    Define your generation function.

    This function should take a text passage and generate a list of questions.
    With the current configuration it always generate one question per passage.

    You may copy this example to use the same configuration as the paper.
    You may also configure the generation parameters (such as using sampling and
    generating multiple questions) for other use cases.
    """

    # Append an end of sequence token </s> after the context.
    doc_text = f"{text} </s>"

    input_ids = qg_tokenizer.encode(doc_text, return_tensors='pt').to(device)
    outputs = qg_model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=False,
        n_beams=4,
    )

    return [qg_tokenizer.decode(output) for output in outputs]


def get_questions(corpus):
    return [
        [generate_questions(span) for span in preprocess(doc)]
        for doc in tqdm(corpus)
    ]