from itertools import islice

from google.cloud import translate_v2 as translate
from google.oauth2.service_account import Credentials
import csv
from pathlib import Path
from typing import List
import numpy as np

BATCH_SIZE = 100

filename = 'train.tsv'
in_path = Path('data/en/raw/')
out_path = Path('data/lv/raw/')

if not out_path.is_dir():
    out_path.mkdir()

in_file = open(in_path / filename, mode='r', encoding='utf-8')
out_file = open(out_path / filename, mode='w', encoding='utf-8')

tsv_reader = csv.reader(in_file, delimiter='\t')
tsv_writer = csv.writer(out_file, delimiter='\t')


def translate_text(texts: List[str]):
    credentials = Credentials.from_service_account_file('translation/university-386313-daeb20fdda69.json')

    client = translate.Client(credentials=credentials)

    result = client.translate(
        values=texts,
        target_language="lv",
        source_language="en",
    )

    return np.array([list(item.values()) for item in result])[:, 0]


epoch = 0
while True:
    batch = np.array(list(islice(tsv_reader, BATCH_SIZE)))

    if batch.size == 0:
        break

    if epoch > 500:  # to prevent me from going broke
        break

    batch_texts = batch[:, 0]
    translated_batch_texts = translate_text(batch_texts.tolist())

    for i, translated_text in enumerate(translated_batch_texts):
        tsv_writer.writerow([translated_text, batch[i, 1]])

    epoch += 1
