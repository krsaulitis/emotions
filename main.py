import numpy as np

import data_setup
import engine
import helpers
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import classification_report


def calculate_data_stats():
    df = pd.read_csv('data/en/raw/train.tsv', delimiter='\t', names=['text', 'labels', 'id'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df['count'] = df['text'].apply(lambda text: len(tokenizer.tokenize(text)))

    mean_count = df['count'].mean()
    median_count = df['count'].median()
    min_count = df['count'].min()
    max_count = df['count'].max()

    print(f'Mean count: {mean_count}')
    print(f'Median count: {median_count}')
    print(f'Minimum count: {min_count}')
    print(f'Maximum count: {max_count}')

    plt.hist(df['text_length'], bins=1000)
    plt.title('Histogram of token count')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.show()


def validate_model():
    device = "mps"

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    test_data = data_setup.GoEmotionsDataset('data/en/raw/test.tsv', 64, 'data/en/raw/classes', tokenizer)
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=28)
    model = helpers.load_model(model, 'models', 'emotion-bert-cased-en-warm-wildflower-63-e5')
    model.to(device)

    test_metrics, test_labels, test_preds = engine.test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=BCEWithLogitsLoss(),
        device=device,
    )

    report = classification_report(test_labels, test_preds, target_names=test_data.classes(), zero_division=0)
    print(np.mean(test_metrics['loss']))
    print(report)


validate_model()
