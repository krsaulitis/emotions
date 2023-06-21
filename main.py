import torch
import data_setup
import engine
import helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import logging, BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix
from tqdm.auto import tqdm


def calculate_count_stats():
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


def calculate_label_stats():
    print('Test')


def load_model_tokenizer_data(base_model, model_dir, model_name, is_lv_model, device="mps"):
    logging.set_verbosity_error()
    tokenizer = BertTokenizer.from_pretrained(base_model)
    data = data_setup.GoEmotionsDataset(f"data/{'lv' if is_lv_model else 'en'}/test.tsv", 'data/classes.txt', 64, tokenizer)
    dataloader = DataLoader(
        dataset=data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if is_lv_model:
        model_config = BertConfig.from_json_file('models/lv_bert/config.json')
    else:
        model_config = BertConfig.from_pretrained(base_model)
    model_config.num_labels = 28

    model = BertForSequenceClassification.from_pretrained(
        f"{base_model}/model.bin" if is_lv_model else base_model,
        config=model_config,
    )
    model = helpers.load_state(model, model_dir, model_name)
    model.to(device)
    model.eval()

    return model, tokenizer, dataloader


def validate_model_samples():
    device = "cpu"
    base_model = 'bert-base-cased'
    model_dir = 'models/spring-shadow-22'
    model_name = 'model-e3'
    is_lv = False
    model, tokenizer, dataloader = load_model_tokenizer_data(base_model, model_dir, model_name, is_lv, device)

    loss_fn = BCEWithLogitsLoss()

    test_losses = []
    test_truths = []
    test_preds = []

    with torch.inference_mode():
        for x, x_mask, y in tqdm(dataloader):
            x, x_mask, y = x.to(device), x_mask.to(device), y.to(device)
            for i in range(x.shape[0]):
                test_pred = model(x[i:i+1], x_mask[i:i+1]).logits
                loss = loss_fn(test_pred, y[i:i+1])

                test_pred = (torch.sigmoid(test_pred) > 0.5).float()
                test_losses.append(loss.cpu().item())
                test_truths.append(y.cpu().numpy())
                test_preds.append(test_pred.cpu().numpy())

    top10_worst_idx = np.argsort(test_losses)[-10:]
    # top10_worst_examples = np.array(test_examples)[top10_worst_idx]
    top10_worst_truths = np.array(test_truths)[top10_worst_idx]
    print(top10_worst_truths)
    for truth in top10_worst_truths:
        print(tokenizer.decode(truth))


def validate_model():
    device = "mps"
    base_model = 'bert-base-cased'
    model_dir = 'models/spring-shadow-22'
    model_name = 'model-e3'
    is_lv = False
    model, tokenizer, dataloader = load_model_tokenizer_data(base_model, model_dir, model_name, is_lv, device)

    loss_fn = BCEWithLogitsLoss()

    test_losses = []
    test_truths = []
    test_preds = []

    with torch.inference_mode():
        for x, x_mask, y in tqdm(dataloader):
            x, x_mask, y = x.to(device), x_mask.to(device), y.to(device)

            test_pred = model(x, x_mask).logits
            loss = loss_fn(test_pred, y)

            test_pred = (torch.sigmoid(test_pred) > 0.5).float()
            test_losses.append(loss.cpu().item())
            test_truths.append(y.cpu().numpy())
            test_preds.append(test_pred.cpu().numpy())

    report = classification_report(
        np.concatenate(test_truths),
        np.concatenate(test_preds),
        target_names=dataloader.dataset.classes(),
        zero_division=0,
    )
    print(f"Loss: {np.mean(test_losses)}")
    print(f"Accuracy: {accuracy_score(np.concatenate(test_truths), np.concatenate(test_preds))}")
    print(report)


def build_onnx_model():
    device = "mps"
    base_model = 'bert-base-cased'
    model_dir = ''
    model_name = ''
    is_lv = False
    model, tokenizer, dataloader = load_model_tokenizer_data(base_model, model_dir, model_name, is_lv, device)

    inputs = tokenizer.encode_plus("Sveiki, kā jums iet?", padding='max_length', max_length=64, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    dummy_input = (input_ids, attention_mask)

    tokenizer.save_vocabulary(f"{model_dir}/vocab.txt")
    torch.onnx.export(
        model,
        dummy_input,
        f"{model_dir}/model.onnx",
        export_params=True,
        do_constant_folding=True,
        opset_version=14,
        input_names=['input', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'attention_mask': {0: 'batch_size'},
                      'output': {0: 'batch_size'}})


# validate_model()
# build_onnx_model()
validate_model_samples()
