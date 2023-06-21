import torch
import helpers
from transformers import logging, BertTokenizer, BertForSequenceClassification, BertConfig

logging.set_verbosity_error()

device = "mps"
base = "bert-base-cased"
text = "Es tevi mīlu!"

tokenizer = BertTokenizer.from_pretrained('models/lv_bert')

if True:
    model_config = BertConfig.from_json_file("models/lv_bert/config.json")
else:
    model_config = BertConfig.from_pretrained(base)

model_config.num_labels = 28
model = BertForSequenceClassification.from_pretrained("models/lv_bert/model.bin", config=model_config)
model = helpers.load_state(model, 'models/exalted-snowflake-22', 'model-e3')
model.to(device)
model.eval()

with torch.inference_mode():
    inputs = tokenizer.encode_plus(text, padding="max_length", max_length=64, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    print(input_ids, attention_mask)

    preds = model(input_ids.to(device), attention_mask.to(device))

    preds = (torch.sigmoid(preds.logits) > 0.3)

    print(torch.argmax(preds).cpu().tolist())
