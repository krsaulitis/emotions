import torch
from torch import nn
from transformers import pipeline
from transformers.models.bert import BertConfig, BertTokenizer
from transformers.models.bert import BertForPreTraining as BertModel

config = BertConfig.from_json_file('models/lv_bert/bert_config.json')
model = BertModel.from_pretrained('models/lv_bert/pytorch_model.bin', config=config)
tokenizer = BertTokenizer.from_pretrained('models/lv_bert/vocab.txt')

# Model to evaluation mode
model.eval()

sentence = 'Es esmu [MASK] [MASK]'
inputs = tokenizer.encode(sentence, return_tensors='pt')
attention_mask = torch.ones(inputs.shape)

with torch.inference_mode():
    outputs = model(inputs, attention_mask=attention_mask)
    logits = outputs.logits

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits

predicted_tokens = tokenizer.decode(logits[0].argmax(1))

print(f'Full sentence is: {predicted_tokens}')
