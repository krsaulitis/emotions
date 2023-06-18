from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Model to evaluation mode
model.eval()

# The sentence you want to use for prediction
# The word 'MASK' is the placeholder for the word to predict
sentence = 'I [MASK] to play [MASK] while it\'s sunny outside.'

# Tokenize the sentence and convert tokens to input IDs
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Create attention mask
attention_mask = torch.ones(input_ids.shape)

# Predict
with torch.no_grad():
    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits

predicted_tokens = tokenizer.decode(logits[0].argmax(1))

print(f'Full sentence is: {predicted_tokens}')
