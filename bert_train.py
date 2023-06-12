import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from torch.nn import BCEWithLogitsLoss
from transformers import BertForPreTraining

EPOCHS = 10
LEARNING_RATE = 1e-5
NUM_CLASSES = 25

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class GoEmotionsDataset(Dataset):
    def __init__(self, filename, max_len):
        super().__init__()

        # Load data and labels
        self.sentences = []
        self.labels = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pieces = line.split('\t')
                self.sentences.append(pieces[0])
                self.labels.append([int(x) for x in pieces[1].split(",")])

        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Selecting the sentence and label at the specified index in the data frame
        sentence = self.sentences[index]
        labels = self.labels[index]

        # Preprocess the text to be suitable for BERT
        tokens = tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            tokens = tokens[:self.max_len - 1] + ['[SEP]']

        # Converting the sequence to ids with BERT Vocabulary
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)

        # Attention mask
        attn_mask = (tokens_ids_tensor != 0).long()

        # Converting the list of labels to Tensor
        labels = np.eye(num_classes)[labels]
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return tokens_ids_tensor, attn_mask, labels_tensor


# Let's assume we have a dataset with input IDs, attention masks, and labels
# input_ids, attention_masks, labels = ...

# Load the pre-trained model
model = BertForPreTraining.from_pretrained('bert-base-uncased')

# Add a new output layer for the classification task
model.classifier = torch.nn.Linear(model.config.hidden_size, NUM_CLASSES)

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Move the model to the device
model = model.to(device)

# Define the data loader
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for batch in dataloader:
        # Load the batch to the device
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch

        # Forward pass
        outputs = model(input_ids, input_mask)
        logits = outputs.logits

        # Compute the loss
        loss_function = BCEWithLogitsLoss()
        loss = loss_function(logits.view(-1, num_classes), labels.type_as(logits).view(-1, num_classes))

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()
