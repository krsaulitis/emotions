import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from config import Config


class GoEmotionsDataset(Dataset):
    def __init__(self, path: str, class_path: str, max_len: int, tokenizer):
        super().__init__()

        self.class_path = class_path
        self.max_len = max_len
        self.tokenizer = tokenizer

        self.sentences = []
        self.labels = []

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pieces = line.split('\t')
                self.sentences.append(pieces[0])

                # Create a one-hot encoding vector
                label_vec = [0] * len(self.classes())
                for idx in pieces[1].split(","):
                    label_vec[int(idx)] = 1
                self.labels.append(label_vec)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        labels = self.labels[index]

        # Not used as LV pre-trained model does not support this
        # sentence = sentence.replace("[NAME]", "[unused1]")
        # sentence = sentence.replace("[RELIGION]", "[unused2]")

        # Preprocess the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            tokens = tokens[:self.max_len - 1] + ['[SEP]']

        tokens_ids_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        attn_mask = (tokens_ids_tensor != 0).long()
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return tokens_ids_tensor, attn_mask, labels_tensor

    def classes(self):
        with open(self.class_path, 'r') as file:
            class_names = [line.strip() for line in file]

        return class_names


def create_dataloaders(
        train_path: str,
        test_path: str,
        class_path: str,
        config: Config,
        num_workers: int = 0,
):
    tokenizer = transformers.BertTokenizer.from_pretrained(config.base_model_vocab)
    # Not used as LV pre-trained model does not support this
    # tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})

    train_data = GoEmotionsDataset(train_path, class_path, config.max_len, tokenizer)
    test_data = GoEmotionsDataset(test_path, class_path, config.max_len, tokenizer)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
