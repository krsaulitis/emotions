import torch
import engine
import data_setup
import helpers
import wandb
import argparse
from torch.nn import BCEWithLogitsLoss
# from model import EmotionBert
from transformers import logging, get_linear_schedule_with_warmup, BertForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-dir', default='data/en')
parser.add_argument('-md', '--model-dir', default='models')
parser.add_argument('-e', '--epochs', default=5, type=int)
parser.add_argument('-bs', '--batch-size', default=32, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float)
parser.add_argument('-rn', '--run-name', default=None, type=str)
parser.add_argument('-pn', '--project-name', default='emotion-bert', type=str)
parser.add_argument('-mn', '--model-name', default='emotion-bert-en', type=str)
parser.add_argument('-me', '--model-epoch', default=None, type=int)
arguments = parser.parse_args()

DATA_DIR = arguments.data_dir
MODEL_DIR = arguments.model_dir
EPOCHS = arguments.epochs
BATCH_SIZE = arguments.batch_size
LEARNING_RATE = arguments.learning_rate
WEIGHT_DECAY = 0.05
RUN_NAME = arguments.run_name
PROJECT_NAME = arguments.project_name
MODEL_NAME = arguments.model_name
MODEL_EPOCH = arguments.model_epoch
CLASSIFIER_DROPOUT = 0.1
NUM_CLASSES = 28
MAX_LEN = 64
SEED = 42
FORCE_CPU = True
START_WANDB = False
LOCK_BASE = False
USE_SCHEDULER = True


model_file_exists = helpers.check_file_in_dir("models", f"{MODEL_NAME}-{RUN_NAME}")
model_exists = model_file_exists if RUN_NAME is not None else False

if START_WANDB:
    wandb.init(
        # id="hfp4uqk5",
        project=PROJECT_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "num_classes": NUM_CLASSES,
            "use_scheduler": USE_SCHEDULER,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "model": f"{MODEL_NAME}",
            "optimizer": "AdamW",
            "seed": SEED,
        },
        name=RUN_NAME,
        resume=model_exists
    )

    RUN_NAME = wandb.run.name

print(f"Project Name {PROJECT_NAME} | Run Name: {RUN_NAME} | Model Name: {MODEL_NAME} \n"
      f"Data dir: {DATA_DIR} | Model dir: {MODEL_DIR} \n"
      f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE} | Weight decay: {WEIGHT_DECAY}")

device = "cpu"
if not FORCE_CPU and torch.cuda.is_available():
    device = "cuda"
elif not FORCE_CPU and torch.backends.mps.is_available():
    device = "mps"


print(f"Using {device} device")

dataloader_train, dataloader_test = data_setup.create_dataloaders(
    train_path='data/en/raw/train.tsv',
    test_path='data/en/raw/test.tsv',
    class_path='data/en/raw/classes',
    batch_size=BATCH_SIZE,
    max_len=MAX_LEN,
)

logging.set_verbosity_error()

model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased',
    num_labels=NUM_CLASSES,
    classifier_dropout=CLASSIFIER_DROPOUT,
)
# model = EmotionBert.from_pretrained('bert-base-uncased')

if LOCK_BASE:
    for param in model.bert.parameters():
        param.requires_grad = False

loss_function = BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

scheduler = None
if USE_SCHEDULER:
    batch_count = len(dataloader_train)
    total_steps = EPOCHS * batch_count
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

if model_exists:
    model_name = f"{MODEL_NAME}-{RUN_NAME}-e{MODEL_EPOCH}"
    model = helpers.load_model(model, MODEL_DIR, model_name)

    if USE_SCHEDULER:
        scheduler_name = f"{MODEL_NAME}-{RUN_NAME}-e{MODEL_EPOCH}-s"

        if helpers.check_file_in_dir('model', scheduler_name):
            scheduler = helpers.load_model(scheduler, MODEL_DIR, scheduler_name)


model = model.to(device)
model.zero_grad()

torch.manual_seed(SEED)

engine.train(
    model=model,
    train_dataloader=dataloader_train,
    test_dataloader=dataloader_test,
    loss_fn=loss_function,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=EPOCHS,
    device=device,
)

wandb.finish()
