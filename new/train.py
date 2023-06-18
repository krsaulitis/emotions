import torch.optim

import data_setup
import engine
import helpers
import argparse
from config import Config
from logger import MetricsLogger
from torch.nn import BCEWithLogitsLoss
from transformers import logging, get_linear_schedule_with_warmup, BertForSequenceClassification

FORCE_CPU = True
USE_LOGGING = False


def _init_model(config: Config) -> (torch.nn.Module, int):
    logging.set_verbosity_error()
    model = BertForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=28,
        classifier_dropout=config.classifier_dropout,
    )

    model_path = f"models/${config.run_name}"
    loaded_model, epoch = helpers.load_model_w_epoch(model, model_path)
    if loaded_model:
        model = loaded_model

    model = model.to(helpers.get_device(FORCE_CPU))
    model.zero_grad()
    return model, epoch


def _init_optim(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    _optimizer = getattr(torch.optim, config.optimizer)
    return _optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def _init_scheduler(optimizer: torch.optim.Optimizer, batch_count: int, config: Config) -> torch.optim.lr_scheduler:
    total_steps = config.epochs * batch_count
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup),
        num_training_steps=total_steps,
    )
    scheduler_path = f"models/${config.run_name}"
    loaded_scheduler = helpers.load_scheduler(scheduler, scheduler_path)
    if loaded_scheduler:
        scheduler = loaded_scheduler

    return scheduler


def train(config: Config):
    logger = None

    if USE_LOGGING:
        logger = MetricsLogger(config)

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_path=f"{config.data_dir}/test_sample.tsv",
        test_path=f"{config.data_dir}/test_sample.tsv",
        class_path=f"{config.data_dir}/classes.txt",
        batch_size=config.batch_size,
        max_len=config.max_len,
    )

    model, epoch = _init_model(config)
    optimizer = _init_optim(model, config)
    scheduler = _init_scheduler(optimizer, len(train_dataloader), config)

    engine.train(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_function=BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=helpers.get_device(FORCE_CPU),
        logger=logger,
        current_epoch=epoch,
    )

    logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-pn', '--project-name', default='emotion-bert', type=str)
    parser.add_argument('-mn', '--model-name', default='emotion-bert-en', type=str)
    parser.add_argument('-rn', '--run-name', default=None, type=str)
    parser.add_argument('-rid', '--run-id', default=None, type=str)

    parser.add_argument('-d', '--data-dir', default='data/en')

    parser.add_argument('-bm', '--base-model', default='bert-base-cased', type=str)
    parser.add_argument('-o', '--optimizer', default='AdamW', type=str)
    parser.add_argument('-wd', '--weight-decay', default=None, type=str)
    parser.add_argument('-w', '--warmup', default=None, type=str)

    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument('-bs', '--batch-size', default=32, type=int)
    parser.add_argument('-lr', '--learning-rate', default=1e-2, type=float)
    parser.add_argument('-s', '--seed', default=42, type=int)

    train(Config(**vars(parser.parse_args())))
