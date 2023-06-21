import torch.optim
import data_setup
import engine
import helpers
import argparse
from config import Config
from logger import MetricsLogger
from torch.nn import BCEWithLogitsLoss
from transformers import logging, get_linear_schedule_with_warmup, BertForSequenceClassification, BertConfig
from torch.optim.lr_scheduler import LRScheduler
from typing import Union, Tuple

FORCE_CPU = False
USE_LOGGING = True


def _init_model(classes_count: int, config: Config) -> Tuple[torch.nn.Module, int]:
    logging.set_verbosity_error()

    if config.base_model_config:
        model_config = BertConfig.from_json_file(config.base_model_config)
    else:
        model_config = BertConfig.from_pretrained(config.base_model)

    model_config.num_labels = classes_count
    model_config.classifier_dropout = config.classifier_dropout

    model = BertForSequenceClassification.from_pretrained(config.base_model, config=model_config)

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


def _init_scheduler(optimizer: torch.optim.Optimizer, batch_count: int, config: Config) -> Union[LRScheduler, None]:
    if not config.warmup:
        return None

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
    torch.manual_seed(config.seed)
    logger = None

    if USE_LOGGING:
        logger = MetricsLogger(config)

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_path=f"{config.data_dir}/{config.data_lang}/train.tsv",
        test_path=f"{config.data_dir}/{config.data_lang}/test.tsv",
        class_path=f"{config.data_dir}/classes.txt",
        config=config,
    )

    model, epoch = _init_model(len(train_dataloader.dataset.classes()), config)
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

    parser.add_argument('-pn', '--project-name', default='emotion-bert', type=str)
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
