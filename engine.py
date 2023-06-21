import math
import torch
import torch.utils.data
import helpers
from tqdm.auto import tqdm
from typing import Optional, Union
from logger import MetricsLogger
from config import Config


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        logger: Optional[MetricsLogger] = None):
    model.train()

    for x, x_mask, y in tqdm(dataloader):
        x, x_mask, y = x.to(device), x_mask.to(device), y.to(device)

        y_pred = model(x, x_mask).logits
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        y_pred = (torch.sigmoid(y_pred) > 0.5).float()

        logger.step_results(
            'train',
            loss.detach().cpu().item(),
            y.cpu().numpy(),
            y_pred.cpu().numpy(),
        )


def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        logger: Optional[MetricsLogger] = None):
    model.eval()

    with torch.inference_mode():
        for x, x_mask, y in tqdm(dataloader):
            x, x_mask, y = x.to(device), x_mask.to(device), y.to(device)

            test_pred = model(x, x_mask).logits
            loss = loss_fn(test_pred, y)

            test_pred = (torch.sigmoid(test_pred) > 0.5).float()

            logger.step_results(
                'test',
                loss.detach().cpu().item(),
                y.cpu().numpy(),
                test_pred.cpu().numpy(),
            )


def train(
        model: torch.nn.Module,
        config: Config,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        logger: Union[MetricsLogger] = None,
        current_epoch: int = 0,
):
    best_test_loss = math.inf

    for epoch in range(current_epoch, current_epoch + config.epochs):
        train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
        )

        test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_function,
            device=device,
            logger=logger,
        )

        test_loss = logger.get_loss('test')
        logger.log_results(epoch, train_dataloader.dataset.classes())

        # Stop training if test loss starts to increase
        # if test_loss > best_test_loss:
        #     break

        best_test_loss = test_loss

        base_dir = f"models/{config.run_name}"
        helpers.save_state(model, base_dir, f"model-e{epoch + 1}")

        if scheduler:
            helpers.save_state(scheduler, base_dir, f"scheduler")
