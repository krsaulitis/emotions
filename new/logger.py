import numpy as np
import wandb
from config import Config
from sklearn.metrics import classification_report


class MetricsLogger:
    def __init__(self, config: Config):
        self.results = {
            'train': {
                'y_true': [],
                'y_pred': [],
            },
            'test': {
                'y_true': [],
                'y_pred': [],
            }
        }

        wandb.init(
            id=config.run_id,
            project='emotion-bert',
            config=config.__dict__,
            name=config.run_name,
        )

        config.run_id = wandb.run.id
        config.run_name = wandb.run.name

    def step_results(self, loop: str, loss, y, y_pred):
        self.results[loop]['loss'].append(loss)
        self.results[loop]['y_true'].append(y)
        self.results[loop]['y_pred'].append(y_pred)

    def get_loss(self, loop):
        return np.mean(self.results[loop]['loss'])

    def log_results(self, epoch, labels):
        for loop, results in self.results.items():
            y_true = np.concatenate(results["y_true"])
            y_pred = np.concatenate(results["y_pred"])

            report = classification_report(
                y_true,
                y_pred,
                target_names=labels,
                zero_division=0,
                output_dict=True,
            )

            wandb.log({loop: {"loss": self.get_loss(loop)}}, commit=False)
            wandb.log({loop: report}, commit=False)

        wandb.log({"epoch": epoch})
        self.results = {}

    def finish(self):
        wandb.finish()
        self.results = {}
