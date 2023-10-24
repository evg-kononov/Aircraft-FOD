import wandb
import torch
import torchmetrics
from tqdm import tqdm
from typing import Union
from dataset import *


class Metrics:
    def __init__(
            self,
            task: str,
            threshold: float = 0.5,
            num_classes: Union[int, None] = None,
            device: Union[str, torch.device] = None
    ):
        self.metrics = {
            "Accuracy": torchmetrics.Accuracy(task=task, threshold=threshold, num_classes=num_classes).to(device),
            "Precision": torchmetrics.Precision(task=task, threshold=threshold, num_classes=num_classes).to(device),
            "Recall": torchmetrics.Recall(task=task, threshold=threshold, num_classes=num_classes).to(device),
            "F1": torchmetrics.F1Score(task=task, threshold=threshold, num_classes=num_classes).to(device),
        }

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for key in self.metrics:
            self.metrics[key].update(preds, target)

    def compute(self):
        computed_metrics = dict()
        for key in self.metrics:
            computed_metrics[key] = self.metrics[key].compute().item()
        return computed_metrics

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()


def weights_ema(model_ema, model, decay=0.999):
    """
    For visualizing and evaliating generator output at any given point during the training
    use an exponential running average for the weights of the generator with decay 0.999.
    """
    par1 = dict(model_ema.named_parameters())
    par2 = dict(model.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def train(
        train_dl,
        val_dl,
        model,
        optimizer,
        criterion,
        num_epochs,
        device,
        save_freq,
        initial_epoch=1,
        use_wandb=False,
        model_ema=None,
        ema_decay=0.9995,
        task="binary",
        num_classes=None,
        threshold=0.5,
):
    pbar = range(num_epochs)
    pbar = tqdm(pbar, initial=initial_epoch, dynamic_ncols=True)

    train_metrics = Metrics(task=task, threshold=threshold, num_classes=num_classes, device=device)
    val_metrics = Metrics(task=task, threshold=threshold, num_classes=num_classes, device=device)

    for epoch in range(num_epochs):
        epoch += initial_epoch

        # ---------------------- Training ----------------------------------
        train_loss = 0.0
        model.train()
        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            # TODO: понять, как преобразются изображения меньшего размера (ведь нейронка обучалась на другом разрешении)
            prediction = model(images)
            loss = criterion(labels, prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_ema is not None:
                weights_ema(model_ema, model, ema_decay)

            if task == "binary":
                pass  # TODO: сделать, чтобы правильно считались метрики
            elif task == "multiclass":
                labels = torch.argmax(labels, dim=1)

            train_loss += loss
            train_metrics.update(prediction, labels)

        train_loss = train_loss.item() / len(train_dl)
        computed_train_metrics = train_metrics.compute()
        train_metrics.reset()

        print("TRAINING")
        print(computed_train_metrics["loss"], computed_train_metrics["accuracy"], computed_train_metrics["precision"],
              computed_train_metrics["recall"], computed_train_metrics["f1"]
              )
        # -----------------------------------------------------------------

        # ---------------------- Validation --------------------------------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_dl:
                images = images.to(device)
                labels = labels.to(device)

                prediction = model(images)

                if task == "binary":
                    pass  # TODO: сделать, чтобы правильно считались метрики
                elif task == "multiclass":
                    labels = torch.argmax(labels, dim=1)

                val_loss += loss
                val_metrics.update(prediction, labels)

            val_loss = val_loss.item() / len(train_dl)
            computed_val_metrics = val_metrics.compute()
            val_metrics.reset()

        print("VALIDATION")
        print(val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"])
        # -----------------------------------------------------------------

        # ---------------------- Checkpoint -------------------------------
        if wandb and use_wandb:
            log_dict = {
                "Loss": train_loss,
                "Loss (val)": val_loss,
            }
            log_dict.update(computed_train_metrics)

            for key in computed_val_metrics:
                computed_val_metrics[key + " (val)"] = computed_val_metrics.pop(key)
            log_dict.update(computed_val_metrics)

            wandb.log(
                log_dict,
                step=epoch
            )

        if epoch % save_freq == 0:
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics
            }

            if model_ema is not None:
                save_dict["model_ema"] = model_ema.state_dict()

            if wandb and use_wandb:
                save_dict["wandb_run_id"] = wandb.run.id

            torch.save(save_dict, f"checkpoint/{str(epoch).zfill(6)}.pt")
