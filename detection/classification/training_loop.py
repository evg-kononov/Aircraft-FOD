import wandb
import torch
import torchmetrics
from tqdm import tqdm
from typing import Union, Optional
from dataset import *


class Metrics:
    def __init__(
            self,
            task: str,
            threshold: float = 0.5,
            num_classes: Union[int, None] = None,
            top_k: int = 1,
            device: Union[str, torch.device] = None
    ):
        self.metrics = {
            "accuracy": torchmetrics.Accuracy(task=task, threshold=threshold, num_classes=num_classes, top_k=top_k).to(device),
            "precision": torchmetrics.Precision(task=task, threshold=threshold, num_classes=num_classes, top_k=top_k).to(device),
            "recall": torchmetrics.Recall(task=task, threshold=threshold, num_classes=num_classes, top_k=top_k).to(device),
            "f1": torchmetrics.F1Score(task=task, threshold=threshold, num_classes=num_classes, top_k=top_k).to(device),
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
    train_metrics = Metrics(task=task, threshold=threshold, num_classes=num_classes, device=device)
    val_metrics = Metrics(task=task, threshold=threshold, num_classes=num_classes, device=device)

    for epoch in range(num_epochs):
        epoch += initial_epoch

        # ---------------------- Training ----------------------------------
        train_loss = 0.0
        b = b
        model.train()
        with tqdm(total=len(train_dl), dynamic_ncols=True) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for images, labels in train_dl:
                images = images.to(device)
                labels = labels.to(device)

                # TODO: понять, как преобразются изображения меньшего размера (ведь нейронка обучалась на другом разрешении)
                prediction = model(images)
                loss = criterion(prediction, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if model_ema is not None:
                    weights_ema(model_ema, model, ema_decay)

                train_loss += loss
                train_metrics.update(prediction, labels)

                pbar.set_postfix(loss=loss.item())
                pbar.update()
                pbar.refresh()

            train_loss = train_loss.item() / len(train_dl)
            computed_train_metrics = train_metrics.compute()
            train_metrics.reset()

            pbar.set_postfix(loss=train_loss, **computed_train_metrics)
            pbar.update(n=0)
            pbar.refresh()
        # -----------------------------------------------------------------

        # ---------------------- Validation --------------------------------
        val_loss = 0.0
        model.eval()
        with tqdm(total=len(val_dl), dynamic_ncols=True) as pbar:
            pbar.set_description(f"Epoch {epoch} (val)")
            with torch.no_grad():
                for images, labels in val_dl:
                    images = images.to(device)
                    labels = labels.to(device)

                    prediction = model(images)
                    loss = criterion(prediction, labels)

                    val_loss += loss
                    val_metrics.update(prediction, labels)

                    pbar.set_postfix(loss=loss.item())
                    pbar.update()
                    pbar.refresh()

            val_loss = val_loss.item() / len(train_dl)
            computed_val_metrics = val_metrics.compute()
            val_metrics.reset()

            pbar.set_postfix(loss=val_loss, **computed_val_metrics)
            pbar.update(n=0)
            pbar.refresh()
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
