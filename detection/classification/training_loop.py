import wandb
import torch
import torchmetrics
from tqdm import tqdm
from dataset import *


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

    accuracy = torchmetrics.Accuracy(task=task, threshold=threshold, num_classes=num_classes).to(device)
    precision = torchmetrics.Precision(task=task, threshold=threshold, num_classes=num_classes).to(device)
    recall = torchmetrics.Recall(task=task, threshold=threshold, num_classes=num_classes).to(device)
    f1 = torchmetrics.F1Score(task=task, threshold=threshold, num_classes=num_classes).to(device)

    for epoch in range(num_epochs):
        epoch += initial_epoch

        # ---------------------- Trainig ----------------------------------
        train_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        model.train()
        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            prediction = model(images)
            loss = criterion(labels, prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_ema is not None:
                weights_ema(model_ema, model, ema_decay)

            if task == "binary":
                pass # TODO: сделать, чтобы правильно считались метрики
            elif task == "multiclass":
                labels = torch.argmax(labels, dim=1)

            train_metrics["loss"] += loss
            accuracy(prediction, labels)
            precision(prediction, labels)
            recall(prediction, labels)
            f1(prediction, labels)

        train_metrics["loss"] = train_metrics["loss"].item() / len(train_dl)
        train_metrics["accuracy"] = accuracy.compute()
        train_metrics["precision"] = precision.compute()
        train_metrics["recall"] = recall.compute()
        train_metrics["f1"] = f1.compute()
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1.reset()
        print("TRAINING")
        print(train_metrics["loss"], train_metrics["accuracy"], train_metrics["precision"], train_metrics["recall"], train_metrics["f1"])
        # -----------------------------------------------------------------

        # ---------------------- Validaion --------------------------------
        val_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        model.eval()
        with torch.no_grad():
            for images, labels in val_dl:
                images = images.to(device)
                labels = labels.to(device)

                prediction = model(images)

                if task == "binary":
                    pass # TODO: сделать, чтобы правильно считались метрики
                elif task == "multiclass":
                    labels = torch.argmax(labels, dim=1)

                val_metrics["loss"] += loss
                accuracy(prediction, labels)
                precision(prediction, labels)
                recall(prediction, labels)
                f1(prediction, labels)

            val_metrics["loss"] = val_metrics["loss"].item() / len(train_dl)
            val_metrics["accuracy"] = accuracy.compute()
            val_metrics["precision"] = precision.compute()
            val_metrics["recall"] = recall.compute()
            val_metrics["f1"] = f1.compute()
            accuracy.reset()
            precision.reset()
            recall.reset()
            f1.reset()
        print("VALIDATION")
        print(val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"])
        # -----------------------------------------------------------------

        # ---------------------- Checkpoint -------------------------------
        if wandb and use_wandb:
            wandb.log(
                {
                    "Loss": train_metrics["loss"],
                    "Accuracy": train_metrics["accuracy"],
                    "Precision": train_metrics["precision"],
                    "Recall": train_metrics["recall"],
                    "F1": train_metrics["f1"],
                    "Loss (val)": val_metrics["loss"],
                    "Accuracy (val)": val_metrics["accuracy"],
                    "Precision (val)": val_metrics["precision"],
                    "Recall (val)": val_metrics["recall"],
                    "F1 (val)": val_metrics["f1"],
                },
                step=epoch
            )

        if epoch % save_freq == 0:
            save_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics
            }

            if model_ema is not None:
                save_dict["model_ema"] = model_ema.state_dict()

            if wandb and use_wandb:
                save_dict["wandb_run_id"] = wandb.run.id

            torch.save(save_dict, f"checkpoint/{str(epoch).zfill(6)}.pt")
