import timm
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
from training_loop import *
from torchvision import transforms


if __name__ == "__main__":
    root_path = f"C:/Users/conon/Documents/Datasets/cifar10/"
    train_path = os.path.join(root_path, "train")
    val_path = os.path.join(root_path, "test")

    train_labels_path = os.path.join(root_path, "train_labels.csv")
    val_labels_path = os.path.join(root_path, "test_labels.csv")

    # Fine-tuning hyperparameters
    task = "multiclass"
    device = "cuda"
    num_epochs = 30
    batch_size = 1024
    save_freq = 10
    learning_rate = 5e-6
    weight_decay = 1e-8
    label_smoothing = 0.1
    ema_decay = 0.9995  # TODO: добавить функцию, которая делает ema_decay

    # Dataset hyperparameters
    width = 32
    height = 32
    channels = 3
    num_classes = 10

    # Augmentation hyperparameters
    stochastic_depth_rate = [0.2, 0.4]
    data_augmentation = "RandAugment"
    alpha_mixup = 0.8
    alpha_cutmix = 1.0
    random_erase_prob = 0.25

    train_ds = ImageDataset(
        labels_file=train_labels_path,
        root_dir=train_path,
        num_classes=num_classes,
        transform=transforms.ToTensor(),
        target_transform=target_transform
    )
    val_ds = ImageDataset(
        labels_file=val_labels_path,
        root_dir=val_path,
        num_classes=num_classes,
        transform=transforms.ToTensor(),
        target_transform=target_transform
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    model = timm.create_model("fastvit_s12", num_classes=num_classes, in_chans=channels).to(device)
    if ema_decay is not None:
        model_ema = timm.create_model("fastvit_s12", num_classes=num_classes, in_chans=channels).to(device)
        model_ema.eval()
        weights_ema(model_ema, model, 0)
    else:
        model_ema = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)


    train(
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
        task=task,
        num_classes=num_classes,
        threshold=0.5
    )


