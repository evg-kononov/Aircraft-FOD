import argparse
import timm
from torch import nn
from torch.utils.data import DataLoader
from dataset import *
from training_loop import *
from torchvision import transforms
from config import *
from network import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier trainer/fine-tuner")

    # Basic hyperparameters
    # parser.add_argument("data_path", type=str, help="Root directory for dataset")
    parser.add_argument(
        "--data_path", type=str, default=f"C:/Users/admin/Documents/cifar10/",
        help="Root directory for dataset"
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoints to resume training")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--use_wandb", action="store_true", help="Use weights and biases logging")
    parser.add_argument("--model_name", type=str, default="fastvit_sa12", help="Name of model to instantiate from timm")
    parser.add_argument("--pretrained", type=str, default="store_true",
                        help="Whether or not there is a pre-training of the timm model")

    # Fine-tuning hyperparameters
    parser.add_argument(
        "--task", type=str, default="multiclass", choices=["binary", "multiclass", "multilabel"],
        help="Type of classification task"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--initial_epoch", type=int, default=1, help="Initialization epoch")
    parser.add_argument("--save_freq", type=int, default=10, help="Models save frequency")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay coefficient")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing coefficient")
    parser.add_argument("--ema_decay", type=float, default=None, help="Exponential moving average coefficient")

    # Dataset hyperparameters
    parser.add_argument("--width", type=int, default=32, help="Width of training images")
    parser.add_argument("--height", type=int, default=32, help="Height of training images")
    parser.add_argument("--channels", type=int, default=3, help="Channels of training images")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes of training images")

    # Augmentation hyperparameters
    stochastic_depth_rate = 0.2
    data_augmentation = "RandAugment"
    alpha_mixup = 0.8
    alpha_cutmix = 1.0
    random_erase_prob = 0.25

    args = parser.parse_args()
    train_path = os.path.join(args.data_path, "train")
    val_path = os.path.join(args.data_path, "val")
    train_labels_path = os.path.join(args.data_path, "train_labels.csv")
    val_labels_path = os.path.join(args.data_path, "val_labels.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Model ----------------------------------
    model, model_cfg = fastvit_sa12(num_classes=args.numclasses, in_chans=args.channels, device=device)
    if args.ema_decay is not None:
        model_ema, _ = fastvit_sa12(num_classes=args.numclasses, in_chans=args.channels, device=device)
        model_ema.eval()
        weights_ema(model_ema, model, 0)
    else:
        model_ema = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # -----------------------------------------------------------------

    # ---------------------- Dataset ----------------------------------
    # TODO: добавить возможность аугментации в transform
    train_transform = transforms.Compose([
        transforms.Resize(size=model_cfg["input_size"][1:], interpolation=model_cfg["interpolation"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_cfg["mean"], std=model_cfg["std"]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=model_cfg["input_size"][1:], interpolation=model_cfg["interpolation"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=model_cfg["mean"], std=model_cfg["std"]),
    ])

    train_ds = ImageDataset(
        labels_file=train_labels_path,
        root_dir=train_path,
        num_classes=args.num_classes,
        transform=train_transform,
        # target_transform=target_transform
    )
    val_ds = ImageDataset(
        labels_file=val_labels_path,
        root_dir=val_path,
        num_classes=args.num_classes,
        transform=val_transform,
        # target_transform=target_transform
    )

    train_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    # -----------------------------------------------------------------

    wandb_run_id = None
    if args.ckpt_path is not None:
        print("Load model:", args.ckpt_path)
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
        args.initial_epoch = checkpoint["epoch"]
        wandb_run_id = checkpoint.get("wandb_run_id", None)
        model.load_state_dict(checkpoint["model"])
        if args.ema_decay is not None:
            model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if wandb and args.use_wandb:
        wandb_init["config"].update(args.__dict__)
        wandb_init["id"] = wandb_run_id
        wandb.init(**wandb_init)

    train(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_ema=model_ema,
        **args.__dict__
    )
