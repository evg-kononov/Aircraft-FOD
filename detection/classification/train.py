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
        "--data_path", type=str, default=r"C:\Users\admin\Desktop\Aircraft-FOD-DS-v2-Day-Binary-Reduced",
        help="Root directory for dataset"
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoints to resume training")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--use_wandb", action="store_true", help="Use weights and biases logging")

    # Training/Fine-tuning hyperparameters
    parser.add_argument(
        "--mode", type=str, default="fine-tuning", choices=["training", "fine-tuning"],
        help="Learning mode")
    parser.add_argument(
        "--pretrained", action="store_false",
        help="Whether or not there is a pre-training of the timm model")
    parser.add_argument(
        "--task", type=str, default="binary", choices=["binary", "multiclass", "multilabel"],
        help="Type of classification task"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate")  # 5e-6
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")  # 64
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs")  # 30
    parser.add_argument("--initial_epoch", type=int, default=1, help="Initialization epoch")  # 1
    parser.add_argument("--save_freq", type=int, default=None, help="Models save frequency")  # 10
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay coefficient") # 1e-8
    parser.add_argument("--label_smoothing", type=float, default=0, help="Label smoothing coefficient")  # 0.1
    parser.add_argument("--ema_decay", type=float, default=None, help="Exponential moving average coefficient")

    # Dataset hyperparameters
    # parser.add_argument("--width", type=int, default=288, help="Width of training images")
    # parser.add_argument("--height", type=int, default=288, help="Height of training images")
    parser.add_argument("--channels", type=int, default=3, help="Channels of training images")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes of training images")

    # Augmentation hyperparameters
    stochastic_depth_rate = 0.2
    data_augmentation = "RandAugment"
    alpha_mixup = 0.8
    alpha_cutmix = 1.0
    random_erase_prob = 0.25

    args = parser.parse_args()
    train_path = os.path.join(args.data_path, "training")
    val_path = os.path.join(args.data_path, "validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- Model ----------------------------------
    args.model_name = fastvit_sa24.__name__
    model, model_cfg = fastvit_sa24(
        num_classes=args.num_classes,
        in_chans=args.channels,
        device=device,
        mode=args.mode,
        pretrained=args.pretrained
    )
    if args.ema_decay is not None:
        model_ema, _ = fastvit_sa24(
            num_classes=args.num_classes,
            in_chans=args.channels,
            device=device,
            mode=args.mode,
            pretrained=args.pretrained
        )
        model_ema.eval()
        weights_ema(model_ema, model, 0)
    else:
        model_ema = None


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # TODO: реализовать label_smoothing для ВСЕWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
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
        root_dir=train_path,
        num_classes=args.num_classes,
        transform=train_transform,
        # target_transform=target_transform
    )
    val_ds = ImageDataset(
        root_dir=val_path,
        num_classes=args.num_classes,
        transform=val_transform,
        # target_transform=target_transform
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=4, persistent_workers=True)
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
    import time
    t1 = time.perf_counter()
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
    print(time.perf_counter() - t1)
