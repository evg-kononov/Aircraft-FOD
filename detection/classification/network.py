import timm
from torchvision.transforms.functional import InterpolationMode
from typing import Literal


def vgg11(num_classes, in_chans, device, mode: Literal["training", "feature-extraction", "fine-tuninig"] = None, pretrained=False):
    model = timm.create_model("vgg11", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans).to(device)

    if mode == "feature-extraction" and pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    elif mode == "fine-tuning":
        requires_grad = False
        for i, module in enumerate(model.named_parameters()):
            name, param = module
            if "pre_logits" in name:
                requires_grad = True
            param.requires_grad = requires_grad

    model_cfg = model.default_cfg
    model_cfg["interpolation"] = InterpolationMode.BILINEAR
    return model, model_cfg


def efficientnet_b2(num_classes, in_chans, device, mode: Literal["training", "feature-extraction", "fine-tuninig"] = None, pretrained=False):
    model = timm.create_model("efficientnet_b2", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans).to(device)

    if mode == "feature-extraction" and pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif mode == "fine-tuning":
        requires_grad = False
        for i, module in enumerate(model.named_parameters()):
            name, param = module
            if "blocks.1" in name:
                requires_grad = True
            param.requires_grad = requires_grad

    model_cfg = model.default_cfg
    model_cfg["interpolation"] = InterpolationMode.BICUBIC
    return model, model_cfg


def efficientnet_b3(num_classes, in_chans, device, mode: Literal["training", "feature-extraction", "fine-tuninig"] = None, pretrained=False):
    model = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans).to(device)

    if mode == "feature-extraction" and pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif mode == "fine-tuning":
        requires_grad = True
        for i, module in enumerate(model.named_parameters()):
            name, param = module
            if "blocks.6" in name:
                requires_grad = True
            param.requires_grad = requires_grad

    model_cfg = model.default_cfg
    model_cfg["interpolation"] = InterpolationMode.BICUBIC
    return model, model_cfg


def fastvit_sa12(num_classes, in_chans, device, mode: Literal["training", "feature-extraction", "fine-tuninig"] = None, pretrained=False):
    model = timm.create_model("fastvit_sa12", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans).to(device)

    if mode == "feature-extraction" and pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    elif mode == "fine-tuning":
        requires_grad = True
        for i, module in enumerate(model.named_parameters()):
            name, param = module
            if "stages.2" in name:
                requires_grad = True
            param.requires_grad = requires_grad

    model_cfg = model.default_cfg
    model_cfg["interpolation"] = InterpolationMode.BICUBIC
    return model, model_cfg


def fastvit_sa24(num_classes, in_chans, device, mode: Literal["training", "feature-extraction", "fine-tuninig"] = None, pretrained=False):
    model = timm.create_model("fastvit_sa24", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans).to(device)

    if mode == "feature-extraction" and pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    elif mode == "fine-tuning":
        requires_grad = True
        for i, module in enumerate(model.named_parameters()):
            name, param = module
            if "stages.2" in name:
                requires_grad = True
            param.requires_grad = requires_grad

    model_cfg = model.default_cfg
    model_cfg["interpolation"] = InterpolationMode.BICUBIC
    return model, model_cfg
