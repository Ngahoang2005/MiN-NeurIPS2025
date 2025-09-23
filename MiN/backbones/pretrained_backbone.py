from torch import nn
import timm
import torch
from backbones.ViT_MiN import VisionTransformer


def get_pretrained_backbone(args):
    name = args['backbone_type']
    if name == "pretrained_vit_b16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        model.layer_num = 12
    elif name == "pretrained_vit_b16_224_in21k_min":
        model_f = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model = VisionTransformer(num_classes=0, weight_init='skip', args=args)
        model.load_state_dict(model_f.state_dict(), strict=False)
        model.out_dim = 768
        model.layer_num = 12
    else:
        raise 'No this type backbone!'

    return model