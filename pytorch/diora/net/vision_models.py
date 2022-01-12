"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models

TYPE_CLASSES = {
    'chair': {'num_classes': 13, 'num_heads': 1},
    'table': {'num_classes': 10, 'num_heads': 1},
    'bed': {'num_classes': 8, 'num_heads': 1},
    'bag': {'num_classes': 3, 'num_heads': 1},
}

def get_model(options):
    backbone = resnet18()
    num_classes = TYPE_CLASSES[options.vision_type]['num_classes']
    num_heads = TYPE_CLASSES[options.vision_type]['num_heads']

    model = ClusteringModel(backbone, num_classes, num_heads)

    print(os.getcwd())

    if options.vision_pretrain_path is not None and os.path.exists(options.vision_pretrain_path):
        print('Loading model from ', options.vision_pretrain_path)
        state = torch.load(options.vision_pretrain_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    else:
        raise Exception("No vison_pretrain_path given.")
    
    return model


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nclusters = nclusters
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out

def resnet18():
    backbone = models.resnet18()
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 512}


def resnet50():
    backbone = models.__dict__['resnet50']()
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 2048}