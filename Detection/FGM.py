# coding: UTF-8
import torch

EMB_NAME = 'embeddings.'

class ATModel():
    '''FGM - Fast Gradient Method'''

    def __init__(self, model):
        self.model = model    
        self.emb_backup = {}
        self.epsilon = 1.0

    def attack_param(self, param):
        norm = torch.norm(param.grad)
        if norm != 0:
            r_at = self.epsilon * param.grad / norm
            param.data.add_(r_at)

    def attack_emb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and EMB_NAME in name:
                self.emb_backup[name] = param.data.clone()
                self.attack_param(param)

    def restore_emb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and EMB_NAME in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}