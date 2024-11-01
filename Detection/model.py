import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, encoder, config, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)
    
    def get_unixcoder_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2), output_hidden_states=True)
        token_embeddings = out[0]
        code_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege
        return code_embeddings, token_embeddings
    
    def forward(self, input_ids, method_label=None):
        method_vec, _ = self.get_unixcoder_vec(input_ids)        
        method_logits = self.classifier(method_vec)
        method_prob = nn.functional.softmax(method_logits, dim=-1)
        loss = F.cross_entropy(method_logits, method_label)
        return loss, method_prob, method_label