import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF
import numpy as np
from transformers import AutoModel


class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self, encoder, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = encoder
        self.num_tags = config.num_labels
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2
        
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        # single dropout and multisample dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        """
        self.dropouts = []
        for p in np.linspace(0.1, 0.5, 5):
            self.dropouts.append(nn.Dropout(p))
        """


    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    
    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]
        
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)

        # single dropout
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        # multisample dropout
        # https://arxiv.org/abs/1905.09788
        """
        emissions = []
        for dropout in self.dropouts:
            emissions.append(self.hidden2tag(dropout(sequence_output)))
        emissions = torch.mean(torch.stack(emissions, axis = 0), axis = 0)
        """

        return emissions
    
    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())


# FGM
# https://wmathor.com/index.php/archives/1537/
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# weighted layer pooling
# https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
