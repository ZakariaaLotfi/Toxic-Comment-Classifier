import torch
import torch.nn as nn
import random
import numpy as np
from transformers import AutoModel, set_seed

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RobertaContextualClassifier(nn.Module):
    def __init__(self, model_path_or_name, num_extra_features=0):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_path_or_name)
        self.num_extra_features = num_extra_features
        
        hidden_size = self.roberta.config.hidden_size
        in_features = hidden_size + num_extra_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, extra_features=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token vector

        if self.num_extra_features > 0 and extra_features is not None:
            combined = torch.cat((cls_rep, extra_features), dim=1)
        else:
            combined = cls_rep

        logits = self.classifier(combined).squeeze(-1)
        return logits