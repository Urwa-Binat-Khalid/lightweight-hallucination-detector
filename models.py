import torch
import torch.nn as nn
from transformers import RobertaModel


class FrozenRobertaBiGRU(nn.Module):
    def __init__(
        self,
        model_name="roberta-base",
        num_classes=2,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.encoder = RobertaModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size

        self.bigru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        sequence_output = encoder_output.last_hidden_state
        gru_output, _ = self.bigru(sequence_output)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (gru_output * mask).sum(dim=1) / mask.sum(dim=1)

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        return logits
