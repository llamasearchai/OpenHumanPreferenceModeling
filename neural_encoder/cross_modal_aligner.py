import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Tuple

# Load config
with open("configs/eeg_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    fusion_conf = config["fusion"]


class CrossModalAligner(nn.Module):
    def __init__(self):
        super(CrossModalAligner, self).__init__()
        self.embedding_dim = fusion_conf["embedding_dim"]
        self.temperature = fusion_conf["temperature"]

        # Multi-Head Attention for Fusion
        # Query: Text (or central modality), Key/Value: EEG + Physio (Concatenated stack)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=fusion_conf["attention_heads"],
            batch_first=True,
        )

        # Learnable scalars for interpretability (simple softmax over 3 modalities)
        self.modality_weights = nn.Parameter(torch.ones(3))

    def contrastive_loss(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE Loss
        emb1: [batch, dim]
        emb2: [batch, dim]
        """
        # Normalize
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        # Similarity matrix
        logits = torch.matmul(emb1, emb2.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(
        self, h_eeg: torch.Tensor, h_text: torch.Tensor, h_physio: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            h_eeg, h_text, h_physio: [batch, embedding_dim]
        Returns:
            fused_embedding: [batch, embedding_dim]
            weights: Dict for logging
        """
        # 1. Alignment Loss (Training only, but computed here for logic flow usually returned separately)
        # We assume independent contrastive steps usually, but if end-to-end:
        # loss_eeg_text = self.contrastive_loss(h_eeg, h_text)

        # 2. Fusion via Attention
        # Query = h_text (unsqueeze to [batch, 1, dim])
        query = h_text.unsqueeze(1)

        # Key/Value = Stack of all modalities [batch, 3, dim]
        kv = torch.stack([h_eeg, h_text, h_physio], dim=1)

        attn_out, attn_weights = self.mha(query, kv, kv)
        # attn_weights: [batch, 1, 3] -> (EEG, Text, Physio) order in stack

        fused_embedding = attn_out.squeeze(1)

        # Extract mean attention weights for monitoring
        mean_weights = attn_weights.mean(dim=0).squeeze()
        weight_dict = {
            "eeg_weight": mean_weights[0],
            "text_weight": mean_weights[1],
            "physio_weight": mean_weights[2],
        }

        return fused_embedding, weight_dict
