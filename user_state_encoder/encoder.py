import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Optional
import yaml

# Load config
with open("user_state_encoder/config.yaml", "r") as f:
    config = yaml.safe_load(f)

from .positional_encoding import PositionalEncoding


class UserStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.sliding_window = config["sliding_window"]

        # Load pre-trained prompt encoder
        self.prompt_encoder = AutoModel.from_pretrained(config["model_name"])

        # Projection for input features to hidden dim
        input_dim = config["prompt_embedding_dim"] + config["choice_vector_dim"]
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)

        self.pos_encoder = PositionalEncoding(
            self.hidden_dim, max_len=self.sliding_window
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=config["num_heads"], batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config["num_layers"]
        )

    def process_features(
        self,
        prompt_embeddings: torch.Tensor,
        choice_vectors: torch.Tensor,
        context_features: Dict[str, float],
    ) -> torch.Tensor:
        """
        Combines prompt, choice, and context into a single input vector.
        Currently simple concatenation of prompt and choice, context handling to be refined.

        Args:
            prompt_embeddings: [batch, seq_len, 512]
            choice_vectors: [batch, seq_len, 64]
            context_features: Dictionary of scalar features

        Returns:
            Combined tensor [batch, seq_len, hidden_dim]
        """
        # Ensure inputs match config dims (placeholder validation)
        batch_size, seq_len, _ = prompt_embeddings.shape

        combined = torch.cat(
            [prompt_embeddings, choice_vectors], dim=-1
        )  # [batch, seq_len, 576]
        projected = self.input_projection(combined)  # [batch, seq_len, 768]

        return projected

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        choice_vectors: torch.Tensor,
        context_features: Optional[Dict[str, float]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeddings: [batch, seq_len, 512]
            choice_vectors: [batch, seq_len, 64]
            mask: Optional mask for padding/future tokens

        Returns:
            State embedding [batch, hidden_dim] (pooling over sequence)
        """
        src = self.process_features(
            prompt_embeddings, choice_vectors, context_features or {}
        )

        # Apply Positional Encoding (needs permutation for [seq_len, batch, dim] if batch_first=False)
        # But we set batch_first=True in TransformerEncoder, so we handle PE carefully
        src = src.permute(1, 0, 2)  # [seq_len, batch, dim]
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # [batch, seq_len, dim]

        # Transformer Encoding
        # Generate causal mask if not provided
        if mask is None:
            seq_len = src.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
            ).to(src.device)

        output = self.transformer_encoder(src, mask=mask, is_causal=True)

        # Pooling: Use the last token as the current state representation
        state_embedding = output[:, -1, :]

        return state_embedding

    def encode_user_state(self, texts: list[str]) -> torch.Tensor:
        """
        Helper for integration tests. Encodes a list of text events into a state.
        In real usage, this would include Tokenizer logic.
        """
        # Mock embeddings generation for text input
        # [batch=1, seq_len=len(texts), dim=512]
        seq_len = len(texts)
        if seq_len == 0:
            return torch.zeros(1, self.hidden_dim)

        # Mock prompt embeddings
        device = next(self.parameters()).device
        mock_prompt_emb = torch.randn(1, seq_len, 512).to(device)

        # Mock choice vectors (zeros)
        mock_choice = torch.zeros(1, seq_len, 64).to(device)

        # Forward pass
        with torch.no_grad():
            state = self.forward(mock_prompt_emb, mock_choice)

        return state
