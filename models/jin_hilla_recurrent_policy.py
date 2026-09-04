"""Phase 2-B: Recurrent (LSTM) policy for M8 Jin Hilla training.

The current MLP policy (scripts/train.py Policy class) treats every step
independently and cannot remember how many hits have occurred since the last
soul slash, or how close the episode is to a red-skull termination. This
recurrent policy adds an LSTM core so the agent can carry that information
forward across steps within an episode.

Usage sketch in train.py (Phase 2 integration):

    from models.jin_hilla_recurrent_policy import RecurrentPolicy

    policy = RecurrentPolicy(observations, actions).to(device)
    hx = torch.zeros(1, policy.lstm_hidden, device=device)
    cx = torch.zeros(1, policy.lstm_hidden, device=device)

    # inside run_episode, replace `logits = policy(x)` with:
    logits, hx, cx = policy(x, hx, cx)

    # reset hx, cx to zeros at the start of every new episode (env.reset()).
"""

from __future__ import annotations

import torch
from torch import nn


class RecurrentPolicy(nn.Module):
    """MLP encoder + LSTMCell core + linear action head."""

    def __init__(
        self,
        observations: int,
        actions: int,
        embed_hidden: int = 128,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.encoder = nn.Sequential(
            nn.Linear(observations, embed_hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(embed_hidden, lstm_hidden)
        self.action_head = nn.Linear(lstm_hidden, actions)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hx = torch.zeros(batch_size, self.lstm_hidden, device=device)
        cx = torch.zeros(batch_size, self.lstm_hidden, device=device)
        return hx, cx

    def forward(
        self,
        observation: torch.Tensor,
        hx: torch.Tensor,
        cx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(observation)
        hx, cx = self.lstm(embedding, (hx, cx))
        logits = self.action_head(hx)
        return logits, hx, cx
