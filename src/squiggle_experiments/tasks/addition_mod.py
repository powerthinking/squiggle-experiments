from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdditionModTask:
    """
    Toy task:
      input tokens: [a, PLUS, b, EQUAL]
      target token: (a + b) % P

    Vocab layout:
      0..P-1  -> numbers
      P       -> PLUS
      P+1     -> EQUAL
    """
    p: int = 97

    @property
    def plus_id(self) -> int:
        return self.p

    @property
    def equal_id(self) -> int:
        return self.p + 1

    @property
    def vocab_size(self) -> int:
        return self.p + 2

    @property
    def seq_len(self) -> int:
        return 4

    def sample_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(0, self.p, (batch_size,), device=device)
        b = torch.randint(0, self.p, (batch_size,), device=device)

        x = torch.stack(
            [
                a,
                torch.full_like(a, self.plus_id),
                b,
                torch.full_like(a, self.equal_id),
            ],
            dim=1,
        )  # (B, 4)

        y = (a + b) % self.p  # (B,)
        return x, y
