import torch


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
        """
        Polyak‐averages the parameters from `source` into `target`.
        After calling this, target.params = tau*source.params + (1-tau)*target.params.
        """
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)