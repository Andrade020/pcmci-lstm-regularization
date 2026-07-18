"""Pydantic request/response models for the API."""
from typing import Optional

from pydantic import BaseModel


class RunConfig(BaseModel):
    """Subset of pipeline config that the UI is allowed to set."""
    window: int = 12
    tau_max: int = 6
    alpha: float = 0.05
    pc_alpha: float = 0.1
    hidden: int = 64
    num_layers: int = 2
    epochs: int = 150
    lr: float = 5e-4
    batch: int = 32
    transform: Optional[str] = None      # None | "log_diff" | "diff"

    def to_pipeline_cfg(self) -> dict:
        return {
            "window": self.window,
            "tau_max": self.tau_max,
            "alpha": self.alpha,
            "pc_alpha": self.pc_alpha,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch": self.batch,
        }


class RunExampleRequest(BaseModel):
    dataset: str
    config: RunConfig = RunConfig()


class JobResponse(BaseModel):
    job_id: str
