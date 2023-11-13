from typing import Dict, List

from pydantic import BaseModel


class TrainerModelConfig(BaseModel):
    language: str
    training_data: Dict = None
    pipeline: List
    trained_at: str = ""
    version: str = ""

    @property
    def component_names(self):
        if self.pipeline:
            return [c.get("name") for c in self.pipeline]
        else:
            return []


class TrainingData(BaseModel):
    pass
