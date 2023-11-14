from typing import Dict, List

from pydantic import BaseModel


class BKChatModel(BaseModel):
    text: str
    filter: Dict = None


class KBTrainModel(BaseModel):
    knowledge_base_id: str
    training_data: Dict


class KBPredictModel(BaseModel):
    knowledge_base_id: str
    question: str
    history: List = []
    llm: bool = False
