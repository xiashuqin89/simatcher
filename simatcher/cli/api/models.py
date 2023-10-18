from typing import Dict

from pydantic import BaseModel


class BKChatModel(BaseModel):
    text: str
    filter: Dict = None
