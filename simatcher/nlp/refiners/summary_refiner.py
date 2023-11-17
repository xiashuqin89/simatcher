from typing import Dict, Text, Any

from simatcher.constants import (
    REFINE_SUMMARY, RANKING, INTENT, REFINE_PROMPT_TEMPLATE
)
from simatcher.meta.message import Message
from simatcher.log import logger
from .refiner import Refiner


class SummaryRefiner(Refiner):
    name = REFINE_SUMMARY
    requires = [INTENT, RANKING]
    provides = [INTENT, RANKING]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(SummaryRefiner, self).__init__(component_config)
        self.llm_model = self.component_config.get('llm_model', 'chatglm2-6b')
        self.endpoint_url = self.component_config.get('endpoint_url')
        self.max_token = self.component_config.get('max_token', 20000)
        self.history = self.component_config.get('history', [])

    def process(self, message: Message, **kwargs):
        intent = message.get(INTENT)
        if not intent:
            logger.error('there is no intent to be used for llm')
            return
        model = self._load_llm_model()
        intent['summary'] = self._summary(model, message.text, intent['metadata']['text'])
        message.set(INTENT, intent)
