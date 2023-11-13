import abc
from typing import Text, Dict

from simatcher.nlp.base import Component
from simatcher.meta.message import Message
from simatcher.constants import TOKENS


class Splitter(Component):
    def process(self, message: Message, **kwargs):
        message.set(TOKENS, self.tokenize(message.text))

    def train(self, training_data: Dict, cfg: Dict = None, **kwargs):
        for example in training_data.get('training_examples', []):
            example.set(TOKENS, self.tokenize(example.text))

    @abc.abstractmethod
    def tokenize(self, text: Text):
        pass
