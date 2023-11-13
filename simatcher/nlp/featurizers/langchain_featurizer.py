from typing import Dict, Text, Any, List

from langchain.embeddings import HuggingFaceEmbeddings

from simatcher.constants import (
    FEATURIZER_LANGCHAIN, POOL_FEATURES,
)
from simatcher.meta.message import Message
from simatcher.log import logger
from .featurizer import Featurizer


class LangchainFeaturizer(Featurizer):
    name = FEATURIZER_LANGCHAIN
    provides = [POOL_FEATURES]

    def __init__(self,
                 component_config: Dict[Text, Any] = None):
        super(LangchainFeaturizer, self).__init__(component_config)
        self.pre_model = self.component_config.get('pre_model', 'text2vec-base-chinese')

    @classmethod
    def required_packages(cls) -> List[Text]:
        return [
            'langchain',
            'sentence-transformers'
        ]

    def train(self, training_data: Dict = None, cfg: Dict = None, **kwargs):
        training_data[POOL_FEATURES] = HuggingFaceEmbeddings(model_name=f'./model/{self.pre_model}')

    def process(self, message: Message, **kwargs):
        logger.info(f'langchain featurizer: {message.text}')
