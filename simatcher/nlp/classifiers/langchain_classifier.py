import os
from typing import (
    Dict, Text, Any, List,
)

from langchain.vectorstores import FAISS

from simatcher.constants import (
    CLASSIFIER_LANGCHAIN, RANKING, INTENT, KNOWLEDGE_BASE_DIR,
    POOL_FEATURES,
)
from simatcher.meta.message import Message
from simatcher.exceptions import MissingArgumentError
from simatcher.log import logger
from .classifier import Classifier


class LangchainClassifier(Classifier):
    name = CLASSIFIER_LANGCHAIN
    requires = [POOL_FEATURES]
    provides = [INTENT, RANKING]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(LangchainClassifier, self).__init__(component_config)
        self.knowledge_base_id = self.component_config.get('knowledge_base_id', 'default')
        self.with_score = self.component_config.get('with_score', True)
        self.encoder_model = None
        self.split_docs = None
        self.vectorstore_index = None

    @classmethod
    def required_packages(cls) -> List[Text]:
        return [
            'langchain',
            'faiss-cpu',
            'sentence-transformers'
        ]

    def train(self, training_data: Dict = None, cfg: Dict = None, **kwargs):
        if 'split_docs' not in training_data or POOL_FEATURES not in training_data:
            raise MissingArgumentError(message=f'there is no split_docs or {POOL_FEATURES}')
        self.split_docs = training_data.get('split_docs')
        self.encoder_model = training_data.get(POOL_FEATURES)

    def process(self, message: Message, **kwargs):
        logger.info(f'langchain classifier: {message.text}')
        knowledge_base_dir = os.path.join(KNOWLEDGE_BASE_DIR, self.knowledge_base_id)
        vectorstore_index = FAISS.load_local(knowledge_base_dir, self.encoder_model)
        results = vectorstore_index.similarity_search_with_score(message.text)
        message.set(RANKING, results)
        message.set(INTENT, results[0])

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        knowledge_base_dir = os.path.join(KNOWLEDGE_BASE_DIR, self.knowledge_base_id)
        self.vectorstore_index = FAISS.from_documents(self.split_docs, self.encoder_model)
        self.vectorstore_index.save_local(knowledge_base_dir)
        super().persist(model_dir)
