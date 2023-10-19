import os
from typing import Dict, Text, Any, List, Union

import pandas as pd
import numpy as np
from torch import Tensor
from sentence_transformers import SentenceTransformer

from simatcher.constants import (
    FEATURIZER_BERT, TEXT_FEATURES, POOL_FEATURES,
    TEXT, POOL, TEXT_COL, POOL_DATA_FRAME
)
from simatcher.meta.message import Message
from .featurizer import Featurizer


class BertFeaturizer(Featurizer):
    name = FEATURIZER_BERT
    provides = [TEXT_FEATURES, POOL_FEATURES, POOL_DATA_FRAME]
    requires = [TEXT, POOL]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(BertFeaturizer, self).__init__(component_config)
        self.stop_words = self.component_config.get('stop_words')
        if self.stop_words and os.path.isfile(self.stop_words):
            with open(self.stop_words, mode='r', encoding='utf-8') as f:
                data = f.read()
                self.stop_words = frozenset(data.split('\n'))
        self.pool = None
        self.encoder_model = None
        self.train()

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['sentence-transformers']

    def train(self, training_data: Dict = None, cfg: Dict = None, **kwargs):
        # real-time training
        if training_data is None:
            pre_model = self.component_config.get('pre_model', 'all-MiniLM-L6-v2')
            self.encoder_model = SentenceTransformer(f'./model/{pre_model}')

    def process(self, message: Message, **kwargs):
        # matrix
        pool: List[Dict] = message.get(POOL)
        df = pd.DataFrame(pool)
        pool: Union[List[Tensor], np.ndarray, Tensor] = self.encoder_model.encode(df[message.get(TEXT_COL)])
        message.set(POOL_FEATURES, pool)
        message.set(POOL_DATA_FRAME, df)
        # vector
        text: Union[List[Tensor], np.ndarray, Tensor] = self.encoder_model.encode(message.text)
        message.set(TEXT_FEATURES, text)
