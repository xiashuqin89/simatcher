from typing import (
    Dict, Text, Any, List, Tuple
)

import pandas as pd
import numpy as np
from simatcher.constants import (
    CLASSIFIER_L2, TEXT_FEATURES, POOL_FEATURES,
    RANKING, POOL, INTENT, POOL_DATA_FRAME
)
from simatcher.meta.message import Message
from simatcher.algorithm.beta import SentenceFaiss
from .classifier import Classifier


class L2Classifier(Classifier):
    name = CLASSIFIER_L2
    provides = [INTENT, RANKING]
    requires = [TEXT_FEATURES, POOL_FEATURES]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(L2Classifier, self).__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['faiss-cpu']

    def train(self, training_data: Dict = None, cfg: Dict = None, **kwargs):
        # real-time training
        if training_data is None:
            sf = SentenceFaiss(kwargs.get('message').get(POOL_FEATURES))
            sf.train()
            return sf

    def process(self, message: Message, **kwargs):
        model = self.train(message=message)
        similarity = model.process(message.get(TEXT_FEATURES), 5)
        similarity = pd.DataFrame({
            'distances': similarity['distances'][0],
            'ann': similarity['ann'][0]
        })
        pool_df = message.get(POOL_DATA_FRAME)
        merge = pd.merge(similarity, pool_df, left_on='ann', right_index=True)
        results = merge.to_dict('records')
        message.set(RANKING, results)
        message.set(INTENT, results[0])

    def predict(self, x: List) -> Tuple[np.ndarray, np.ndarray]:
        pass
