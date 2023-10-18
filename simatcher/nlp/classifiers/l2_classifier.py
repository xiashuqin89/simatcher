from typing import (
    Dict, Text, Any, List, Tuple
)

import numpy as np
from simatcher.constants import (
    CLASSIFIER_L2, TEXT_FEATURES, POOL_FEATURES,
    RANKING
)
from simatcher.meta.message import Message
from simatcher.algorithm.beta import SentenceFaiss
from .classifier import Classifier


class L2Classifier(Classifier):
    name = CLASSIFIER_L2
    provides = []
    requires = [TEXT_FEATURES, POOL_FEATURES]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(L2Classifier, self).__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['faiss-cpu']

    def train(self, training_data: Dict, cfg: Dict = None, **kwargs):
        pass

    def process(self, message: Message, **kwargs):
        sf = SentenceFaiss(message.get(POOL_FEATURES))
        sf.train()
        similarity = sf.process(message.get(TEXT_FEATURES), 5)
        message.set(RANKING, {
            'distances': similarity['distances'][0],
            'ann': similarity['ann'][0]
        })
        # results = pd.DataFrame({
        #     'distances': similarity['distances'][0],
        #     'ann': similarity['ann'][0]
        # })
        # merge = pd.merge(results, df, left_on='ann', right_index=True)
        # labels = df['id']
        # id = labels[similarity['ann'][0][0]]

    def predict(self, x: List) -> Tuple[np.ndarray, np.ndarray]:
        pass
