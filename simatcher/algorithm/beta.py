from typing import Union, List, Dict

import faiss
import numpy as np
from torch import Tensor

"""
before: embedding
input: query[List], pool[List[List]]
output: item[Union[List, Dict]]
"""


class SentenceFaiss:
    def __init__(self, vector_pool: Union[List[Tensor], np.ndarray, Tensor]):
        self.vector_pool = vector_pool
        self.faiss_index = faiss.IndexFlatL2(vector_pool.shape[1])

    @classmethod
    def normalize(cls, vector: Union[List[Tensor], np.ndarray, Tensor]):
        return faiss.normalize_L2(vector)

    def process(self, query: Union[List[Tensor], np.ndarray, Tensor], top_k: int) -> Dict:
        _vector = np.array([query])
        self.normalize(_vector)
        distances, ann = self.faiss_index.search(_vector, k=top_k)
        return {
            'distances': distances,
            'ann': ann
        }

    def train(self):
        self.normalize(self.vector_pool)
        self.faiss_index.add(self.vector_pool)

    def load(self):
        pass

    def persist(self):
        pass
