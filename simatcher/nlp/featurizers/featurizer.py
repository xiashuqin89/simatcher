import os
from typing import Text, Dict, Any, Optional, List

import numpy as np

from simatcher.nlp.base import Component
from simatcher.constants import TEXT_FEATURES
from simatcher.meta.message import Message
from simatcher.meta.model import Metadata
from simatcher.log import logger
from simatcher.exceptions import PipelineRunningAbnormalError
from simatcher.common.io import py_cloud_pickle, py_cloud_unpickle


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_text_features(message: Message,
                                             additional_features: np.ndarray):
        if message.get(TEXT_FEATURES) is not None:
            return np.hstack((message[TEXT_FEATURES], additional_features))
        else:
            return additional_features

    @staticmethod
    def _transform_list2str(tokens: List[Text]):
        if not tokens:
            raise PipelineRunningAbnormalError('Need to do tokenizer before feature')
        document = ' '.join([token.text for token in tokens])
        return document.strip()

    @classmethod
    def load(cls,
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional[Component] = None,
             **kwargs):
        meta = model_metadata.for_component(cls.name)
        if model_dir is not None and meta.get('featurizer_file'):
            file_name = meta['featurizer_file']
            featurizer_file = os.path.join(model_dir, file_name)
            return py_cloud_unpickle(featurizer_file)
        else:
            logger.warning(f"Failed to load featurizer. Maybe path {model_dir} "
                           "doesn't exist")
            return cls(meta)

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        py_cloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}
