import os
from typing import Text, Optional, Any, Dict

from simatcher.nlp.base import Component
from simatcher.meta.message import Message
from simatcher.meta.model import Metadata
from simatcher.common.io import py_cloud_pickle, py_cloud_unpickle


class Classifier(Component):
    def train(self, *args, **kwargs):
        pass

    def process(self, message: Message, **kwargs):
        pass

    def persist(self, model_dir: Text) -> Dict[Text, Text]:
        file_name = f'{self.name}.pkl'
        classifier_file = os.path.join(model_dir, file_name)
        py_cloud_pickle(classifier_file, self)
        return {"classifier_file": file_name}

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs) -> Any:
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", f'{cls.name}.pkl')
        if model_dir is not None:
            classifier_file = os.path.join(model_dir, file_name)
            if os.path.exists(classifier_file):
                return py_cloud_unpickle(classifier_file)
        else:
            return cls(meta)
