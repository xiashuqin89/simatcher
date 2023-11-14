import os
from typing import Dict, List

from jsonschema import validate as json_validate
from jsonschema.exceptions import ValidationError

from simatcher.engine.base import Trainer, Runner
from simatcher.log import logger
from simatcher.common.io import read_json_file
from simatcher.exceptions import MissingArgumentError
from .config import (
    KB_PIPELINE_CONFIG, KB_ARCHIVE_PATH, TRAIN_DATA_SCHEMA
)


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


class KnowledgeBaseEngine:
    def __init__(self, pipeline_config: Dict = KB_PIPELINE_CONFIG, *args, **kwargs):
        self.pipeline_config = pipeline_config.copy()

    def _merge(self, old: Dict, new: Dict):
        for key, val in new.items():
            if key in old:
                val.extend(old[key])
        return new

    def _insert_url(self):
        pass

    def train(self,
              training_data: Dict,
              knowledge_base_id: str):
        """
        1, validate input name
        2, check whether kb is existed or create one
        3, save raw files
        4, save vector store
        """
        json_validate(training_data, TRAIN_DATA_SCHEMA)
        if not validate_kb_name(knowledge_base_id):
            raise MissingArgumentError
        if os.path.isdir(os.path.join(KB_ARCHIVE_PATH, knowledge_base_id)):
            archive_train_data = read_json_file(os.path.join(KB_ARCHIVE_PATH, knowledge_base_id, 'model'))
            training_data = self._merge(archive_train_data, archive_train_data)
        trainer = Trainer(self.pipeline_config)
        trainer.train(training_data)
        dir_name = trainer.persist(KB_ARCHIVE_PATH, project_name=knowledge_base_id, fixed_model_name='model')
        logger.info(f'Train successfully...Archive at: {dir_name}')
        return dir_name

    def predict(self,
                question: str,
                knowledge_base_id: str) -> Dict:
        """
        1, check kb
        2, search docs
        3, llm prompt do summary
        4, local_doc_url
        5, ping iwiki url
        """
        if not validate_kb_name(knowledge_base_id):
            raise MissingArgumentError
        runner = Runner.load(os.path.join(KB_ARCHIVE_PATH, knowledge_base_id, 'model'))
        message = runner.parse(question)
        return message.as_dict()