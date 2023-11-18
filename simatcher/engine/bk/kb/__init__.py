import os
import copy
from typing import Dict

from jsonschema import validate as json_validate

from simatcher.engine.base import Trainer, Runner
from simatcher.common.io import read_json_file
from simatcher.exceptions import MissingArgumentError
from simatcher.nlp.persistor import BKRepoPersistor
from simatcher.constants import KNOWLEDGE_BASE_DIR
from simatcher.log import logger
from .config import (
    KB_PIPELINE_CONFIG, KB_ARCHIVE_PATH, KB_TRAIN_DATA_SCHEMA, KB_REFINE_NODE
)


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


class KnowledgeBaseEngine:
    def __init__(self, pipeline_config: Dict = KB_PIPELINE_CONFIG, *args, **kwargs):
        self.pipeline_config = copy.deepcopy(pipeline_config)

    def _merge(self, old: Dict, new: Dict):
        for key, val in new.items():
            if key in old:
                val.extend(old[key])
        return new

    def _insert_url(self):
        pass

    def train(self,
              training_data: Dict,
              knowledge_base_id: str,
              llm_model: str = None,
              is_remove_archive: bool = False,
              **kwargs):
        """
        1, validate input name
        2, check whether kb is existed or create one
        3, save raw files
        4, save vector store
        """
        # do validate
        json_validate(training_data, KB_TRAIN_DATA_SCHEMA)
        if not validate_kb_name(knowledge_base_id):
            raise MissingArgumentError
        # merge data
        if os.path.isdir(os.path.join(KB_ARCHIVE_PATH, knowledge_base_id)):
            archive_train_data = read_json_file(os.path.join(KB_ARCHIVE_PATH, knowledge_base_id,
                                                             'model', 'training_data.json'))
            training_data = self._merge(archive_train_data, archive_train_data)
        # set config & train data
        self.pipeline_config['pipeline'][2]['knowledge_base_id'] = knowledge_base_id
        if llm_model is not None:
            self.pipeline_config['pipeline'].append(KB_REFINE_NODE)
        logger.info(f'Begin train model...\n{self.pipeline_config}')
        trainer = Trainer(self.pipeline_config)
        trainer.train(training_data)
        # persist
        logger.info(f'Begin persist model...')
        persistor = BKRepoPersistor() if is_remove_archive else None
        dir_name = trainer.persist(KB_ARCHIVE_PATH,
                                   persistor=persistor,
                                   project_name=knowledge_base_id,
                                   fixed_model_name='model')
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

    @staticmethod
    def clear(knowledge_base_id: str):
        """need to set a superuser"""
        if os.path.isdir(os.path.join(KB_ARCHIVE_PATH, knowledge_base_id)):
            os.chdir(KB_ARCHIVE_PATH)
            os.rmdir(knowledge_base_id)
        if os.path.isdir(KNOWLEDGE_BASE_DIR):
            os.rmdir(knowledge_base_id)
