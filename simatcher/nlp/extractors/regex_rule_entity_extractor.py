import os
import re
from typing import Dict, Text, Any, List, Optional, Iterable
from collections import deque

from simatcher.common.io import write_json_to_file, read_json_file
from simatcher.meta.message import Message
from simatcher.meta.model import Metadata
from simatcher.constants import (
    EXTRACTOR_REGEX_RULE, ENTITIES, TOKENS, INTENT, REGEX_FEATURES,
    ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE, ENTITY_REGEX_FILE_NAME
)
from simatcher.log import logger
from .extractor import EntityExtractor


class RegexRuleEntityExtractor(EntityExtractor):
    name = EXTRACTOR_REGEX_RULE
    provides = [ENTITIES]
    requires = [TOKENS, INTENT, REGEX_FEATURES]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 known_patterns: List[Any] = None):
        super(RegexRuleEntityExtractor, self).__init__(component_config)
        self.component_config = component_config
        self.case_sensitive = self.component_config.get('case_sensitive')
        self.patterns = known_patterns or []
        self.stupid_patterns = ['.*', '^.+$', '']

    def _preprocess_text(self, message: Message) -> Iterable:
        if all([slot['pattern'] in self.stupid_patterns for slot in self.patterns]):
            clean_params = re.split(r"\?+|\s+", message.text)[1:]
        else:
            params = re.split(r"\?+|\s+", str(''.join([letter if ord(letter) < 128 else '?'
                                                       for letter in message.text])))
            utterance = message.get(INTENT).get('utterance', '')
            clean_params = [
                letter.strip() for letter in params if letter and letter not in utterance
            ]
        return deque(clean_params)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """
        1, default value can not be used
        2, max len match method
        3, if contain special ${}, catch it by order
        4, add biz special function
        """
        clean_params = self._preprocess_text(message)
        entities = []
        if not self.case_sensitive:
            flags = re.IGNORECASE

        for slot in self.patterns:
            if slot.get('usage') and slot['usage'] != message.get(INTENT).get('id'):
                continue
            if slot[ENTITY_ATTRIBUTE_VALUE] in self.component_config.get('sys_pattern_value'):
                entities.append(slot)
                continue
            if slot['pattern'] in self.stupid_patterns:
                try:
                    slot[ENTITY_ATTRIBUTE_VALUE] = clean_params.popleft()
                    entities.append(slot)
                except IndexError:
                    logger.warning('There is no more text for extracting')
                continue

            max_len = 0
            pattern = re.compile(slot['pattern'], flags=flags)
            for segment in clean_params:
                result = pattern.search(segment)
                seg_len = len(result.group()) if result else 0
                if seg_len > max_len:
                    slot.update({
                        ENTITY_ATTRIBUTE_VALUE: result.group(),
                        ENTITY_ATTRIBUTE_START: result.start(),
                        ENTITY_ATTRIBUTE_END: result.end()
                    })
                    max_len = seg_len
            if slot[ENTITY_ATTRIBUTE_VALUE]:
                try:
                    clean_params.remove(slot[ENTITY_ATTRIBUTE_VALUE])
                except ValueError:
                    pass
            entities.append(slot)
        return entities

    def train(self,
              training_data: Dict = None,
              cfg: Dict = None,
              **kwargs):
        if training_data is None:
            message = kwargs.get('message')
            self.patterns.extend(message.get(REGEX_FEATURES))
        else:
            self.patterns.extend(training_data.get(REGEX_FEATURES))
        if not self.patterns:
            logger.warning('No regex input')

    def process(self, message: Message, **kwargs):
        """
        [
            {
                 'name': slot_name,
                 'pattern': sign_regex(list(set(v))),
                 'usage': intent
            }
        ]
        """
        if not self.patterns:
            self.train(message=message)

        extracted_entities = self._extract_entities(message)
        extracted_entities = self._add_extractor_name(extracted_entities)
        message.set(ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True)

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[EntityExtractor] = None,
             **kwargs) -> EntityExtractor:
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("regex_file", ENTITY_REGEX_FILE_NAME)
        if model_dir is not None:
            regex_file = os.path.join(model_dir, file_name)
            if os.path.exists(regex_file):
                known_patterns = read_json_file(regex_file)
                return RegexRuleEntityExtractor(meta, known_patterns=known_patterns)
            else:
                return RegexRuleEntityExtractor(meta)
        else:
            return cls(meta)

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        if self.patterns:
            regex_file = os.path.join(model_dir, ENTITY_REGEX_FILE_NAME)
            write_json_to_file(regex_file, self.patterns, indent=4)
        return {"entity_regex_file": ENTITY_REGEX_FILE_NAME}
