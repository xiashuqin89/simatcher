import os
import re
from typing import Dict, Text, Any, List, Optional

from simatcher.common.io import (
    Message, TrainingData, TrainerModelConfig,
    logger, write_json_to_file, read_json_file
)
from simatcher.meta.message import Message
from simatcher.meta.model import Metadata
from simatcher.constants import (
    REGEX_EXTRACTOR_REGEX, ENTITIES, TOKENS, INTENT,
    ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE, ENTITY_REGEX_FILE_NAME
)
from .extractor import EntityExtractor


class RegexEntityExtractor(EntityExtractor):
    name = REGEX_EXTRACTOR_REGEX
    provides = [ENTITIES]
    requires = [TOKENS, INTENT]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 known_patterns: List[Any] = None):
        super(RegexEntityExtractor, self).__init__(component_config)
        self.component_config = component_config
        self.case_sensitive = self.component_config.get('case_sensitive')
        self.patterns = known_patterns or []

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        entities = []
        if not self.case_sensitive:
            flags = re.IGNORECASE

        for pattern in self.patterns:
            if pattern.get('usage') and pattern['usage'] != message.get(INTENT).get('name'):
                continue
            matchers = re.finditer(pattern['pattern'], message.text, flags=flags)
            for matcher in matchers:
                start = matcher.start()
                end = matcher.end()
                entities.append({
                    ENTITY_ATTRIBUTE_TYPE: pattern['name'],
                    ENTITY_ATTRIBUTE_START: start,
                    ENTITY_ATTRIBUTE_END: end,
                    ENTITY_ATTRIBUTE_VALUE: message.text[start:end]
                })

        return entities

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig,
              **kwargs):
        self.patterns.extend(training_data.regex_features)
        if not self.patterns:
            logger.warning('No regex input')

    def process(self, message: Message, **kwargs):
        if not self.patterns:
            return

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
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            known_patterns = read_json_file(regex_file)
            return RegexEntityExtractor(meta, known_patterns=known_patterns)
        else:
            return RegexEntityExtractor(meta)

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        if self.patterns:
            regex_file = os.path.join(model_dir, ENTITY_REGEX_FILE_NAME)
            write_json_to_file(regex_file, self.patterns, indent=4)
        return {"entity_regex_file": ENTITY_REGEX_FILE_NAME}
