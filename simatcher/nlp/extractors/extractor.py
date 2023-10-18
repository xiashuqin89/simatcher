from typing import Dict, Any, Text, List

from simatcher.nlp.base import Component


class EntityExtractor(Component):
    def _add_extractor_name(self, entities: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        for entity in entities:
            entity["extractor"] = self.name
        return entities

    def _add_processor_name(self, entity: Dict[Text, Any]) -> Dict[Text, Any]:
        if 'processors' in entity:
            entity['processors'].append(self.name)
        else:
            entity['processors'] = [self.name]
        return entity
