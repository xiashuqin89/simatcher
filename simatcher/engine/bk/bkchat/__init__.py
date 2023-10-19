import json
import itertools
from typing import Dict, List, Text, Generator
import urllib

import aiohttp

from simatcher.exceptions import ActionFailed
from simatcher.engine.base import Runner
from simatcher.constants import RANKING, INTENT
from .config import *


async def _load_data_from_remote(path: str,
                                 host: str = BKCHAT_APIGW_ROOT,
                                 method: str = 'POST', **kwargs) -> Dict:
    access_token = urllib.parse.urlencode({'bk_app_code': BKCHAT_APP_ID,
                                           'bk_app_secret': BKCHAT_APP_SECRET})
    url = f"{host}/{path}?{access_token}"
    try:
        async with aiohttp.request(method, url, **kwargs) as resp:
            if 200 <= resp.status < 300:
                return json.loads(await resp.text())
            raise ActionFailed(502)
    except aiohttp.InvalidURL:
        raise ActionFailed(401, 'API root url invalid')
    except aiohttp.ClientError:
        raise ActionFailed(403, 'HTTP request failed with client error')


class BKChatEngine:
    def __init__(self, pipeline_config: Dict = BKCHAT_PIPELINE_CONFIG, *args, **kwargs):
        self.pipeline_config = pipeline_config.copy()
        self.runner = Runner.load(self.pipeline_config)

    @classmethod
    async def load_slots(cls, **kwargs) -> Generator:
        response = await _load_data_from_remote('api/v1/exec/admin_describe_tasks', json=kwargs)
        tasks = response.get('data', [])
        regex_features = []
        if tasks:
            for task in tasks:
                slots = task['slots']
                slots.reverse()
                for slot in slots:
                    slot.setdefault('value', '')
                    slot['usage'] = task['index_id']
                    regex_features.append(slot)
        return regex_features

    @classmethod
    async def load_corpus_text(cls, **kwargs) -> List:
        response = await _load_data_from_remote('api/v1/exec/admin_describe_intents', json=kwargs)
        db_intents = response.get('data', [])
        if not db_intents:
            return None

        intent_map = {intent['id']: intent for intent in db_intents}

        response = await _load_data_from_remote('api/v1/exec/admin_describe_utterances',
                                                json={'data': {'index_id__in': list(intent_map.keys())}})
        db_utterances = response.get('data', [])
        return list(itertools.chain(*[
            [
                {
                    'utterance': sentence,
                    'id': intent_map[utterance['index_id']]['id'],
                    'intent_name': intent_map[utterance['index_id']]['intent_name'],
                    'is_commit': intent_map[utterance['index_id']]['is_commit'],
                    'status': intent_map[utterance['index_id']]['status'],
                    'available_user': intent_map[utterance['index_id']]['available_user'],
                    'available_group': intent_map[utterance['index_id']]['available_group'],
                    'biz_id': intent_map[utterance['index_id']]['biz_id'],
                    'updated_by': intent_map[utterance['index_id']]['updated_by'],
                    'approver': intent_map[utterance['index_id']]['approver'],
                    'notice_discern_success': intent_map[utterance['index_id']].get('notice_discern_success', True),
                    'notice_start_success': intent_map[utterance['index_id']].get('notice_start_success', True),
                    'notice_exec_success': intent_map[utterance['index_id']].get('notice_exec_success', True)
                } for sentence in utterance['content']
            ] for utterance in db_utterances
        ]))

    def classify(self,
                 text: Text,
                 pool: List,
                 output_properties: Dict = None,
                 only_output_properties=True,
                 regex_features: List = None):
        output_properties = output_properties or {RANKING, INTENT}
        message = self.runner.parse(text, output_properties=output_properties,
                                    pool=pool, text_col='utterance', regex_features=regex_features)
        return message.as_dict(only_output_properties=only_output_properties)

    def extractor(self):
        pass

    def run(self):
        pass
