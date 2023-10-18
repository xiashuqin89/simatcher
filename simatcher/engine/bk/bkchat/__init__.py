import json
import itertools
from typing import Dict, List, Text
import urllib

import aiohttp
import pandas as pd

from simatcher.exceptions import ActionFailed
from simatcher.engine.base import Runner
from simatcher.constants import RANKING
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
    def __init__(self, config: Dict = BKCHAT_PIPELINE_CONFIG, *args, **kwargs):
        self.config = config
        self.runner = Runner.load(config)

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
                 only_output_properties=True):
        message = self.runner.parse(text, output_properties=output_properties, pool=pool)
        df = pd.DataFrame(pool)
        results = pd.DataFrame(message.get(RANKING))
        merge = pd.merge(results, df, left_on='ann', right_index=True)
        return merge.to_dict('records')

    def extractor(self):
        pass

    def run(self):
        pass
