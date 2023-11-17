import os
from typing import Text, Optional, Any, Dict

import requests
from langchain.llms import ChatGLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

from simatcher.nlp.base import Component
from simatcher.meta.message import Message
from simatcher.constants import REFINE_PROMPT_TEMPLATE
from simatcher.exceptions import MissingArgumentError
from simatcher.log import logger
from simatcher.meta.model import Metadata
from simatcher.exceptions import ActionFailed
from simatcher.common.io import py_cloud_pickle, py_cloud_unpickle


class Refiner(Component):
    def _load_llm_model(self, *args, **kwargs) -> LLM:
        if self.llm_model == 'chatglm2-6b':
            model = ChatGLM(
                endpoint_url=self.endpoint_url,
                max_token=self.max_token,
                history=self.history,
                top_p=0.7,
                model_kwargs={"sample_model_args": False},
            )
            return model
        elif self.llm_model == 'huanyuan':
            pass
        raise MissingArgumentError(f'{self.llm_model} is not existed')

    def _summary(self, model: LLM, question: str, context: str) -> str:
        prompt = PromptTemplate(template=REFINE_PROMPT_TEMPLATE, input_variables=["question", "context"])
        llm_chain = LLMChain(prompt=prompt, llm=model)
        try:
            return llm_chain.run(question=question, context=context)
        except ValueError:
            logger.error('llm server fail to run')

    def train(self, *args, **kwargs):
        pass

    def process(self, message: Message, **kwargs):
        pass


class HunYuan(LLM):
    pass
