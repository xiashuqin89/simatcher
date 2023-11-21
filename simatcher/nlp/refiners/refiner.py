from typing import Optional, List, Any, Mapping

import requests
from langchain.llms import ChatGLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.callbacks.manager import CallbackManagerForLLMRun

from simatcher.nlp.base import Component
from simatcher.meta.message import Message
from simatcher.constants import REFINE_PROMPT_TEMPLATE
from simatcher.exceptions import MissingArgumentError
from simatcher.log import logger


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
        elif self.llm_model == 'jarvis':
            model = Jarvis(endpoint_url=self.endpoint_url,
                           api_key=self.api_key,
                           max_token=self.max_token,
                           history=self.history,
                           model_kwargs={'model': self.model})
            return model
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


class Jarvis(LLM):
    """Jarvis LLM Service"""
    endpoint_url: str = "http://127.0.0.1:8000/"
    """Jarvis LLM key"""
    api_key: str = ""
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    max_token: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""
    @property
    def _llm_type(self) -> str:
        return "jarvis"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        """
        "messages": [
            {
                "role": "user",
                "content": "你好"
            },
            {
                "role": "assistant",
                "content": "您好! 有什么我可以帮助您的?"
            },
            {
                "role": "user",
                "content": "你是谁?"
            }
        ]
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        _model_kwargs = self.model_kwargs or {}
        messages = []
        if self.with_history:
            messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": prompt
        })
        payload = {
            "messages": messages,
            # "top_p": self.top_p,
            # "temperature": self.temperature,
        }
        payload.update(_model_kwargs)
        payload.update(kwargs)
        logger.debug(f"Jarvis payload: {payload}")

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            # Check if response content does exists
            if isinstance(parsed_response, dict):
                content_keys = "choices"
                if content_keys in parsed_response:
                    text = parsed_response[content_keys][0]['message']['content']
                else:
                    raise ValueError(f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }
