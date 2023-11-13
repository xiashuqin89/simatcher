import importlib
import re
from typing import Dict, Text, Any, List, Optional, Union

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from simatcher.meta.message import Message
from simatcher.constants import (
    SPLITTER_LANGCHAIN, TEXT_SPLITTER,
)
from simatcher.log import logger
from .splitter import Splitter


class LangchainSplitter(Splitter):
    name = SPLITTER_LANGCHAIN

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(LangchainSplitter, self).__init__(component_config)
        self.component_config = component_config
        self.splitter_name = self.component_config.get('splitter_name')
        self.chunk_size = self.component_config.get('chunk_size', 100)
        self.chunk_overlap = self.component_config.get('chunk_overlap', 0)
        self.zh_title_enhance = self.component_config.get('zh_title_enhance', False)
        try:
            text_splitter_module = importlib.import_module('langchain.text_splitter')
            self.text_splitter = getattr(text_splitter_module, TEXT_SPLITTER)(chunk_size=self.chunk_size,
                                                                              chunk_overlap=self.chunk_overlap)
        except (AttributeError, TypeError):
            self.text_splitter = ChineseRecursiveTextSplitter(chunk_size=self.chunk_size,
                                                              chunk_overlap=self.chunk_overlap)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return [
            'langchain',
        ]

    def tokenize(self, doc: Union[Text, List[Document]]) -> List[Document]:
        if isinstance(doc, Text):
            return self.text_splitter.split_documents([Document(page_content=doc)])
        else:
            return self.text_splitter.split_documents(doc)

    def train(self, training_data: Dict, cfg: Dict = None, **kwargs):
        list_of_documents = [
            Document(page_content=example.get('text'), metadata=example)
            for example in training_data.get('training_examples', [])
        ]
        training_data['split_docs'] = self.tokenize(list_of_documents)

    def process(self, message: Message, **kwargs):
        logger.info(f'langchain splitter: {message.text}')


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ):
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]
