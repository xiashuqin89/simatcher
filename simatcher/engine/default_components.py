from simatcher.nlp.featurizers import (
    BertFeaturizer, LangchainFeaturizer,
)
from simatcher.nlp.classifiers import (
    L2Classifier, LangchainClassifier,
)
from simatcher.nlp.extractors import (
    RegexRuleEntityExtractor,
)
from simatcher.nlp.splitters import (
    LangchainSplitter,
)


COMPONENT_CLASSES = [
    LangchainSplitter,
    BertFeaturizer,
    LangchainFeaturizer,
    L2Classifier,
    LangchainClassifier,
    RegexRuleEntityExtractor,
]
