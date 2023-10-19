from simatcher.nlp.featurizers import (
    BertFeaturizer,
)
from simatcher.nlp.classifiers import (
    L2Classifier,
)
from simatcher.nlp.extractors import (
    RegexRuleEntityExtractor,
)


COMPONENT_CLASSES = [
    BertFeaturizer,
    L2Classifier,
    RegexRuleEntityExtractor,
]
