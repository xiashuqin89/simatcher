TEXT = 'text'
TEXT_FEATURES = 'text_features'
TEXT_COL = 'text_col'
TEXT_SPLITTER = 'ChineseRecursiveTextSplitter'

TOKENS = 'tokens'
INTENT = 'intent'
ENTITIES = 'entities'
RANKING = 'intent_ranking'

VECTOR_STORE = 'vectorstore'

POOL = 'pool'
POOL_FEATURES = 'pool_features'
POOL_DATA_FRAME = 'pool_data_frame'

REGEX_FEATURES = 'regex_features'

ENTITY_SYNONYMS_FILE_NAME = 'entity_synonyms.json'
ENTITY_ATTRIBUTE_TYPE = 'entity'
ENTITY_ATTRIBUTE_START = 'start'
ENTITY_ATTRIBUTE_END = 'end'
ENTITY_ATTRIBUTE_VALUE = 'value'
ENTITY_REGEX_FILE_NAME = 'entity_regex.json'

SPLITTER_LANGCHAIN = 'LangchainSplitter'

FEATURIZER_BERT = 'BertFeaturizer'
FEATURIZER_LANGCHAIN = 'LangchainFeaturizer'

CLASSIFIER_L2 = 'L2Classifier'
CLASSIFIER_LANGCHAIN = 'LangchainClassifier'

EXTRACTOR_REGEX_RULE = 'RegexRuleEntityExtractor'

REFINE_SUMMARY = 'SummaryRefiner'

KNOWLEDGE_BASE_DIR = '/app/archive/knowledge_base'

REFINE_PROMPT_TEMPLATE = """
<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>

<已知信息>{context}</已知信息>

<问题>{question}</问题>
"""