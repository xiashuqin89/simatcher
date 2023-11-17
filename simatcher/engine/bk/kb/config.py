KB_ARCHIVE_PATH = 'archive'
KB_PIPELINE_CONFIG = {
    "language": "zh",
    "training_data": "",
    "pipeline": [
        {
            "name": "LangchainSplitter",
            "classifier_file": "LangchainSplitter.pkl",
            "class": "simatcher.nlp.splitters.LangchainSplitter",
            "chunk_size": 100,
            "chunk_overlap": 0,
            "zh_title_enhance": False
        },
        {
            "name": "LangchainFeaturizer",
            "classifier_file": "LangchainFeaturizer.pkl",
            "class": "simatcher.nlp.featurizers.LangchainFeaturizer",
            "pre_model": "text2vec-base-chinese"
        },
        {
            "name": "LangchainClassifier",
            "classifier_file": "LangchainClassifier.pkl",
            "class": "simatcher.nlp.classifiers.LangchainClassifier",
            "knowledge_base_id": "bk",
            "top_k": 4,
            "score_threshold": 0.5,
            "with_score": True
        }
    ],
    "version": "0.0.0"
}
KB_REFINE_NODE = {
    "name": "SummaryRefiner",
    "class": "simatcher.nlp.refiners.SummaryRefiner",
    "llm_model": "chatglm2-6b",
    "endpoint_url": "http://9.150.39.164:8081",
    "history": [],
}
KB_TRAIN_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "training_examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": ["string", "number"]},
                    "intent": {"type": "string"},
                    "entities": {"type": "array"},
                },
                "required": ["text"],
                "extra_options": ["intent", "entities"]
            }
        },
        "regex_features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["name", "pattern"]
            }
        },
        "entity_synonyms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "synonyms": {"type": "array"},
                },
                "required": ["value", "synonyms"]
            }
        },
        "intent_examples": {"type": "array"},
        "common_examples": {"type": "array"}
    },
    "required": ["training_examples"],
    "entity_synonyms": ["regex_features", "entity_synonyms", "intent_examples", "common_examples"]
}
