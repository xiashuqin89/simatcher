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
            "knowledge_base_id": "debug",
            "with_score": True
        }
    ],
    "trained_at": "20231110-145515",
    "version": "0.0.0"
}
