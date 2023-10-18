import os


BKCHAT_APP_ID = os.getenv('BKCHAT_APP_ID')
BKCHAT_APP_SECRET = os.getenv('BKCHAT_APP_SECRET')
BKCHAT_APIGW_ROOT = os.getenv('BKCHAT_APIGW_ROOT')
BKCHAT_PIPELINE_CONFIG = {
    "language": "zh",
    "training_data": "",
    "pipeline": [
        {
            "name": "BertFeaturizer",
            "featurizer_file": "BertFeaturizer.pkl",
            "class": "simatcher.nlp.featurizers.BertFeaturizer",
            "pre_model": "sbert-chinese-general-v2"
        },
        {
            "name": "L2Classifier",
            "classifier_file": "L2Classifier.pkl",
            "class": "simatcher.nlp.classifiers.L2Classifier"
        }
    ],
    "trained_at": "20231016-145515",
    "version": "0.0.0"
}
