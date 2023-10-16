import os
from typing import Optional

from fastapi import FastAPI, Request, Depends, Cookie, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from simatcher.exceptions import Error
from simatcher.cli.api.auth import SelfOAuth2PasswordBearer
from simatcher.cli.api.reponse import Response
from simatcher.engine.base import Runner


oauth2_scheme = SelfOAuth2PasswordBearer(tokenUrl="token")
app = FastAPI(dependencies=[Depends(oauth2_scheme)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('ALLOW_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Error)
async def unicorn_exception_handler(request: Request, exc: Error):
    return JSONResponse(
        status_code=418,
        content={
            'code': getattr(exc, 'retcode'),
            'result': False,
            'data': None,
            'message': getattr(exc, 'message')
        }
    )


@app.get("/")
async def root():
    return "Hello world!"


@app.get("/api/model/")
def load(bk_uid: Optional[str] = Cookie(None)):
    Runner.load({
        "language": "zh",
        "training_data": "",
        "pipeline": [
            {
                "name": "BertFeaturizer",
                "featurizer_file": "BertFeaturizer.pkl",
                "class": "simatcher.nlp.featurizers.BertFeaturizer"
            },
            {
                "name": "L2Classifier",
                "classifier_file": "L2Classifier.pkl",
                "class": "simatcher.nlp.classifiers.L2Classifier"
            }
        ],
        "trained_at": "20231016-145515",
        "version": "0.0.0"
    })
    return Response()
