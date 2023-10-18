import os
from typing import Optional

from fastapi import FastAPI, Request, Depends, Cookie, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from simatcher.exceptions import Error
from simatcher.cli.api.auth import SelfOAuth2PasswordBearer
from simatcher.cli.api.reponse import Response
from simatcher.engine import BKChatEngine
from .models import BKChatModel


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


@app.post("/api/bkchat/")
async def predict_bkchat(item: BKChatModel, bk_uid: Optional[str] = Cookie(None)):
    engine = BKChatEngine()
    pool = await engine.load_corpus_text(**item.filter)
    result = engine.classify(item.text, pool=pool)
    return Response(data=result)
