import os
from typing import Optional

from fastapi import FastAPI, Request, Depends, Cookie, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from simatcher.cli.api.auth import SelfOAuth2PasswordBearer
from simatcher.cli.api.reponse import Response
from simatcher.engine import BKChatEngine, KnowledgeBaseEngine
from simatcher.exceptions import Error
from .models import (
    BKChatModel, KBTrainModel, KBPredictModel
)


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
    slots = await engine.load_slots(**item.filter)
    result = engine.classify(item.text, pool=pool, regex_features=slots)
    return Response(data=result)


@app.post("/api/kb/train/")
def train_kb(item: KBTrainModel, background_tasks: BackgroundTasks, bk_uid: Optional[str] = Cookie(None)):
    kb = KnowledgeBaseEngine()
    background_tasks.add_task(kb.train,
                              item.training_data,
                              item.knowledge_base_id,
                              item.llm_model,
                              item.is_remove_archive)
    return Response()


@app.post("/api/kb/predict/")
def predict_kb(item: KBPredictModel, bk_uid: Optional[str] = Cookie(None)):
    kb = KnowledgeBaseEngine()
    result = kb.predict(item.question, item.knowledge_base_id)
    try:
        intent = result['intent']
        response = intent['page_content']
        if 'summary' in intent:
            response = f'{response}\nSummary:\n{intent["summary"]}'
        source_documents = []
        for i, src in enumerate(result['intent_ranking']):
            content = src["page_content"]
            if src["metadata"].get('doc_url', '').startswith('http'):
                content = f'{content}\nWIKI: {src["metadata"]["doc_url"]}'
            source_documents.append(f'出处 [{i + 1}] {src["metadata"]["intent"]}:\n\n{content}')
    except (KeyError, TypeError):
        return {
            'question': item.question,
            'response': 'no answer for this knowledge base'
        }

    return {
        'question': item.question,
        'response': response,
        'history': [],
        'source_documents': source_documents
    }


@app.delete("/api/kb/{knowledge_base_id}/")
def delete_kb(knowledge_base_id: str, bk_uid: Optional[str] = Cookie(None)):
    KnowledgeBaseEngine.clear(knowledge_base_id)
    return Response()


@app.get("/api/kb/{knowledge_base_id}/check/")
def check_kb(knowledge_base_id: str, bk_uid: Optional[str] = Cookie(None)):
    result = KnowledgeBaseEngine.check(knowledge_base_id)
    return Response(result=result)
