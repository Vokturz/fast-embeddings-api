import torch
from transformers import AutoTokenizer
from starlette.concurrency import run_in_threadpool
from optimum.onnxruntime import ORTModelForFeatureExtraction
from typing import List, Union
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from optimum.exporters.onnx import main_export
from shutil import rmtree
import os

router = APIRouter()
def create_app():
    app = FastAPI(
        title="Fast Embeddings API",
        version="0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    return app


class CreateEmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(description="The input to embed.")

class Embedding(BaseModel):
    embedding: List[float] = Field(max_length=2048)


class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]


tokenizer = None
model = None
device = None
provider = None

def save_model(model_name, _device, _reload=False):
    global device, provider
    device = _device
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    optimize = "O4" if device == "cuda" else "O3"
    ort_model_output = f"model_auto_opt_{optimize}"
    model_exists = os.path.exists(ort_model_output) and os.path.exists(ort_model_output + "/config.json")
    if (model_exists and _reload) or not model_exists:
        if model_exists:
            rmtree(ort_model_output)
        print(f"Loading {model_name} model and exporting it to ONNX..")
        _model = ORTModelForFeatureExtraction.from_pretrained(model_name, provider=provider, export=True,
                                                             cache_dir="/tmp")
        _model.save_pretrained(ort_model_output)
    elif (model_exists and not _reload):
        print(f"Model {model_name} already exists! use RELOAD=True to reload")
    return "./" + ort_model_output

def load_model(model_name, _device, _reload):
    global tokenizer, model
    ort_model_output = save_model(model_name, _device, _reload=_reload)
    print(f"Loading {model_name} model from {ort_model_output}..")
    tokenizer = AutoTokenizer.from_pretrained(ort_model_output)
    model = ORTModelForFeatureExtraction.from_pretrained(ort_model_output, provider=provider)


def _create_embedding(input: Union[str, List[str]]):
    global tokenizer, model
    if not isinstance(input, list):
            input = [input]
            
    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        sentence_embeddings = model(**encoded_input)[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()
    data = [Embedding(embedding=embedding) for embedding in sentence_embeddings]
    return CreateEmbeddingResponse(data=data)

@router.post(
    "/v1/embeddings",
    response_model=CreateEmbeddingResponse
)
async def create_embedding(
    request: CreateEmbeddingRequest):
    return await run_in_threadpool(
        _create_embedding, **request.dict()
    )