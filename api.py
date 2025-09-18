from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import Retriever
from llm_client import call_groq
from cache import get_cached_response, set_cached_response
import os


app = FastAPI()


class QueryIn(BaseModel):
query: str




# initialize retriever globally (load index, docs)
RETRIEVER = Retriever()
CACHE_TTL = int(os.environ.get('CACHE_TTL_SECONDS', '86400'))


SYSTEM_PROMPT