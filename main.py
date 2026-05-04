from typing import List, TypedDict
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

FAISS_PERSIST_DIR = "./faiss_index"

# --- Nomic embedding wrapper (fix for removed query_instruction) ---
class NomicEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str):
        return super().embed_query("search_query: " + text)

    def embed_documents(self, texts: List[str]):
        texts = ["search_document: " + t for t in texts]
        return super().embed_documents(texts)

# --- Setup embeddings ---
model_name = "nomic-ai/nomic-embed-text-v1.5"
model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': True}

embeddings = NomicEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)