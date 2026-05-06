from typing import List, TypedDict
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
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

# --- S-Tier Ingestion Logic (Saves your CPU) ---
if os.path.exists(FAISS_PERSIST_DIR):
    print("Loading existing FAISS index...")
    vector_store = FAISS.load_local(
        FAISS_PERSIST_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
else:
    print("No index found. Ingesting books (this will take time)...")
    docs = (
        PyPDFLoader("books/book1.pdf").load() +
        PyPDFLoader("books/book2.pdf").load() +
        PyPDFLoader("books/book3.pdf").load()
    )
    chunks = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)
    
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", errors="ignore").decode("utf-8","ignore")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_PERSIST_DIR)
    print("Index saved to disk.")


base_retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})
llm = ChatGroq( model="openai/gpt-oss-120b", temperature=0)

class State(TypedDict):
    question:str
    docs : List[Document]
    answer:str

def retrieve(state):
    q = state['question']
    # Use 'base_retriever', not 'retriever' from imports
    return {"docs": base_retriever.invoke(q)}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer only from the context. If not in context, say you don't know."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
def generate(state):
    context = "\n\n".join(d.page_content for d in state["docs"])
    out = (prompt | llm).invoke({"question": state["question"], "context": context})
    return {"answer": out.content}


g = StateGraph(State)
g.add_node('retriever', retrieve)
g.add_node('generate', generate)
g.add_edge(START, 'retriever')
g.add_edge('retriever', 'generate')
g.add_edge('generate', END)
app = g.compile()


# 5) Run
res = app.invoke({"question": "WHat is a transformer in deep learning.", "docs": [], "answer": ""})
print(res["answer"])

print(res['docs'][0].page_content)
print('*'*100)
print(res['docs'][1].page_content)
print('*'*100)
print(res['docs'][2].page_content)
print('*'*100)
print(res['docs'][3].page_content)
