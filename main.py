from typing import TypedDict
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
from typing import List, TypedDict 
import re
from pydantic import BaseModel

load_dotenv()

FAISS_PERSIST_DIR = "./faiss_index"

# --- Setup BGE-Small (Fast & High Quality) ---
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
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
    strips : List[str]
    kept_strips : List[str]
    refined_context : str
    refined_strips : List[str]
    answer : str

def retrieve(state):
    q = state['question']
    # Use 'base_retriever' to fetch relevant documents
    docs = base_retriever.invoke(q)
    return {"docs": docs}

# sentence level Decomposition
def decompose_to_sentences(text: str)-> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip())>20]


# filter
class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that decides whether to keep or drop a sentence based on its relevance to the question."),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

# REFINING (Decompose -> Filter -> Recompose)
# -----------------------------
def refine(state: State) -> State:

    q = state["question"]

    # Combine retrieved docs into one context string
    context = "\n\n".join(d.page_content for d in state["docs"]).strip()

    # 1) DECOMPOSITION: context -> sentence strips
    strips = decompose_to_sentences(context)

    # 2) FILTER: keep only relevant strips
    kept: List[str] = []
    
    for s in strips:
        res = filter_chain.invoke({"question": q, "sentence": s})
        keep = getattr(res, "keep", None)
        if keep is None and isinstance(res, dict):
            keep = res.get("keep", False)
        if keep:
            kept.append(s)

    # 3) RECOMPOSE: glue kept strips back together (internal knowledge)
    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }




answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say: 'I don't know based on the provided books.'",
        ),
        ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
    ]
)

def generate(state):
    # Use refined_context produced by the refine step
    refined = state.get("refined_context", "")
    out = (answer_prompt | llm).invoke({"question": state["question"], "refined_context": refined})
    ans = getattr(out, "content", None)
    if ans is None and isinstance(out, dict):
        ans = out.get("content", str(out))
    return {"answer": ans}


g = StateGraph(State)
g.add_node('retrieve', retrieve)
g.add_node('refine', refine)
g.add_node('generate', generate)

g.add_edge(START, 'retrieve')
g.add_edge('retrieve', 'refine')
g.add_edge('refine', 'generate')
g.add_edge('generate', END)
app = g.compile()


# 5) Run
# res = app.invoke({"question": "WHat is a transformer in deep learning.", "docs": [], "answer": ""})
res = app.invoke({
    "question": "WHat is a transformer in deep learning",
    "docs": [],
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
    "answer": ""
})
print(res["answer"])

print(res['docs'][0].page_content)
print('*'*100)
print(res['docs'][1].page_content)
print('*'*100)
print(res['docs'][2].page_content)
print('*'*100)
print(res['docs'][3].page_content)
