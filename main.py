from typing import TypedDict
import re
import json
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

UPPER_TH = 0.7
LOWER_TH = 0.3

class State(TypedDict):
    question:str
    docs : List[Document]
    good_docs : List[Document]
    verdict : str
    reason : str
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


# score- based doc eval
class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

def eval_each_doc_node(state: State) -> State:
    q = state["question"]

    scores : List[float] = []
    reasons : List[str] = []
    good: List[Document] = []
    for d in state["docs"]:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)

        if out.score > LOWER_TH:
            good.append(d)
    
    result = {
        **state,
        "good_docs": good,
    }

    # correct if at least one doc is above UPPER_TH,
    if any(s> UPPER_TH for s in scores):
        result["verdict"] = "CORRECT"
        result["reason"] = f"At least one retrieved chunk scored>{UPPER_TH}."
        return result

    # Incorrect if all docs are below LOWER_TH
    if len(scores)>0 and all(s < LOWER_TH for s in scores):
        why= "No chunks was sufficient"
        result["verdict"] = "INCORRECT"
        result["good_docs"] = []
        result["reason"] = f"All retrieved chunks scored<{LOWER_TH}. {why}"
        return result

    # Anything in between => AMBIGUOUS
    why = "Mixed relevance signals."
    result["verdict"] = "AMBIGUOUS"
    result["reason"] = f"Some chunks scored above {LOWER_TH} but none reached {UPPER_TH}. {why}"
    return result




# sentence level Decomposition
def decompose_to_sentences(text: str)-> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip())>10]


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
    context = "\n\n".join(d.page_content for d in state["good_docs"]).strip()

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
        **state,
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



def fail_node(state: State) -> State:
    return {"answer": f"FAIL: {state['reason']}"}

def ambiguous_node(state: State) -> State:
    return {"answer": f"Ambiguous: {state['reason']}"}

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    elif state["verdict"] == "INCORRECT":
        return "web_search"
    else:
        return "ambiguous"


g = StateGraph(State)
g.add_node('retrieve', retrieve)
g.add_node('eval_each_doc', eval_each_doc_node)
g.add_node('refine', refine)
g.add_node('generate', generate)
g.add_node('fail', fail_node)
g.add_node('ambiguous', ambiguous_node)

g.add_edge(START, 'retrieve')
g.add_edge('retrieve', 'eval_each_doc')
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "web_search": "fail",
        "ambiguous": "ambiguous"
    }
)
g.add_edge('refine', 'generate')
g.add_edge('generate', END)
g.add_edge('fail', END)

app = g.compile()


# 5) Run
# res = app.invoke({"question": "WHat is a transformer in deep learning.", "docs": [], "answer": ""})
res = app.invoke(
    {
        "question": "Explain the kernel trick in Support Vector Machines and how it allows for linear separation in high-dimensional spaces without explicit feature mapping.",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "answer": "",
    }
)

print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("\nOUTPUT:\n", res["answer"])
print(res["answer"])

# print(res['docs'][0].page_content)
# print('*'*100)
# print(res['docs'][1].page_content)
# print('*'*100)
# print(res['docs'][2].page_content)
# print('*'*100)
# print(res['docs'][3].page_content)
