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
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun

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
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

UPPER_TH = 0.7
LOWER_TH = 0.3

class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    refined_strips: List[str]
    web_docs: List[Document]
    answer: str

def retrieve(state):
    q = state['question']
    docs = base_retriever.invoke(q)
    return {"docs": docs}

# score-based doc eval
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

    scores: List[float] = []
    reasons: List[str] = []
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

    # correct if at least one doc is above UPPER_TH
    if any(s > UPPER_TH for s in scores):
        result["verdict"] = "CORRECT"
        result["reason"] = f"At least one retrieved chunk scored > {UPPER_TH}."
        return result

    # Incorrect if all docs are below LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        result["verdict"] = "INCORRECT"
        result["good_docs"] = []
        result["reason"] = f"All retrieved chunks scored < {LOWER_TH}."
        return result

    # Anything in between => AMBIGUOUS
    result["verdict"] = "AMBIGUOUS"
    result["reason"] = f"Some chunks scored above {LOWER_TH} but none reached {UPPER_TH}."
    return result

# sentence level Decomposition
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= 5]

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
def refine(state: State) -> State:
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        source_docs = state.get("good_docs", [])
        context = "\n\n".join(d.page_content for d in source_docs).strip()
    elif state.get("verdict") == "WEB_SEARCH" or state.get("web_docs"):
        source_docs = state.get("web_docs", [])
        context = "\n\n".join(d.page_content for d in source_docs).strip()
    else:
        context = ""

    if not context:
        return {
            **state,
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }

    strips = decompose_to_sentences(context)

    kept: List[str] = []
    for s in strips:
        try:
            result = filter_chain.invoke({"question": q, "sentence": s})
            if result.keep:
                kept.append(s)
        except Exception:
            kept.append(s)

    # If the filter is overly strict, fall back to the unfiltered sentences.
    refined_context = "\n".join(kept).strip() or "\n".join(strips).strip()

    return {
        **state,
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

def web_search(state: State) -> State:
    """Web search using DuckDuckGo (free, no API key)."""
    question = state["question"]

    try:
        results = search_tool.invoke(question)
        web_doc = Document(
            page_content=results,
            metadata={"source": "duckduckgo", "query": question}
        )
        return {**state, "web_docs": [web_doc], "verdict": "WEB_SEARCH"}
    except Exception as e:
        return {
            **state,
            "web_docs": [],
            "verdict": "INCORRECT",
            "reason": f"Search failed: {e}",
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
    refined = state.get("refined_context", "")
    if not refined:
        return {"answer": "I don't know based on the provided books (Refined context was empty)."}

    out = (answer_prompt | llm).invoke({"question": state["question"], "refined_context": refined})
    return {"answer": out.content}

def fail_node(state: State) -> State:
    return {**state, "answer": f"FAIL: {state['reason']}"}

def ambiguous_node(state: State) -> State:
    return {**state, "answer": f"Ambiguous: {state['reason']}"}

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    elif state["verdict"] == "INCORRECT":
        return "web_search"
    else:
        return "ambiguous"

# Build the graph
g = StateGraph(State)
g.add_node('retrieve', retrieve)
g.add_node('eval_each_doc', eval_each_doc_node)
g.add_node('refine', refine)
g.add_node('generate', generate)
g.add_node('fail', fail_node)
g.add_node('ambiguous', ambiguous_node)
g.add_node('web_search', web_search)  

g.add_edge(START, 'retrieve')
g.add_edge('retrieve', 'eval_each_doc')
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "web_search": "web_search",  # FIXED: Was pointing to "fail", now points to actual web_search node
        "ambiguous": "ambiguous"
    }
)
g.add_edge('refine', 'generate')
g.add_edge('web_search', 'refine')  # FIXED: After web search, go to refine
g.add_edge('generate', END)
g.add_edge('fail', END)
g.add_edge('ambiguous', END)  # FIXED: Add edge from ambiguous to END

app = g.compile()

# Run the query
res = app.invoke(
    {
        "question": "AI news from the last month.",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "refined_strips": [],
        "web_docs": [],
        "answer": "",
    }
)

print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("\nOUTPUT:\n", res["answer"])