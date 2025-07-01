from typing import List, TypedDict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langgraph.config import get_stream_writer
import os
import json

# ---- Environment Setup ----
os.environ["OPENAI_API_KEY"] = "Your-OpenAI-API-Key-Here"
os.environ["TAVILY_API_KEY"] = "Your-Tavily-API-Key-Here"


# ---- Models ----
class SearchResult(BaseModel):
    url: str = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    raw_content: str = Field(..., description="The raw content of the search result")


class ResearchState(TypedDict):
    query: str
    expanded_query: str
    documents: List[SearchResult]
    summary: str


# ---- Research Agent Class ----
class ResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="openai/gpt-4.1-mini",
            streaming=True,
        )
        self.tavily = TavilyClient()
        self.graph = self._build_graph()

    async def expand_query(self, state: ResearchState):
        response = await self.llm.ainvoke(
            f"You are a query writer agent. Rewrite this query to be more specific:\n\n'{state['query']}'."
        )
        return {"expanded_query": response.content.strip()}

    def run_tool_search(self, state: ResearchState):
        writer = get_stream_writer()
        writer(
            {
                "node": "run_tool_search",
                "token": "\nSearching academic sources and research databases...\n",
            }
        )

        try:
            results = self.tavily.search(
                query=state["expanded_query"], max_results=3, include_raw_content=True
            )
        except Exception as e:
            print("Search failed:", e)
            return {"documents": []}

        documents = [
            SearchResult(url=r["url"], title=r["title"], raw_content=r["raw_content"])
            for r in results.get("results", [])
            if r.get("url") and r.get("title") and r.get("raw_content")
        ]
        return {"documents": documents}

    async def summarize_documents(self, state: ResearchState):
        combined = "\n\n".join(
            f"Title: {doc.title}\nURL: {doc.url}\nContent: {doc.raw_content}"
            for doc in state["documents"]
        )
        prompt = (
            f"Please summarize the following search results:\n\n{combined}"
            f"\n\nThis is a summary of the results for '{state['query']}'."
        )
        response = await self.llm.ainvoke(prompt)
        return {"summary": response.content.strip()}

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("expand_query", self.expand_query)
        graph.add_node("run_tool_search", self.run_tool_search)
        graph.add_node("summarize_documents", self.summarize_documents)

        graph.add_edge(START, "expand_query")
        graph.add_edge("expand_query", "run_tool_search")
        graph.add_edge("run_tool_search", "summarize_documents")
        graph.add_edge("summarize_documents", END)

        return graph.compile()


# ---- FastAPI App Setup ----
app = FastAPI(title="LangGraph Research Assistant")
agent = ResearchAgent()


@app.get("/health")
def health():
    return {"status": "ok", "message": "API is operational"}


@app.post("/research/stream")
async def research_stream(request: Request):
    data = await request.json()
    query = data.get("query", "")

    async def stream():
        sent_expand = False
        async for mode, chunk in agent.graph.astream(
            {"query": query}, stream_mode=["messages", "custom"]
        ):
            if mode == "messages":
                node = chunk[1].get("langgraph_node")
                if node == "expand_query" and not sent_expand:
                    sent_expand = True
                    yield f"data: {json.dumps({'node': node, 'token': 'Broadening User query for better results...'})}\n\n"
                elif node == "summarize_documents" and chunk[0].content:
                    yield f"data: {json.dumps({'node': node, 'token': chunk[0].content})}\n\n"
            elif mode == "custom":
                yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
