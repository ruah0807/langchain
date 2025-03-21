from dotenv import load_dotenv
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_teddynote.tools import GoogleNews
from langchain_community.tools.tavily_search import TavilySearchResults
import random
from model import OllamaLLM

load_dotenv()

########## 1. 상태 정의 ##########
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # dummy_data: Annotated[str,"dummy"]
    # 사람에게 질문할지 여부를 묻는 상태 추가
    # ask_human : bool

# 랜덤한 해시값을 생성하는 generate_random_hash 함수를 사용
def generate_random_hash():
    part1 = f"{random.randint(0, 0xffff):05x}"
    part2 = f"{random.randint(0, 0xffff):09x}"
    part3 = f"{random.randint(0, 0xffff):07x}"
    return f"ruah-{part1}-{part2}-{part3}"

def agent_graph():
    ########## 2. 도구 정의 및 바인딩 ##########
    @tool
    def search_news(query: str) -> List[Dict[str, str]]:
        """Look up news by keyword"""
        news_tool = GoogleNews()
        return "\n".join(
        [f'- {news["content"]}' for news in news_tool.search_by_keyword(query, k=5)]
    )

    @tool
    def search_github(query:str)-> List[Dict[str, str]]:
        """Look up github code by keyword"""
        tavily_tool = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=True,
            include_domains=["github.com"]
        )
        return tavily_tool.invoke({"query": query})
    
    # 도구 바인딩
    tools= [search_news, search_github]

    # LLM 초기화
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.67)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.67)
    llm = OllamaLLM()

    # 도구와 LLM 결합
    llm_with_tools = llm.bind_tools(tools)

    ########## 3. 노드 추가 ##########
    def chatbot(state: State):
    # Ollama LLM을 사용하여 메시지 처리
        question = state["messages"][0].content
        response = llm_with_tools.invoke(question)
        return {
            "messages": [response],
        }

    # 상태 그래프 생성
    graph_builder = StateGraph(State)

    # 챗봇 노드 추가
    graph_builder.add_node("chatbot", chatbot)

    # 도구 노드 생성 및 추가
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # 조건부 엣지
    graph_builder.add_conditional_edges("chatbot", tools_condition)

    ########## 4. 엣지 추가 ##########
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    ########## 5. MemorySaver 추가 ##########
    memory = MemorySaver()

    ########## 6. interrupt_before 추가 ##########
    graph = graph_builder.compile(checkpointer=memory)

    return graph
