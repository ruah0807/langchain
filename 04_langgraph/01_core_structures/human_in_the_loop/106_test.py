import asyncio
from a_agent_graph import agent_graph, State
from b_runnable_config import RunnableConfig
from langchain_ollama import ChatOllama

graph = agent_graph()
question = "AI관련 뉴스 찾아줘"

# 입력 상태 정의
input_state = State(messages=[("human", question)])

# config 설정
config = RunnableConfig(
    recursion_limit=10, # 최대 재귀 호출 횟수, 그 이상은 RecursionError 발생
    configurable={"thread_id": "1"},
)

for event in graph.stream(input_state, config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()

