############################## 에이전트 팀 정의 #################################

# Research Team 과 Doc Writing Team 을 정의

##################################### Research Team

# search agent 와 web scraping 을 담당하는 research_agent라는 두개의 작업자 노드를 가집니다. 이들을 생성하고 팀 감독자도 설정해 보겠습니다.

import operator
from typing import List, TypedDict
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_openai import ChatOpenAI
from c_tools_web import search_web, scrape_webpages
from a_agent_factory import agent_factory, MODEL_NAME, llm, create_team_supervisor
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 상태 정의
class ResearchState(TypedDict):
    messages : Annotated[List[BaseMessage], operator.add]
    team_members : List[str] # 멤버 에이전트 목록
    next: str # Supervisor 에이전트에게 다음 작업자를 선택하도록 지시


# Agent Factory를 사용한 에이전트 노드 생성예시
search_agent = create_react_agent(llm, tools=[search_web])
# 에이전트 노드 생성
search_node = agent_factory.create_agent_node(search_agent, name="Searcher")

# 웹 스크래핑 노드 생성
web_scraping_agent = create_react_agent(llm, tools=[scrape_webpages])
web_scraping_node = agent_factory.create_agent_node(web_scraping_agent, name="WebScraper")



# 라우팅할 노드를 선택하는 함수 정의
def get_next_node(x):
    return x["next"]

# supervisor 에이전트 생성
supervisor_agent = create_team_supervisor(
    MODEL_NAME,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: Search, WebScraper. Given the following user request,"
    " responsd with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["Searcher", "WebScraper"]
)



########### Research Team 그래프 생성 



# 그래프 생성
web_research_graph = StateGraph(ResearchState)

# 노드 추가
web_research_graph.add_node("Searcher", search_node)
web_research_graph.add_node("WebScraper", web_scraping_node)
web_research_graph.add_node("Supervisor", supervisor_agent)

# 엣지 추가
web_research_graph.add_edge("Searcher", "Supervisor")
web_research_graph.add_edge("WebScraper", "Supervisor")

web_research_graph.add_conditional_edges(
    "Supervisor",
    get_next_node,
    {
        "Searcher": "Searcher",
        "WebScraper": "WebScraper",
        "FINISH": END
    }
)

web_research_graph.set_entry_point("Supervisor")

# 그래프 컴파일
web_research_app = web_research_graph.compile(checkpointer=MemorySaver())

# ###### 그래프 시각화 
# from IPython.display import Image

# img = Image(web_research_app.get_graph(xray=True).draw_mermaid_png())
# with open("web_research_app.png", "wb") as f:
#     f.write(img.data)
def clean_text(text):
    # 줄바꿈과 탭을 공백으로 대체
    cleaned_text = text.replace('\n', ' ').replace('\t', ' ')
    # 여러 개의 공백을 하나의 공백으로 줄임
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

from langchain_core.runnables import RunnableConfig
import uuid
from langchain_teddynote.messages import invoke_graph

def run_graph(app, message: str, recursive_limit: int = 50):
    config = RunnableConfig(
        recursion_limit=recursive_limit, 
        configurable= {"thread_id": uuid.uuid4()}
    )

    # 질문 입력
    inputs = {
        "messages" : [HumanMessage(content=message)]
    }

    for chunk in app.stream(inputs, config, stream_mode="updates", subgraphs=True):
        print(chunk)
    # for namespace, chunk in app.stream(inputs, config, stream_mode="updates", subgraphs=True):
    #     for key, value in chunk.items():
    #         print(f"\n\n================ {key} =================\n\n")
    #         print(value)
            # # print(value)
            # if 'messages' in value:
            #     print(value["messages"][-1].content)

    # invoke_graph(app, inputs, config)

    return app.get_state(config).values

if __name__ == "__main__":
    output = run_graph(
        web_research_app,
        "https://finance.naver.com/news 의 주요 뉴스 정리해서 출력해줘. 출처(URL) 도 함께 출력해줘.",
    )

    print(output["messages"][-1].content)
