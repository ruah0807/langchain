from typing import List, Annotated, TypedDict
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from c_tools_web import scrape_webpages, search_web
from c_tools_writing import python_repl_tool, read_document, write_document, edit_document, create_outline, WORKING_DIRECTORY
from a_agent_factory import agent_factory, MODEL_NAME, llm, create_team_supervisor

from x_team_researcher import web_research_app, run_graph
from x_team_writer import authoring_app


########################### Super Graph 생성 ###############################

# 해당 설계에서는 상향식 계획 정책을 적용한다. 
# 이미 두개의 그래프를 생성했지만 이들 간의 작업을 어떻게 라우팅할지 결정해야한다.

# Super Graph를 정의하여 이전 두 graph를 조정하고, 
# 이 상위 수준 상태가 서로 다른 그래프 간에 어떻게 공유되는지를 정의하는 연결 요소를 추가한다.

# 팀 감독자 노드 생성
supervisor_node = create_team_supervisor(
    MODEL_NAME,
    "You are asupervisor tasked with managing a conversation between the"
    " following teams : ['ResearchTEam', 'PaperWritingTeam']. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " Respond with FINISH.",
    ["ResearchTeam", "PaperWritingTeam"] 
)


## Super Graph 상태와 노드 정의

class State(TypedDict):
    messages : Annotated[List[BaseMessage], operator.add]
    # 라우팅 결정
    next: str

# 마지막 메시지 반환노드
def get_last_message(state: State) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, str):
        return {"messages": [HumanMessage(content = last_message)]}
    else:
        return{"messages": [last_message.content]}
    
# 응답 종합 노드
def join_graph(response: dict):
    # 마지막 메시지를 추출하여 메시지 목록으로 반환
    return {"messages": [response["messages"][-1]]}


# 라우팅할 노드를 선택하는 함수 정의
def get_next_node(x):
    return x["next"]

######### Super Graph 정의
# 팀 2개를 연결하는 슈퍼 그래프 정의

super_graph = StateGraph(State)
# 서브 supervisor에서 받은 마지막 메시지 | 웹연구팀 graph | 응답 종합노드드
super_graph.add_node("ResearchTeam", get_last_message | web_research_app | join_graph)
super_graph.add_node("PaperWritingTeam", get_last_message | authoring_app | join_graph)
super_graph.add_node("Supervisor", supervisor_node)

super_graph.add_edge("ResearchTeam", "Supervisor")
super_graph.add_edge("PaperWritingTeam", "Supervisor")

super_graph.add_conditional_edges(
    "Supervisor",
    get_next_node,
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END
    }
)

super_graph.set_entry_point("Supervisor")

super_graph = super_graph.compile(checkpointer=MemorySaver())


###############################

if __name__ == "__main__":
    ## # ###### 그래프 시각화 
    from IPython.display import Image, Markdown, display

    img = Image(super_graph.get_graph(xray=True).draw_mermaid_png())
    with open("hierarchical_agent_team.png", "wb") as f:
        f.write(img.data)

#     output = run_graph(
#         super_graph,
#         """주제: multi-agent 구조를 사용하여 복잡한 작업을 수행하는 방법

#         상세 가이드라인:  
# - 주제에 대한 Arxiv 논문 형식의 리포트 생성
# - Outline 생성
# - 각각의 Outline 에 대해서 5문장 이상 작성
# - 상세내용 작성시 만약 chart 가 필요하면 차트 생성 및 추가
# - 한글로 리포트 작성
# - 출처는 APA 형식으로 작성
# - 최종 결과는 .md 파일로 저장""",
#         recursive_limit=150,
#     )
    
#     # 마크다운 형식으로 최종 결과물 출력
#     if hasattr(output["messages"][-1], "content"):
#         display(Markdown(output["messages"][-1].content))
#     else:
#         display(Markdown(output["messages"][-1]))
    
#     print(output["messages"][-1])