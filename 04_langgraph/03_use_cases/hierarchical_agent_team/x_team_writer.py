################################## Doc Writing Team 

import operator
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from pathlib import Path
from c_tools_writing import write_document, edit_document, read_document, python_repl_tool, create_outline, WORKING_DIRECTORY
from a_agent_factory import agent_factory, MODEL_NAME, llm, create_team_supervisor
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor

class DocWritingState(TypedDict):
    messages : Annotated[List[BaseMessage], operator.add]
    team_members : str
    next: str
    current_files : str # 현재 작업중인 파일

# 상태 전처리 노드: 각각의 에이전트가 현재 작업 디렉토리의 상태를 더 잘 인식할 수 있도록 함.
def preprocess(state):
    # 작성된 파일 목록 초기화
    written_files = []
    try:
        # 작업 디렉토리 내의 모든 파일을 검색하여 상대 경로로 변환
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except Exception :
        pass
    # 작성된 파일이 없으면 상태에 "Nofiles written" 메시지 추가
    if not written_files:
        return{**state, "current_files": "No files written."}

    strucrued_state =  {
        **state,
        "current_files": "\nBelow are files your team has written to the directory: \n"
        + "\n".join([f" - {f}" for f in written_files])
    }
    print(f"\n\nstrucrued_state: \n\n{strucrued_state}\n\n")
    return strucrued_state



# 라우팅할 노드를 선택하는 함수 정의
def get_next_node(x):
    return x["next"]

# 문서 작성 에이전트 생성
doc_writer_agent = create_react_agent(
    llm,
    # tools = [write_document, edit_document, read_document],
    tools = [read_document],
    state_modifier = "You are a arxiv researcher. Your mission is to write arxiv style paper on given topic/resources."
)
context_aware_doc_writer_agent = preprocess | doc_writer_agent
doc_writing_node = agent_factory.create_agent_node(
    context_aware_doc_writer_agent, name= "DocWriter"
)

# 아웃라인 작성 노드
note_taking_agent = create_react_agent(
    llm,
    tools = [create_outline, read_document],
    state_modifier = "You are an expert in creating outlines for research papers. Your mission "
    "is to create an outline for a given topic/resources or documents.",
    
)
context_aware_note_taking_agent = preprocess | note_taking_agent

print(f"\n\ncontext_aware_note_taking_agent: \n\n{context_aware_note_taking_agent}\n\n")

note_taking_node = agent_factory.create_agent_node(
    context_aware_note_taking_agent, name= "NoteTaker"
)

# 차트 생성 에이전트 생성성
chart_generating_agent = create_react_agent(
    llm,
    tools = [read_document, python_repl_tool]
)
context_aware_chart_generating_agent = preprocess | chart_generating_agent
chart_generating_node = agent_factory.create_agent_node(
    context_aware_chart_generating_agent, name= "ChartGenerator"
)


# 문서 작성 팀 감독자 생성
doc_writing_supervisor = create_team_supervisor(
    MODEL_NAME,
    "You are a supervisor tasked with managing a conversation between the"
    # " following workers:  ['DocWriter', 'NoteTaker', 'ChartGenerator']. Given the following user request,"
    " following workers:  ['DocWriter']. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    # ["DocWriter", "NoteTaker", "ChartGenerator"],
    ["DocWriter"],
)

#  그래프 생성
authoring_graph = StateGraph(DocWritingState)

# 노드 정의
authoring_graph.add_node("DocWriter", doc_writing_node)
# authoring_graph.add_node("NoteTaker", note_taking_node)
# authoring_graph.add_node("ChartGenerator", chart_generating_node)
authoring_graph.add_node("Supervisor", doc_writing_supervisor)

# 엣지 정의
authoring_graph.add_edge("DocWriter", "Supervisor")
# authoring_graph.add_edge("NoteTaker", "Supervisor")
# authoring_graph.add_edge("ChartGenerator", "Supervisor")

# 조건부 엣지 정의: Supervisor 노드의 결정에 따라 다음 노드로 이동
authoring_graph.add_conditional_edges(
    "Supervisor",
    get_next_node,
    {
        "DocWriter": "DocWriter",
        # "NoteTaker": "NoteTaker",
        # "ChartGenerator": "ChartGenerator",
        "FINISH": END,
    },
)

# 시작 노드 설정
authoring_graph.set_entry_point("Supervisor")

# 그래프 컴파일
authoring_app = authoring_graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    # ###### 그래프 시각화 
    from IPython.display import Image
    from x_team_researcher import run_graph
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.runnables import RunnableConfig
    img = Image(authoring_app.get_graph(xray=True).draw_mermaid_png())
    with open("team_writer.png", "wb") as f:
        f.write(img.data)
    
    
    output = run_graph(
    authoring_app,
    "Transformer 의 구조에 대해서 심층 파악해서 논문의 목차를 한글로 작성해줘. "
    "그 다음 각각의 목차에 대해서 5문장 이상 작성해줘. "
    "상세내용 작성시 만약 chart 가 필요하면 차트를 작성해줘. "
    "최종 결과를 저장해줘. ",
    )
    print(output["messages"][-1].content)