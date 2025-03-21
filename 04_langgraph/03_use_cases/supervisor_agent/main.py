from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-4o"


######################## Define State ########################

import operator
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next : str # 다음으로 라우팅할 에이전트 


######################## Create Agent ########################

##### 도구 (Tool) 생성
# 검색엔진을 사용하여 웹 조사를 수행하는 에이전트와 플롯을 생성하는 에이전트 생성.
    # Research : `TavilySearch`도구를 사용하여 웹 조사를 수행
    # Coder: `PythonREPLTool` 도구를 사용하여 코드 실행

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

# 최대 5개의 검색 결과
tavily_tool = TavilySearchResults(max_results=5)

# 로컬에서 코드를 실행하는 Python REPL 도구 초기화
python_repl_tool = PythonREPLTool()



######################## Agent생성하는 Utility 구현 ########################

# LangGraph를 사용하여 다중 에이전트 시스템을 구축할 때, 도우미함수는 에이전트 노드를
# 생성하고 관리하는데 중요한 역할을 한다. 이런 함수는 코드의 재사용성을 높이고, 에이전트 간의 상호작용을 간소화한다.

    # 에이전트 노드 생성 : 각 에이전트의 역할에 맞는 노드를 생성하기 위한 함수 정의
    # 작업 흐름 관리 : 에이전트 간의 작업 흐름을 조정하고 최적화하는 유틸리티 제공
    # 에러처리 : 에이전트 실행 중 발생할 수 있는 오류를 효율적으로 처리하는 메커니즘 포함

# agent_node 함수 정의 예시
    # 주어진 상태와 에이전트를 사용하여 에이전트 노드를 생성.
    # functools.partial 을 사용하여 호출

from langchain_core.messages import HumanMessage

# 지정한 agent와 name을 사용하여 agent 노드 생성
def agent_node(state, agent, name):
    # agent 호출
    agent_response = agent.invoke(state)
    # agent의 마지막 메시지를 HumanMessage로 변환하여 반환
    return {
        "messages":[
            HumanMessage(content=agent_response["messages"][-1].content, name = name)
        ]
    }


### functools.partial의 역할

# 기존 함수의 일부 인자 또는 키워드 인자를 미리 고정하여 새 함수를 생성하는데 사용. 

# 즉, 자주 사용하는 함수 호출 패턴을 간소화 할수 있도록 함.

    # 1. 미리 정의된 값으로 새 함수 생성: 기존 함수의 일부 인자를 미리 지정해서 새 함수 반환
    # 2. 코드 간결화 : 자주 사용하는 함수 호출 패턴을 단순화하여 코드 중복을 줄임
    # 3. 가독성 향상 : 특정 작업에 맞춰 함수의 동작을 맞춤화해 더 직관저긍로 사용가능
    
    # 예시
    # `research_node = functools.partial(agent_node, agent=research_agent, names="Researcher")`

    # 1. `agent_node` 라는 기존 함수가 있다고 가정했을때,
        # 이 함수는 여러개의 인자와 키워드 인자를 받을 수 있다.
    # 2. `functools.partial` 은 이 함수에 `agent=research_agent` 와 `names="Researcher"` 라는 값을 고정한다.
        # 즉, `research_node` 는 `agent_node` 를 호출할 때 agent와 names값을 따로 지정하지 않아도 된다. 
        # 예 )
        # `agent_node(state, agent=research_agent, names="Researcher")` 
        # 대신,
        # `research_node(state)` 만 호출하면 된다.

# 코드 예시
import functools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# research_agent 생성
research_agent= create_react_agent(ChatOpenAI(model="gpt-4o"), tools=[tavily_tool])

# research node 생성
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# result = research_node(
#     {"messages": [HumanMessage(content="Code hello world and print it to the terminal")]}
# )

# print(result)

# 출력 결과
# {'messages': [HumanMessage(content='The code to print "Hello, World!" to the terminal is:\n\n```python\nprint(\'Hello, World!\')\n```\n\nWhen executed, it outputs:\n\n```\nHello, World!\n```', additional_kwargs={}, response_metadata={}, name='Researcher')]}




###################### Create Supervisor Agent ########################

# 에이전트들을 관리감독하는 감독자 에이전트 생성

from pydantic import BaseModel 
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 멤버 Agent 목록 정의
members = ["Researcher", "Coder"]

# 다음 작업자 선택 옵션 목록 정의
options_for_next = ["FINISH"] + members

# 작업자 선택 응답 모델 정의 : 다음작업자를 선택하거나 작업 완료를 나타냄

class RouteResponse(BaseModel):
                # Literal["Researcher", "Coder", "FINISH"]와 동일함.
    next: Literal[*options_for_next] # type: ignore 

# 시스템 프롬프트 정의 : 작업자 간의 대화를 관리하는 감독자 역할
system_prompt = """
You are asupervisor tasked with managing a conversation between the following workers: '{members}'. 
Given the following user request, respond with worker to act next. Each worker will perform a task andrespond with their results and status. 
When finished, respond with 'FINISH'."
"""

# ChatPromptTemplate 생성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            "Or should be FINISH? Select one of: {options}",
        )
    ]
).partial(options=str(options_for_next), members=", ".join(members))

llm= ChatOpenAI(model=MODEL_NAME, temperature=0)

# Supervisor Agent 생성
def supervisor_agent(state):
    # 프롬프트와 LLM을 결합하여 체인구성
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    # agent 호출
    return supervisor_chain.invoke(state)


###################### Create Workflow ########################

import functools
from langgraph.prebuilt import create_react_agent

# Research Agent 생성
research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Resaercher")


# Code Agent 생성
code_system_prompt = """
Be sure to use the following font in your code for visualization.

##### setting font #####
import platform
# OS jedge
current_os = platform.system()

if current_os == "Windows":
    # Windows 환경 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
    fontprop = fm.FontProperties(fname=font_path, size=12)
    plt.rc("font", family=fontprop.get_name())
elif current_os == "Darwin":  # macOS
    # Mac 환경 폰트 설정
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux 등 기타 OS
    # 기본 한글 폰트 설정 시도
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except:
        print("한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")

##### 마이너스 폰트 깨짐 방지 #####
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 폰트 깨짐 방지
"""
coder_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    state_modifier=code_system_prompt
)

coder_node = functools.partial(agent_node, agent=coder_agent, name="Coder")


from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("Supervisor", supervisor_agent)


workflow.add_edge(START, "Supervisor")

# member node -> Supervisor node로 엣지 추가
for member in members :
    workflow.add_edge(member, "Supervisor")

# add conditional edge
conditional_map = {
    k : k 
    for k in members
}
conditional_map["FINISH"] = END

# 조건부 엣지에 필요한 함수
def get_next(state):
    return state["next"]

# supervisor 노드에ㅐ서 조건부 엣지 추가
workflow.add_conditional_edges(
    "Supervisor",
    get_next,
    conditional_map
)

graph = workflow.compile(checkpointer=MemorySaver())


# from IPython.display import Image

# graph_image = graph.get_graph(xray=True).draw_mermaid_png()
# img = Image(graph_image)
# with open("supervisor_agent_graph.png", "wb") as f:
#     f.write(img.data)




from langchain_core.runnables import RunnableConfig
import uuid

config = RunnableConfig(recursion_limit=10, configurable={"thread_id": uuid.uuid4()})

# 질문
inputs = {"messages": [HumanMessage(content="2010 ~ 2024년까지의 대한민국의 1인당 GDP 추이를 그래프로 시각화 해주세요.")]}

for chunk in graph.stream(inputs, config, stream_mode="updates"):
    for k, v in chunk.items():
        print(f"=================== {k} : \n")
        if "messages" in v:
            v["messages"][-1].pretty_print()
        
