##########################################################################################

### ToolNode 를 사용하여 도구를 호출하는 방법 ###

# 도구호출을 위한 LangGraph의 사전 구축된 `pre-built`의 `ToolNode` 사용방법

# 사전 구축된 Agent와 즉시 사용할수 있도록 설계됨.
# 상태에 적절한 리듀서가 있는 messages키가 포함된 경우 모든 `StateGraph`와 함게 작동 가능

##########################################################################################

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, Dict
load_dotenv()

@tool 
def search_news(query: str) -> List[Dict[str,str]]:
    """Search Web by input keyword"""
    tavily_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
        include_domains=["news.google.com"]
    )
    return tavily_tool.invoke({"query":query})

@tool
def python_code_interpreter(code:str):
    """Call to execute python code"""
    return PythonAstREPLTool().invoke(code)

########`ToolNode`를 사용하여 도구를 호출하는 방법 #########

# 도구 리스트 생성
tools = [search_news, python_code_interpreter]

# ToolNode 초기화
tool_node = ToolNode(tools)

######################### ToolNode 수동으로 호출하기 #########################

# ToolNode는 메시지 목록과 함께 그래프 상태에서 작동한다.

## ** 중요 : 마지막 메시지는 `tool_calls` 속성을 포함하는 `AIMessage`여야함.

# from langchain_core.messages import AIMessage

# # 단일 도구 호출을 포함하는 AI 메시지 객체 생성
# # AI Message객체여야함
# message_with_single_tool_call = AIMessage(
#     content="",
#     tool_calls = [
#         {
#             "name" : "search_news", # 도구 이름
#             "args" : {"query": "AI 에이전트"}, # 도구 인자
#             "id": "tool_call_id", # 도구 호출 ID
#             "type": "tool_call" # 도구 호출 유형 
#         },
#         {
#             "name": "python_code_interpreter",
#             "args":{"code": "print(1+2+3+4)"},
#             "id": "tool_call_id",
#             "type": "tool_call"
#         }
#     ]
# )

# # 도구 노드를 통한 메시지 처리 및 날씨 정보 요청 실행
# # 생성된 메시지를 도구 노드에 전달하여 다중 도구 호출 실행
# invoke = tool_node.invoke({"messages": [message_with_single_tool_call]})

# print(invoke)

######################### ToolNode 자동으로 호출하기(LLM 호출) #########################

# 도구 호출 기능이 있는 채팅 모델을 사용하기 위해, 모델이 사용가능한 도구들은 인식하도록 해야한다.
# `ChatOpenAI` 모델에서 `.bind_tools` 메서드를 호출하여 수행한다.

from langchain_openai import ChatOpenAI

# LLM 모델 초기화 및 도구 바인딩
model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature = 0).bind_tools(tools)

# 도구 호출 확인
check_tool_call = model_with_tools.invoke("처음 5개의 수소를 출력하는 python code를 작성해줘.").tool_calls

print(check_tool_call)

# 출력 결과 
# [
#   {
#       'name': 'python_code_interpreter', 
#       'args': {'code': "# 처음 5개의 수소 원자 번호와 기호를 출력하는 코드\nhydrogen_atoms = [(1, 'H')] * 5\n\n# 출력\nfor atom in hydrogen_atoms:\n    print(f'원자 번호: {atom[0]}, 기호: {atom[1]}')"}, 
#       'id': 'call_02PTKxrz7KuuKDlFTrvKj6aX', 
#       'type': 'tool_call'
#   }
# ]
# 이전에 수동으로 호출한 결과와 동일한 결과를 얻을 수 있다.

#채팅모델이 생성한  AI 메시지에는 이미 `tool_calls` 속성이 포함되어 있으므로, 
# 이를 `toolnode`에 직접 전달할 수 있다.

# 도구 노드를 통한 메시지 처리 및 LLM 모델의 도구 기반 응답 생성
tool_node_invoke = tool_node.invoke({
        "messages": [
            model_with_tools.invoke("처음 5개의 수소를 출력하는 python code를 작성해줘.")
        ]
    })


print(tool_node_invoke)

# 출력 결과
# {'messages': [ToolMessage(content='[1, 2, 3, 4, 5]\n', name='python_code_interpreter', tool_call_id='call_h7WDIN2jQsRGJ4wVgNd3WoRu')]}

######################### Agent 와 함께 사용하기 #########################

# LangGraph 그래프 내에서 ToolNode 를 사용하는 방법

# Agent의 그래프 구현 설정
#    Agent는 쿼리를 입력받아, 쿼리를 해결하는데 필요한 충분한 정보를 얻을 때까지 반복적으로 도구를 호출한다.

from langgraph.graph import StateGraph, MessagesState, START, END

#MessagesState : LangGraph의 내장 상태 타입

# LLM 모델을 사용하여 메시지 처리 및 응답 생성. 도구 호출이 포함된 응답 반환
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return{"messages": [response]}



# 메시지 상태 기반 워크 플로우 그래프 초기화
workflow = StateGraph(MessagesState)

# 에이전트와 도구 노드 정의 및 워크 플로우 그래프에 추가
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

# 에이전트 노드에서 조건부 분기문설정. 도구 노드 또는 종료 지점으로 연결
workflow.add_conditional_edges("agent", tools_condition)

# 도구 노드에서 에이전트 노드로 순환 연결
workflow.add_edge("tools", "agent")

# 에이전트 노드에서 종료 지점으로 연결
workflow.add_edge("agent", END)

# 정의된 워크 플로우 그래프 컴파일 및 실행가능한 어플리케이션 생성
app = workflow.compile()

# 실행 결과 확인
for chunk in app.stream(
    # {"messages":[("human", "처음 5개의 수소를 출력하는 python code를 작성해줘")]}, ### python_code_interpreter 호출
    # {"messages":[("human", "search google news about AI Agent")]}, ### search_news 호출
    {"messages":[("human", "안녕? 반가워 난 피카츄라고 해. ")]}, ### 도구호출 없이 바로 llm응답.
    stream_mode="values"
):
    # 마지막메시지
    chunk["messages"][-1].pretty_print()


