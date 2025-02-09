######################################################################

# Function Calling LLM과 도구호출 노드, Conditional Edge 구축방법 #

######################################################################

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv()

from langchain_teddynote.tools.tavily import TavilySearch
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
import json
from langchain_core.messages import ToolMessage
from langgraph.graph import START, END

# 검색 도구 생성
tool = TavilySearch(max_results=3)
# 도구 목록에 추가
tools = [tool]


# State 정의
class State(TypedDict):
    # list 타입에 add_messages 적용(list 에 message 추가)
    messages: Annotated[list, add_messages]



# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini")
# LLM 에 도구 바인딩
llm_with_tools = llm.bind_tools(tools)
# 노드 함수 정의
def chatbot(state: State):
    answer = llm_with_tools.invoke(state["messages"])
    # 메시지 목록 반환
    return {"messages": [answer]}  # 자동으로 add_messages 적용


# 상태 그래프 초기화
graph_builder = StateGraph(State)
# 노드 추가
graph_builder.add_node("chatbot", chatbot)


class BasicToolNode:
    """Run tools requested in the last AIMessage node"""

    def __init__(self, tools: list) -> None:
        # 도구 리스트
        self.tools_list = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # 메시지가 존재할 경우 가장 최근 메시지 1개 추출
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        # 도구 호출 결과 ***** (중요)
        outputs = []
        for tool_call in message.tool_calls: # tool_call : 도구 호출 정보
            # 도구 호출 후 결과 저장
            tool_result = self.tools_list[tool_call["name"]].invoke(tool_call["args"]) # name : 도구 이름 | arg : 검색 쿼리
            outputs.append(
                # 도구 호출 결과를 메시지로 저장
                ToolMessage(
                    content=json.dumps(
                        tool_result, ensure_ascii=False
                    ),  # 도구 호출 결과를 문자열로 변환
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}

# 도구 노드 생성
tool_node = BasicToolNode(tools=[tool])
# 그래프에 도구 노드 추가
graph_builder.add_node("tools", tool_node)



def route_tools(
    state: State,
):
    if messages := state.get("messages", []):
        # 가장 최근 AI 메시지 추출
        ai_message = messages[-1]
    else:
        # 입력 상태에 메시지가 없는 경우 예외 발생
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    # AI 메시지에 도구 호출이 있는 경우 "tools" 반환
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        # 도구 호출이 있는 경우 "tools" 반환
        return "tools"
    # 도구 호출이 없는 경우 "END" 반환
    return END


# `tools_condition` 함수는 챗봇이 도구 사용을 요청하면 "tools"를 반환하고, 직접 응답이 가능한 경우 "END"를 반환
graph_builder.add_conditional_edges(
    source="chatbot",
    path=route_tools,
    # route_tools 의 반환값이 "tools" 인 경우 "tools" 노드로, 그렇지 않으면 END 노드로 라우팅
    path_map={"tools": "tools", END: END},
)

# tools > chatbot
graph_builder.add_edge("tools", "chatbot")

# START > chatbot
graph_builder.add_edge(START, "chatbot")

# 그래프 컴파일
graph = graph_builder.compile()


if __name__ == "__main__":
    inputs = {"messages": "테디노트 YouTube 채널에 대해서 검색해 줘"}

    for event in graph.stream(inputs, stream_mode="values"):
        for key, value in event.items():
            print(f"\n==============\nSTEP: {key}\n==============\n")
            # display_message_tree(value["messages"][-1])
            print(value[-1].content)