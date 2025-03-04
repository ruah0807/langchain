###############################################################################

############ 대화 기록 요약을 추가하는 방법 ###############

# 대화가 길어질수록 대화기록이 누적되어 context window를 더 많이 차지하게 된다. 
# 이는 토큰 수도 길어지고(비용증가), 잠재적인 오류 발생 가능성을 야기한다.
# 이를 위해 대화 기록 요약본을 생성하고, 이를 최근 N개의 메시지와 함게 사용 해야한다.

### Step 1 : 대화가 너무 긴지 확인(메시지 수나 길이로 확인)
### Step 2 : 너무 길다면 요약본 생성(이를 위한 프롬프트 필요)
### Step 3 : 마지막 N개의 메시지를 제외한 나머지 삭제

# 중요한 부분은 오래된 메시지를 삭제 `DeleteMessage`하는 것이다.

###############################################################################

from dotenv import load_dotenv

load_dotenv()


########################## 긴 대화를 요약하여 대화로 저장 #################################
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages


# memory 저장소 설정
memory = MemorySaver()

# 메시지 상태와 요약 정보를 포함하는 상태 클래스
class State(TypedDict):
    messages : Annotated[list, add_messages]
    summary : str

# 대화 및 요약을 위한 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# `ask_llm`노드는 messages를 LLM에 주입하여 답변을 얻는다
# 만약, 이전대화 요약본이 존재한다면, 이를 시스템 메시지로 추가하여 대화에 포함 시켜야함.
# But, 이전대화 요약본이 존재하지 않는다면, 이전의 대화 내용만 사용

def ask_llm(state: State):
    # 이전 요약 정보 확인
    summary = state.get("summary", "")

    # 이전 요약 정보가 있다면 시스템 메시지로 추가
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        # 시스템 메시지와 이전 메시지 결합
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else : 
        # 이전 메시지만 사용
        messages = state["messages"]

    # 모델 호출
    response = model.invoke(messages)

    # 응답 반환
    return {"messages": [response]}

# `should_continue` 노드는 대화의 길이가 6개 초과일 경우 요약 노드로 이동한다.
# 그렇지 않다면, 즉각 답변을 반환한다.(END 노드로 이동)

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    # 메시지 목록 확인
    # 메시지 목록 확인
    messages = state["messages"]

    # 메시지 수가 6개 초과인 경우 요약 노드로 이동
    if len(messages) > 6:
        return "summarize_conversation"
    return END

# `summarize_conversation` 노드는 대화를 요약하고, 오래된 메시지를 '삭제'한다.

def summarize_conversation(state: State):
    # 이전 요약 정보 확인
    summary = state.get("summary", "")
    # 이전 요약정보가 있다면 요약 메시지 생성
    if summary:
        summary_message = (
            f" This is summary of the conversation to date:{summary}\n\n"
            "Extend the summary by taking into account the new messages above in Korean : "
        )
    else:
        # 요약 메시지 생성
        summary_message = "Create a summary of the conversation above in Korean:"

    # 요약 메시지와 이전 메시지 결합
    messages = state["messages"] + [HumanMessage(content=summary_message)]

    # 모델 호출
    response = model.invoke(messages)

    # 오래된 메시지 삭제
    deleted_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    # 요약 정보 반환
    return{"summary": response.content, "messages":deleted_messages}

# 워크플로우 그래프 초기화
workflow = StateGraph(State)

# 대화 및 요약 노드 추가
workflow.add_node(ask_llm)
workflow.add_node(summarize_conversation)

# 시작점을 대화 노드로 설정
workflow.add_edge(START, "ask_llm")

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "ask_llm",
    should_continue,
)

# 요약 노드에서 종료 노드로의 엣지 추가
workflow.add_edge("summarize_conversation", END)

# 워크플로우 컴파일 및 메모리 체크포인터 설정
app = workflow.compile(checkpointer=memory)

# from IPython.display import Image
# img = Image(app.get_graph(xray=True).draw_mermaid_png())
# print(img)
# with open("summary_graph.png", "wb") as f:
#     f.write(img.data)

# 업데이트 정보 출력 함수
def print_update(update):
    # 업데이트 딕셔너리 순회
    for k, v in update.items():
        # 메시지 목록 출력
        for m in v["messages"]:
            m.pretty_print()
        # 요약 정보 존재 시 출력
        if "summary" in v:
            print(v["summary"])


# 메시지 핸들링을 위한 HumanMessage 클래스 임포트
from langchain_core.messages import HumanMessage

# 스레드 ID가 포함된 설정 객체 초기화
config = {"configurable": {"thread_id": "1"}}

# 첫 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="안녕하세요? 반갑습니다. 제 이름은 김루아입니다.")
input_message.pretty_print()

# 스트림 모드에서 첫 번째 메시지 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 두 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="제 이름이 뭔지 기억하세요?")
input_message.pretty_print()

# 스트림 모드에서 두 번째 메시지 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 세 번째 사용자 메시지 생성 및 출력
input_message = HumanMessage(content="제 직업은 AI 연구원이에요")
input_message.pretty_print()

# 스트림 모드에서 세 번째 메시지 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

    # 상태 구성 값 검색
# values = app.get_state(config).values
# print(values)

# 사용자 입력 메시지 객체 생성
input_message = HumanMessage(
    content="최근 LLM 에 대해 좀 더 알아보고 있어요. LLM 에 대한 최근 논문을 읽고 있습니다."
)

# 메시지 내용 출력
input_message.pretty_print()

# 스트림 이벤트 실시간 처리 및 업데이트 출력
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

##### 상태 구성 값 검색
values = app.get_state(config).values
print(values)

# 사용자 메시지 객체 생성
input_message = HumanMessage(content="제 이름이 무엇인지 기억하세요?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)


######################################################### 그외 테스트
# 8번째 메시지
input_message = HumanMessage(content="전 AI 에이전트를 요즘 구현하고 있어요")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 9번째 메시지
input_message = HumanMessage(content="고민이 많아요.")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 10번째 메시지
input_message = HumanMessage(content="Langgraph를 이용해서 에이전트 껍데기를 구현해야해서.")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 11번째 메시지
input_message = HumanMessage(content="저희 수석님이 압박을 많이 주세요.")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 12번째 메시지
input_message = HumanMessage(content="공부는 해도해도 모자라요.")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

# 13번째 메시지
input_message = HumanMessage(content="그냥 집에서 자고 싶어요.")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

##### 상태 구성 값 검색
values = app.get_state(config).values
print(values)


