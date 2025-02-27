# Human in the Loop

# 사용자가 직접 개입하여 "승인" 또는 "거절" 결정을 내리는 경우
# Langgraph에서 지원하는 `human_in_the_loop` 워크플로우 형태
# `interrupt_before` 또는 `interrupt_after` 노드를 사용하여 사용자 개입 지점 설정

from a_agent_graph import create_human_in_the_loop_graph, State

####################### 7. interrup_before 추가 #######################

from langchain_teddynote.messages import pretty_print_messages
from langchain_core.runnables import RunnableConfig


graph = create_human_in_the_loop_graph()
# 질문
# question = "time-llm 논문과 관련된 코드를 찾아줘."
question = "AI관련 뉴스 찾아줘"

# 초기 입력 State 를 정의
input = State(messages=[("human", question)])

# config 설정
config = RunnableConfig(
    recursion_limit=10, # 최대 재귀 호출 횟수, 그 이상은 RecursionError 발생
    configurable={"thread_id": "1"},
    tags=["my-rag"], # 태그 설정
)

for event in graph.stream(
    input=input,
    config = config,
    stream_mode="values",
    interrupt_before=["tools"], # tools 실행 전 interrupt (tools 노드 실행전 중단)
):
    for key, value in event.items():
        # key는 노드이름
        print(f"\n[{key}]\n")

        # value는 노드의 출력값
        # print(value)
        pretty_print_messages(value)

        # value에는 state가 dict 형태로 저장(value의 key값)
        if "messages" in value:
            print(f"length of messages: {len(value['messages'])}")

### 그래프 상태를 확인하여 제대로 작동했는지 체크크 ###
# 그래프 상태 스냅샷 생성
snapshot = graph.get_state(config)

# 다음 스냅샷 상태
print(f"\n다음 스냅샷 상태: {snapshot.next}\n")


### 도구호출 확인 ###
from langchain_teddynote.messages import display_message_tree

# 메시지 스냅샷에서 마지막 메시지 추출
existing_message = snapshot.values["messages"][-1]

print(f"메세지 스냅샷에서 마지막 메시지 추출 : {existing_message}")
# 메시지 트리 표시
print(f"메시지 트리: {display_message_tree(existing_message.tool_calls)}\n")



### 종료지점부터 이어서 그래프 진행 ###
# 입력에 `None`을 전달하여 종료지점부터 그래프 진행

# `None`은 현재 상태에서 아무것도 추가하지 않음
events = graph.stream(None, config, stream_mode="values")

# 이벤트 반복처리
for event in events:
    # 메시지가 이벤트에 포함된 경우
    if "messages" in event :
        # 마지막 메시지의 예쁜 출력
        event["messages"][-1].pretty_print()


to_replay = None

# 상태 기록 가져오기
for state in graph.get_state_history(config):
    # 메시지 수 및 다음 상태 출력 
    print(f"메시지 수 : {len(state.values["messages"])}, 다음노드 : {state.next}")
    print("-"*60)
    # 특정 상태 선택 기준 : 채팅 메시지 수 
    if len(state.values["messages"]) ==3 :
        to_replay = state

### 원하는 지점은 `to_replay` 변수에 저장한다. 이를 활용하여 다시 시작할 수 있는 지점을 지정 할 수 있다.

# 다음 항목의 다음요소 출력
print(f"to_replay.next : {to_replay.next}")
# 다음 항목의 설정 정보 출력
print(f"to_replay.config : {to_replay.config}")

events = graph.stream(None, to_replay.config, stream_mode="values")

# 이벤트 반복처리
for event in events:
    # 메시지가 이벤트에 포함된 경우
    if "messages" in event :
        # 마지막 메시지의 예쁜 출력
        event["messages"][-1].pretty_print()

# print(f"to_replay.next : {to_replay.next}")
# # 다음 항목의 설정 정보 출력
# print(f"to_replay.config : {to_replay.config}")
