################ Inturrupt 이 후 메시지 상태 업데이트 진행 - 이어서 ################

# TavilySearch 도구에서 검색 쿼리 수정
#
# 다음노드로 진행하기 전 interrupt 를 발생시켜 중단하고 상태State를 갱신한 뒤 이어서 진행하는 방법.
# 
# 먼저, 새로운 thread_id를 생성.

from a_agent_graph import agent_graph, State, generate_random_hash
from langchain_core.messages import AIMessage

graph = agent_graph()

thread_id = generate_random_hash() # 랜덤한 해시값을 생성하는 generate_random_hash 함수

question = "LangGrahp에 대해 배워보고 싶어요. 유용한 자료를 추천해 주세요."

# 초기 입력 상태 정의
input = State(messages=[("user", question)])

# 새로운 config 생성
config = {"configurable": {"thread_id": thread_id}}


events = graph.stream(
    input =input,
    config= config,
    interrupt_before=["tools"],
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

print("thread_id: ", thread_id)
# thread_id: ee17fa
config_copy = config.copy()

# '스냅샷' 상태 가져오기
snapshot = graph.get_state(config)

# messages 의 마지막 메시지 가져오기
existing_message = snapshot.values["messages"][-1]

print("Message ID : ", existing_message.id)
# 출력 
# Message ID :  run-2cec82ed-bf24-41b8-a696-8f3f5826db0d-0


# 마지막 메시지는 tavily_web_search 도구 호출과 관련된 메시지임.
# 주요 속성은 다음과 같다.
    # name: 도구의 이름
    # args: 검색 쿼리
    # id: 도구 호출 ID
    # type: 도구 호출 유형(tool_call)
print(existing_message.tool_calls[0])
# 출력
# {'name': 'search_news', 'args': {'query': 'LangGraph'}, 'id': 'call_bj3lQMdNF2W8UXHKnMEtKvH8', 'type': 'tool_call'}

# 해당 속성 값중 `args`의 `query`를 업데이트 해보자.
# 기존의 exeisting_message를 복사하여 새로운 도구인 new_tool_call을 생성
# copy() 메서드를 사용하여 복사했기 때문에 모든 속성값이 복사된다.
# 이 후, query 매개변수에 원하는 검색쿼리를 입력한다.

# 중요  ! : id는 기존 메시지의 id를 그대로 사용한다. (id가 달라지면 message reducer가 동작하여 메시지 갱신이 불가.)

# new_tool_calls 를 복사하여 새로운 도구 호출 생성
new_tool_call = existing_message.tool_calls[0].copy()

# 쿼리 매개 변수 업데이트(갱신)
new_tool_call["args"] = {"query" : "LangGraph site: teddylee777.github.io"}
print(new_tool_call)

# 변경된 쿼리
# {'name': 'search_news', 'args': {'query': 'LangGraph site: teddylee777.github.io'}, 'id': 'call_rEefq3qOj2kmRkBpEOajENNO', 'type': 'tool_call'}

# AIMessage 생성
new_message = AIMessage(
    content = existing_message.content,
    tool_calls=[new_tool_call],
    # 중요 ! ID는 메시지를 상태에서 추가하는 대신 교체하는 방법
    id = existing_message.id
)

print(new_message.id)

new_message.pretty_print()

# 출력 결과 같은 id를 가진 쿼리가 갱신됨.
# run-86cedf4c-a34d-41ec-891c-72b5548d0246-0
# ================================== Ai Message ==================================
# Tool Calls:
#   search_news (call_AYzNw5kkIfnx8iZZ7vrvekmZ)
#  Call ID: call_AYzNw5kkIfnx8iZZ7vrvekmZ
#   Args:
#     query: LangGraph site: teddylee777.github.io

# 업데이트된 도구 호출 출력
print(new_message.tool_calls[0])
# {'name': 'search_news', 'args': {'query': 'LangGraph site: teddylee777.github.io'}, 'id': 'call_OK1XbsuwsQRVXyMvoHcDlWCa', 'type': 'tool_call'}

# message id 출력
print("\nMessage ID : ", new_message.id)
# Message ID :  run-86cedf4c-a34d-41ec-891c-72b5548d0246-0

# 상태 업데이트를 통해 
graph.update_state(config,{"messages": [new_message]})