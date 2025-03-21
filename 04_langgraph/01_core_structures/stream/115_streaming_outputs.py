##########################################################################################

### 단계별 스트리밍 출력 ###

# `stream()` 출력 함수에 대한 조금 더 자세한 설명

##########################################################################################
# 미리 정의해둔 기본 agent 그래프
from a_agent_graph import agent_graph, State

graph = agent_graph()

######################## StateGraph의 `stream` 메서드 ########################

# `stream()` 메서드는 단일 입력에 대한 그래프 단계를 스트리밍 하는 기능을 제공한다.

# `input` (Union[dict[str,Any], Any]): 그래프에 대한 입력
# `config` (Optional[RunnableConfig]) : 실행구성
# `stream_mode` (Optional[Union[StreamMode, list[StreamMode]]]) : 출력 스트리밍 모드
# `output_keys` (Optional[Union[str, Sequence[str]]]) : 스트리밍할 키
# `interrupt_before` (Optional[Union[All, Sequence[str]]]): 실행 `전` 중단할 모드
# `interrupt_after` (Optional[Union[All, Sequence[str]]]): 실행 `후` 중단할 모드
# `debug` (Optional[bool]) : 디버그 모드 여부
# `subgraph` (bool): 하위 그래프 스트리밍 여부

### 반환값
# Iterator[Union[dict[str, Any], Any]]: 그래프의 각 단계 출력 , 출력 형태는 `stream_mode`에 따라 달라짐

### 주요기능
# 1. 입력된 설정에 다라 그래프 실행을 스트리밍 방식으로 처리
# 2. 각 단계별로 출력되는 값의 형태는 `stream_mode`에 따라 달라짐(`values`, `updates`, `debug`)
# 3. 콜백 관리 및 오류 처리
# 4. 재귀 제한 미ㅏㅊ 중단 조건 처리

### 스트리밍 모드 
# `values` : 각 단계의 현재 상태 값 출력
# `updates` : 각 단계의 상태 업데이트만 출력
# `debug` : 각 단계의 디버그 이벤트 출력

from b_runnable_config import RunnableConfig
# from langchain_core.runnables import RunnableConfig
question = "2024년 최고의 이슈였던 뉴스를 알려주세요."

# 초기 입력상태 정의
input = State(dummy_data="테스트 문자열", messages=[("user", question)])

# config 설정
config = RunnableConfig(
    recursion_limit=10, # 최대 10개까지의 노드방문, 그이상은 RecursionError 발생.
    configurable={"thread_id": "1"},
    tags = ["my-tag"] # tag
)

# # config설정 이후 스트리밍 출력 진행
# for event in graph.stream(input=input, config=config):
#     for key, value in event.items():
#         print(f"\n[ {key} ]\n")
#         # value 에 messages 가 존재하는 경우
#         if "messages" in value:
#             messages = value["messages"]
#             # 가장 최근 메시지 1개만 출력합니다.
#             value["messages"][-1].pretty_print()


######################## `output_keys` 옵션 ########################

# `output_keys` 옵션은 스트리밍 키를 지정하는데 사용된다.

# list 형식으로 지정가능하며, channels에 정의된 키 중 하나 여야한다.

# !tip : 매 단계마다 출력되는 State key가 많은 경우, 일부만 스트리밍하고 싶은 경우에 유용 ! 

#channels 에 정의된 키 목록을 출력
print(list(graph.channels.keys()))
# 출력 확인
# ['messages', 'dummy_data', '__start__', 'chatbot', 'tools', 'branch:chatbot:__self__:chatbot', 
# 'branch:chatbot:__self__:tools', 'branch:tools:__self__:chatbot', 'branch:tools:__self__:tools', 
# 'start:chatbot', 'branch:chatbot:tools_condition:chatbot', 'branch:chatbot:tools_condition:tools']

for event in graph.stream(
    input=input,
    config=config,
    output_keys=["messages", "dummy_data"],  # messages 를 추가해 보세요!
):
    for key, value in event.items():
        # key 는 노드 이름
        print(f"\n[ {key} ]\n")

        # dummy_data 가 존재하는 경우
        if value:
            # value 는 노드의 출력값
            print(value.keys())
            # dummy_data key 가 존재하는 경우
            if "dummy_data" in value:
                print(value["dummy_data"])
            # messages key 가 존재하는 경우
            if "messages" in value:
                # 가장 최근 메시지 1개만 출력
                print(value["messages"])


######################## `stream_mode` 옵션 ########################

# `stream_mode` 옵션은 스트리밍 출력 모드를 지정하는데 사용된다.

# `values`: 각 단계의 현재 상태 값 출력
# `updates`: 각 단계의 상태 업데이트만 출력 (기본값) -> 상태값이 그대로라면 출력되지 않음. (깔끔한 출력 가능)

### `event.items()`
# `key`: State 의 key 값
# `value`: State 의 key 에 대한하는 value


####### stream_mode = "values" 는 아래와 같이 작성
# for event in graph.stream(
#     input=input,
#     stream_mode="values",  # 기본값
# ):

####### stream_mode = "updates" 는 아래와 같이 작성
# for event in graph.stream(
#     input=input,
#     stream_mode="updates",  # 기본값
# ):


########################## `interrupt_before`와 `interrupt_after` 옵션 ########################

# `interrupt_before`: 지정된 노드 이전에 스트리밍 중단
# `interrupt_after`: 지정된 노드 이후에 스트리밍 중단

for event in graph.stream(
    input=input,
    config=config,
    stream_mode="updates",
    # interrupt_before=["tools"]
    interrupt_after=["tools"]
):
    for key, value in event.items():
        print(f"\n[ {key} ]\n")

        # value 가 dict 형식인 경우 
        # value는 노드의 출력값
        if isinstance(value, dict):
            print(value.keys())
            if "messages"in value:
                print(value["messages"])

        if "messages" in value:
            # value에는 state가 dict 형태로 저장(values 의 key값)
            print(f"message 개수 : {len(value["messages"])}")
            
    print("==="*10, " 단계", "==="*10)