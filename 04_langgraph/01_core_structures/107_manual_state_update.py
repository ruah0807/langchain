#### 중간단계의 State 수동 업데이트 ####

from a_agent_graph import create_human_in_the_loop_graph, State
from langchain_core.runnables import RunnableConfig

graph = create_human_in_the_loop_graph()


question = "LangGraph 가 무엇인지 조사하여 알려주세요!"

# 초기 입력 상태를 정의
input = State(messages=[("user", question)])

# config 설정
config = RunnableConfig(
    configurable={"thread_id": "1"},  # 스레드 ID 설정
)

#  채널목록을 출력하여 interrupt_before 와 interrupt_after 를 적용할 수 있는 목록을 출력
print(f"graph.channels: {list(graph.channels)}\n")

# 그래프 스트림 호출
events = graph.stream(
    input, config, interrupt_before=["tools"], stream_mode="values"
)

for event in events:
    # 메시지가 이벤트에 포함된 경우
    if "messages" in event:
        #마지막 메시지의 예쁜 출력
        event["messages"][-1].pretty_print()

# 위 단계는 `toolNode`에 의해 중단 됨.
# 가장 최근 메시지를 확인하면 ToolNode 가 검색을 수행하기 전 query 를 포함하고 있음을 알 수 있다.
# query 가 단순하게 LangGraph 라는 단어만을 포함

# 그래프 상태 스냅샷 생성
snapshot = graph.get_state(config)

#가장 최근 메시지 추출
last_message = snapshot.values["messages"][-1]

#메시지 출력
last_message.pretty_print()


###### 사람의 개입 (Human in the Loop) ######

# TavilySearch 도구에서 검색 결과 를 수정
#  ToolMessage 의 결과가 마음에 들지 않는 경우 
# 사람이 중간에 개입하여 웹 검색 도구인 Tavily Tool 의 검색 결과인 ToolMessage 를 수정하여 LLM 에게 전달 하는 방법



# https://raw.githubusercontent.com/teddylee777/langchain-kr/1999da031d689326fc7db9596b4a29b10076e290/17-LangGraph/01-Core-Features/image/langgraph-01.png

# 수정한 가상의 웹 검색 결과(중간 개입을 위해 만듬)
modified_search_results = """\n[수정된 웹 검색 결과]
LanggrPh는 상태 기반의 다중 액터 애플리케이션을 LLM을 활용해 구축할 수 있도록 지원합니다.
LangGraph는 사이클 흐름, 제어 가능성, 지속성, 클라우드 배포 기능을 제공하는 오픈 소스 라이브러리입니다.

자세한 튜토리얼은 [LangGraph 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/) 과
테디노트의 [랭체인 한국어 튜토리얼](https://wikidocs.net/233785) 을 참고하세요.\n"""

print(modified_search_results)


# 수정한 검색 결과를 `ToolMessage`에 주입

# ** 중요 ** 
# 메시지를 수정하려면 수정하고자 하는 Message와 일치하는 `tool_call_id`를 지정해야한다.


#수정하고자하는 `ToolMessage`의 `tool_call_id` 추출
tool_call_id = last_message.tool_calls[0]["id"]
print(f"tool_call_id: {tool_call_id}\n")

# tool_call_id: call_p3os8QpY5cRiIOCD8Fj6579z

from langchain_core.messages import AIMessage, ToolMessage

new_messages = [
    #LLM API의 도구 호출과 일치하는 ToolMessage 필요
    ToolMessage(
        content = modified_search_results,
        tool_call_id = tool_call_id,
    ),
    # LLM의 응답에 직접적으로 내용추가
    # AIMessage(content=modified_search_results)
]

new_messages[-1].pretty_print()


################ StateGraph의 `update_state` 메서드 사용 ################
# 메시지 수정 이후, 상태값을 업데이트 하지 않는다면 메시지 수정에도 불구하고 갱신이 제대로 일어나지 않는다. 
# 때문에, update_state 메서드를 사용하여 상태값을 업데이트 한다.
# update_state 메서드는 주어진 값으로 그래프의 상태를 업데이트한다. 이 메서든는 마치 `as_node`에서 값이 온 것처럼 동작.

# 매개변수 :
# `config` (RunnableConfig): 실행 구성
# `values` (Optional[Union[dict[str, Any], Any]]): 업데이트할 값들
# `as_node` (Optional[str]): 값의 출처로 간주할 노드 이름. 기본값은 None(어떤 노드로부터 나온 상태값을 업데이트할 것인지)

# 반환값 :
# `RunnableConfig`

# 주요 기능 :
# 체크포인터를 통해 이전 상태를 로드하고 새로운 상태를 저장
# 서브그래프에 대한 상태 업데이트를 처리
# `as_node`가 지정되지 않은 경우, 마지막으로 상태를 업데이트한 노드를 찾음.
# 지정된 노드의 writer들을 실행하여 상태를 업데이트
# 업데이트된 상태를 체크포인트에 저장

# 주요 로직 :
# 체크포인터를 확인하고, 없으면 ValueError를 발생시킴.
# 서브그래프에 대한 업데이트인 경우, 해당 서브그래프의 update_state 메서드를 호출함.
# 이전 체크포인트를 로드하고, 필요한 경우 as_node를 결정함.
# 지정된 노드의 writer 사용하여 상태를 업데이트함.
# 업데이트된 상태를 새로운 체크포인트로 저장함.

# 참고 :
# 이 메서드는 그래프의 상태를 **수동으로 업데이트**할 때 사용됨.
# 체크포인터를 사용하여 상태의 버전 관리와 지속성을 보장.
# as_node를 지정하지 않으면 자동으로 결정되지만, 모호한 경우 오류가 발생할 수 있음.
# 상태 업데이트 중 SharedValues에 쓰기 작업은 허용되지 않음.

graph.update_state(
    config, 
    # 제공할 업데이트된 값 : `State`의 메시지는 "추가전용"으로 기존 상태에 추가됨.
    {"messages": new_messages},
    as_node="tools"            
)

print("(최근 1개의 메시지 출력)\n")
graph.get_state(config).values["messages"][-1].pretty_print()


snapshot = graph.get_state(config)
print(f"snapshot.next_node: {snapshot.next}\n")

# ('chatbot', ) 이라 출력됨. 

# `None`은 현재 상태에 아무것도 추가하지 않음
events = graph.stream(None, config, stream_mode="values")

for event in events :
    #메시지가 이벤트에 포함된 경우
    if "messages" in event : 
        # 마지막 메시지의 예쁜 출력
        event["messages"][-1].pretty_print()


# 메세지 전체 하나씩 출력
# for message in snapshot.values["messages"]:
#     message.pretty_print()