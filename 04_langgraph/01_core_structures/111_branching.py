##########################################################################################

### Branching: 병렬 노드를 실행하기 위한 분기 생성방법 ###

# 노드의 병렬 실행은 전체 그래프 작업으 속도를 향상시키는데 필수적이다. 
# `LangGraph`는 노드의 병렬 실행을 지원하며, 이는 workflow의 성능을 향상시킨다.

# 병렬화 : `fan_out` & `fan_in` 메커니즘- 작업을 나누고 모으는 과정을 설명하는 개념념 
# 표준 엣지와 `conditional_edges`를 활용하여 병렬 노드 실행
##########################################################################################

from dotenv import load_dotenv

load_dotenv()

########################### 병렬 노드 fan-out 및 fan-in ###########################

##### fan-out : 큰 작업을 여러개의 작은 작업으로 나누는 과정
    # 예 ) 피자를 만들때 도우, 소스, 치즈 준비를 각각 별도로 수행
##### fan-in : 나뉜 작업들을 다시 하나로 합치는 과정
    # 예 ) 준비된 재료들을 모두 올려 완성된 피자를 만드는 과정. 

# State에서는 `reducer(add)` 연산자를 지정한다. 
    # State 내 특정 키의 기존 값을 단순히 덮어쓰는 대신 값들을 결합하거나 누적.
    # List의 경우 새로운 리스트를 기존 리스트와 연결하는것을 의미.

# LangGraph 는 State의 특정 키에 대한 reducer 함수를 지정하기 위해 `Annotated`타입을 사용.
    # `Annotated` : 타입 검사를 위해 원래타입 (list)를 유지하면서, 타입 자체를 변경하지 않고 reducer함수(add)를 타입에 첨부 가능하도록 함.


from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 상태 정의 
class MessagesState(TypedDict):
    aggregate: Annotated[list, add_messages]
    which: str

# 노드값 반환 클래스
class ReturnNodeValue:
    #초기화
    def __init__(self, node_secret: str):
        self._value = node_secret

    #호출시 상태 업데이트
    def __call__(self, state: MessagesState)-> Any:
        print(f"Adding {self._value} to {state['aggregate']}")
        return{"aggregate":[self._value]}
    

builder = StateGraph(MessagesState)

builder.add_node("node_1", ReturnNodeValue("I am A"))
builder.add_node("node_2", ReturnNodeValue("I am B"))
builder.add_node("node_3", ReturnNodeValue("I am C"))
builder.add_node("node_4", ReturnNodeValue("I am D"))
builder.add_node("node_5", ReturnNodeValue("I am E"))

builder.add_edge(START, "node_1")

# 상태의 'which'값에 따른 조건부 라우팅 경로 결정함수
def route_bc_or_cd(state: MessagesState)-> Sequence[str]:
    if state["which"] == "cd":
        return ["node_3", "node_4"]
    return ["node_2", "node_3"]

# 전체 병렬 처리할 노드 목록
intermediates = ["node_2", "node_3", "node_4"]

builder.add_conditional_edges(
    "node_1",
    route_bc_or_cd,
    intermediates
)
for node in intermediates:
    builder.add_edge(node, "e")

# 최종 노드 연결 및 그래프 컴파일
builder.add_edge("e", END)


# builder.add_edge("node_1", "node_2")
# builder.add_edge("node_1", "node_3")
# builder.add_edge("node_2", "node_4")
# builder.add_edge("node_3", "node_4")
# builder.add_edge("node_4", END)

graph = builder.compile()

# from IPython.display import Image
# img = Image(graph.get_graph(xray=True).draw_mermaid_png())
# with open("branching_graph.png", "wb") as f:
#     f.write(img.data)

#################################################

# 그래프 실행
# graph.invoke({"aggregate": []})

# 출력 결과
# AddingI am A to []
# AddingI am B to [HumanMessage(content='I am A', additional_kwargs={}, response_metadata={}, id='6b8013f9-99c6-4c13-bf29-8d251945d102')]
# AddingI am C to [HumanMessage(content='I am A', additional_kwargs={}, response_metadata={}, id='6b8013f9-99c6-4c13-bf29-8d251945d102')]
# AddingI am D to [HumanMessage(content='I am A', additional_kwargs={}, response_metadata={}, id='6b8013f9-99c6-4c13-bf29-8d251945d102'), HumanMessage(content='I am B', additional_kwargs={}, response_metadata={}, id='85646522-bede-4898-ba1f-9b92f7f46726'), HumanMessage(content='I am C', additional_kwargs={}, response_metadata={}, id='83de9308-9680-420e-be57-792b6a7499d4')]


########################### 조건부 분기(conditional branching) ###########################

# fan-out이 결정적이지 않은 경우, `add_conditional_edges`를 직접 사용할수 있다.

# 조건문 분기 이후 연결될 알려진 "sink"노드가 있는 경우, 조건부 엣지를 생성할 때 `then="실행할 노드명"`을 제공.
