##########################################################################################

### 서브그래프 추가 및 사용법 ###

# SubGraph 의 일반적인 사용 사례는 멀티 에이전트 시스템 구축

### SubGraph 를 추가할 때 주요 고려사항 :
# - 상위 그래프와 SubGraph 가 어떻게 통신하는지(그래프 실행 중에 상태(State) 를 서로 어떻게 전달하는지)


### 두가지 시나리오
# 1. 상위 그래프와 서브그래프가 스키마 키를 공유하는 경우. 이 경우 컴파일된 서브그래프로 노드를 추가 할 수 있다.
# 2. 상위 그래프와 서브그래프가 서로 다른 스키마를 가지는 경우. 이 경우 서브그래프를 호출하는 노드 함수를 추가 해야 한다.
##########################################################################################

from dotenv import load_dotenv
from langgraph.graph import START,END, StateGraph
from typing import TypedDict

load_dotenv()


############################## 1. 스키마 키를 공유하는 경우 ##############################

class ChildState(TypedDict):
    name : str # 부모 그래프와 공유되는 상태키
    family_name : str

# 서브그래프와 첫 번째 노드, family_name 키에 초기값 설정
def subbraph_node_1(state: ChildState):
    return{"family_name": "Kim"}

# 서브그래프의 두번째 노드. 서브그래프 전용 family_name 키와 공유 name 키를 결합하여 새로운 상태 적용.
def subbraph_node_2(state: ChildState):
    # 서브그래프 내부에서만 사용 가능한 family_name 키와 공유 상태 키 name를 사용하여 업데이트 수행
    return{"name": f'{state["name"]} {state["family_name"]}'}

# 서브그래프 구조 정의 및 노드간 연결
subgraph_builder = StateGraph(ChildState)
subgraph_builder.add_node("sub_1", subbraph_node_1)
subgraph_builder.add_node("sub_2", subbraph_node_2)

subgraph_builder.add_edge(START, "sub_1")
subgraph_builder.add_edge("sub_1", "sub_2")
subgraph_builder.add_edge("sub_2", END)

subgraph = subgraph_builder.compile()

####################################################

# 부모 그래프의 상태 정의를 위한 TypedDict 클래스, name 키만 포함
class ParentState(TypedDict):
    name: str
    company: str


# 부모 그래프의 첫 번째 노드, name 키의 값을 수정하여 새로운 상태 생성
def node_1(state: ParentState):
    return {"name": f'My name is {state["name"]}'}


# 부모 그래프 구조 정의 및 서브그래프를 포함한 노드 간 연결 관계 설정
builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
# 컴파일된 서브그래프를 부모 그래프의 노드로 추가
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph = builder.compile()

####################################################

# 그래프 스트림에서 청크 단위로 데이터 처리 및 각 청크 순차 출력
# subgraphs 파라미터를 True로 설정하여 하위 그래프 포함 스트리밍 처리
for chunk in graph.stream({"name": "Ruah"}, subgraphs=True):
    print(chunk)


############################## 2. 스키마 키를 공유하지 않는 경우 ##############################


# 서브그래프의 상태 타입 정의 (부모 그래프와 키를 공유하지 않음)
class ChildState(TypedDict):
    # 부모 그래프와 공유되지 않는 키들
    name: str
# 서브그래프의 첫 번째 노드: name 키에 초기값 설정
def subgraph_node_1(state: ChildState):
    return {"name": "Teddy " + state["name"]}


# 서브그래프의 두 번째 노드: name 값 그대로 반환
def subgraph_node_2(state: ChildState):
    return {"name": f'My name is {state["name"]}'}

# 서브그래프 빌더 초기화 및 노드 연결 구성
subgraph_builder = StateGraph(ChildState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

####################################################

# 부모 그래프의 상태 타입 정의
class ParentState(TypedDict):
    family_name: str
    full_name: str


# 부모 그래프의 첫 번째 노드: family_name 값 그대로 반환
def node_1(state: ParentState):
    return {"family_name": state["family_name"]}


# 부모 그래프의 두 번째 노드: 서브그래프와 상태 변환 및 결과 처리
def node_2(state: ParentState):
    # 부모 상태를 서브그래프 상태로 변환
    response = subgraph.invoke({"name": state["family_name"]})
    # 서브그래프 응답을 부모 상태로 변환
    return {"full_name": response["name"]}


# 부모 그래프 빌더 초기화 및 노드 연결 구성
builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)

# 컴파일된 서브그래프 대신 서브그래프를 호출하는 node_2 함수 사용
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph = builder.compile()

# 그래프 스트리밍 처리를 통한 서브그래프 데이터 청크 단위 순차 출력
# subgraphs=True 옵션으로 하위 그래프 포함하여 스트림 데이터 처리
for chunk in graph.stream({"family_name": "Lee"}, subgraphs=True):
    print(chunk)

####################################################



