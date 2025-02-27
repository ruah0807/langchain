##########################################################################################

### SubGraph의 입력과 출력을 변환하는 방법 ###

# subgraph 상태가 parent graph 상태와 완전히 독립적일 수 있다.
# 즉, 두 그래프 간에 중복되는 상태 키(state keys) 가 없을 수 있다는 말임.
# 이럴경우,
# subgraph를 호출하기 전에 입력을 변환하고, 반환하기 전에 한번더 출력을 변환해야함.

##########################################################################################

from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph.state import StateGraph, START, END
load_dotenv()


################## graph와 subgraph 정의 ##################

# 1. parent graph
# 2. parent graph 에 의해 호출될 child subgraph
# 3. child graph 에 의해 호출될 grandchild subgraph

#### 1. 손자 subgraph 정의 ####
class GrandChildState(TypedDict):
    my_grandchild_key: str

# 손자노드의 상태를 처리하는 함수. 입력된 문자열에 인사말 추가
def grandchild_1(state: GrandChildState):
    # 자식 또는 부모 키는 여기서 접근 불가
    return {"my_grandchild_key": f'([GrandChild] {state["my_grandchild_key"]})'}

# 손자 노드의 상태 그래프 초기화
grandchild = StateGraph(GrandChildState)

grandchild.add_node("grandchild_1", grandchild_1)
grandchild.add_edge(START, "grandchild_1")
grandchild.add_edge("grandchild_1", END)

grandchild_graph = grandchild.compile()


# for chunk in grandchild_graph.stream({"my_grandchild_key": "Hi, Ruah!!!!"}, subgraphs=True):
#     print("\n", chunk, "\n")

###########################################################

#### 2. 자식 subgraph 정의 ####

class ChildState(TypedDict):
    my_child_key : str


# 손자 그래프 호출 및 상태 변환함수, 자식 상태를 입력받아 변환된 자식 상태 반환
def call_grandchild_graph(state: ChildState):
    # 참고 : 부모 또는 손자 키는 여기서 접근 불가능
    
    # 자식 상태 채널에서 손자상태 채널로 상태 변환
    grandchild_graph_input = { "my_grandchild_key": state["my_child_key"]}

    # 손자 상태 채널에서 자식 상태 채널로 상태 변환 후 결과 반환.
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)

    return {"my_child_key": f'([Child] {grandchild_graph_output["my_grandchild_key"]})'}




# 자식 상태 그래프 초기화
child = StateGraph(ChildState)
# 참고: 컴파일된 그래프 대신 함수 전달
# 자식 그래프에 노드 추가 및 시작-종료 엣지 연결
child.add_node("child_1", call_grandchild_graph)
child.add_edge(START, "child_1")
child.add_edge("child_1", END)
# 자식 그래프 컴파일
child_graph = child.compile()

###########################################################
# child_graph 그래프 호출
# for chunk in child_graph.stream({"my_child_key": "Hi, SungWoo!!!!"}, subgraphs=True):
#     print("\n", chunk, "\n")


# grandchild_graph의 호출을 별도의 함수(call_grandchild_graph)로 감싸고 있음.
# 이 함수는 grandchild 그래프를 호출하기 전에 입력 상태를 변환하고, grandchild 그래프의 출력을 다시 child 그래프 상태로 변환
# 이러한 변환 없이 grandchild_graph를 직접 .add_node에 전달하면, child와 grandchild 상태 간에 공유된 상태 키(State Key) 이 없기 때문에 LangGraph에서 오류가 발생

# 중요 ::
# `child subgraph` 와 `grandchild subgraph`는 `parent graph`와 공유되지 않는 자신만의 독립적인 state를 가지고 있다는 점에 유의


#### 3. parent graph 정의 ####

class ParentState(TypedDict):
    my_parent_key : str

# 부모 상태의 my_parent_key 값에 '[Parent 1]' 문자열을 추가하는 변환함수
def parent_1(state: ParentState)-> ParentState:
    # 참고 : 자식 또는 손자키는 여기서 접근 불가
    return {"my_parent_key": f'([Parent 1] {state["my_parent_key"]})'}


# **부모 상태와 자식 상태 간의 데이터 변환 및 자식 그래프 호출 처리**
def call_child_graph(state: ParentState) -> ParentState:
    # 부모상태 채널  my_parent_key 에서 자식상태 채널 my_child_key로 상태 변환
    child_graph_input = {"my_child_key": state["my_parent_key"]}

    # 자식 상태 채널 my_child_key 에서 부모 상태 채널 my_parent_key로 상태 변환 
    child_graph_output = child_graph.invoke(child_graph_input)

    return {"my_parent_key": child_graph_output["my_child_key"]}


# 부모 상태의 my_parent_key 값에 '[Parent 2]' 문자열을 추가하는 변환함수
def parent_2(state: ParentState) -> ParentState:
    return {"my_parent_key": f'([Parent 2] {state["my_parent_key"]})'}

# 부모 상태 그래프 초기화 및 노드구성
parent = StateGraph(ParentState)
parent.add_node("parent_1", parent_1)
parent.add_node("child", call_child_graph)
parent.add_node("parent_2", parent_2)

parent.add_edge(START, "parent_1")
parent.add_edge("parent_1", "child")
parent.add_edge("child", "parent_2")
parent.add_edge("parent_2", END)

parent_graph = parent.compile()

# from IPython.display import Image
# img = Image(parent_graph.get_graph(xray=True).draw_mermaid_png())
# with open("parent_graph.png", "wb") as f:
#     f.write(img.data)


for i, chunk in enumerate(parent_graph.stream({"my_parent_key": "Hi, SungWoo & Ruah!!!!"}, subgraphs=True)):
    print(f"chunk {i} : ", chunk, "\n")


#  가장 마지막 청크 : ((), {'parent_2': {'my_parent_key': '([Parent 2] ([Child] ([GrandChild] ([Parent 1] Hi, SungWoo & Ruah!!!!))))'}})

# 실행순서  : Parent 1 -> GrandChild -> Child -> Parent 2
