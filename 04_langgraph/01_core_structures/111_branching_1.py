###################### fan-out 값의 신뢰도에 따른 실행 순서 결정 ##########################

# 병렬로 펼쳐진 노드들은 하나의 "super-step"으로 실행된다.
# 각 super-step에서 발생한 업데이트들은 해당 super-step 이 완료된 후 순차적으로 상태에 적용됨.

# 병렬 super-step 에서 일관된 사전 정의도니 업데이트 순서가 필요한 경우 
#   1. 출력값을 식별 키와함께 상태의 별도 필드에 기록
#   2. fan-out 된 각 노드에서 fan-in 지점까지 일반 `edge`를 추가하여 'sink' 노드에서 이들을 결합. 

###  예 : 병렬 단계의 출력을 '신뢰도'에 따라 정렬하고자 하는 경우

from typing import Annotated, Sequence, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


# fan-out 값들의 병합 로직 구현, 빈리스트 처리 및 리스트 연결 수행
def reduce_fanouts(left, right):
    if left is None:
        left = []
    if not right:
        # 덮어쓰기
        return []
    return left + right 

# 상태 관리를 위한 타입 정의, 집계 및 팬아웃 값 저장 구조 설정
class State(TypedDict):
    aggregate: Annotated[list, add_messages]
    fanout_values : Annotated[list, reduce_fanouts]
    which : str



def a_node(state:State):
    msg = "I'm A"
    return {"aggregate": msg}

# 그래프 초기화
builder = StateGraph(State)
builder.add_node("a", a_node)
builder.add_edge(START, "a")

# 병렬 노드 값 반환 클래스
class ParallelReturnNodeValue:
    def __init__(self, node_secret: str, weight: float):
        self._value = node_secret
        self._weight = weight
    
    # 호출시 상태 업데이트
    def __call__(self, state: State) -> Any:
        print(f"Adding {self._value} to {state['aggregate']} in parallel.")
        return {
            "fanout_values": [
                {
                    "value":[self._value],
                    "weight": self._weight
                }
            ]
        }
    
# weight가 다른 병렬 노드 추가
builder.add_node("b", ParallelReturnNodeValue("I'm B", weight=0.3))
builder.add_node("c", ParallelReturnNodeValue("I'm C", weight=0.4))
builder.add_node("d", ParallelReturnNodeValue("I'm D", weight=0.9))

# fan-out 값들을 신뢰도 기준으로 정렬하고 최종 집계 수행
def aggregate_fanout_values(state:State) -> Any:
    #신뢰도 기준 설정
    ranked_values = sorted(
        state["fanout_values"], key = lambda x: x["weight"], reverse=True
    )
    print(ranked_values)
    return {
        "aggregate": [x["value"][0] for x in ranked_values] + ["I'm E"]
    }

# 집계 노드
builder.add_node("e", aggregate_fanout_values)

# 상태에 따른 조건부 라우팅 로직 구현
def route_bc_or_cd(state:State)-> Sequence[str]:
    if state["which"] == "cd":
        return["c", "d"]
    return ["b","c", "d"]

# 중간 노드들 설정 및 조건부 엣지 추가
intermediates = ["b", "c", "d"]
builder.add_conditional_edges("a", route_bc_or_cd, intermediates)

# 중간 노드들과 최종 집계 노드 연결
for node in intermediates:
    builder.add_edge(node, "e")

# 컴파일
graph = builder.compile()


# from IPython.display import Image

# img = Image(graph.get_graph(xray=True).draw_mermaid_png())
# print(img)
# with open("branching_graph.png", "wb") as f:
#     f.write(img.data)

# 그래프 실행
print_graph =graph.invoke({"aggregate":[], "which": "bc", "fanout_values":[]})
print(print_graph)
# 출력 결과
# Adding I'm B to [HumanMessage(content="I'm A", additional_kwargs={}, response_metadata={}, id='962ec3f3-9893-49b7-b1ab-eda116c197d9')] in parallel.
# Adding I'm C to [HumanMessage(content="I'm A", additional_kwargs={}, response_metadata={}, id='962ec3f3-9893-49b7-b1ab-eda116c197d9')] in parallel.
# [{'value': ["I'm C"], 'weight': 0.4}, {'value': ["I'm B"], 'weight': 0.3}]

# {'aggregate': [HumanMessage(content="I'm A", additional_kwargs={}, response_metadata={}, id='2e45b932-764a-4a60-8a5e-d121a001e996'), 
# HumanMessage(content="I'm C", additional_kwargs={}, response_metadata={}, id='3e16d38b-f644-41de-aa6c-d3ee7b1c907d'), 
# HumanMessage(content="I'm B", additional_kwargs={}, response_metadata={}, id='936436d1-5f7d-43f8-8db7-ad0367308753'), 
# HumanMessage(content="I'm E", additional_kwargs={}, response_metadata={}, id='44162280-6921-4369-8036-0422fa7c4600')], 
# 'fanout_values': [{'value': ["I'm B"], 'weight': 0.3}, {'value': ["I'm C"], 'weight': 0.4}], 'which': 'bc'}

# print_graph =graph.invoke({"aggregate":[], "which": "cd", "fanout_values":[]})
# print(print_graph)

# Adding I'm C to [HumanMessage(content="I'm A", additional_kwargs={}, response_metadata={}, id='916f709f-b95a-486a-96c6-6cc644ff4753')] in parallel.
# Adding I'm D to [HumanMessage(content="I'm A", additional_kwargs={}, response_metadata={}, id='916f709f-b95a-486a-96c6-6cc644ff4753')] in parallel.
# [{'value': ["I'm D"], 'weight': 0.9}, {'value': ["I'm C"], 'weight': 0.4}]
# {'aggregate': [HumanMessage(content="I'm A", additional_kwargs={}, response_metadata={}, id='916f709f-b95a-486a-96c6-6cc644ff4753'), 
# HumanMessage(content="I'm D", additional_kwargs={}, response_metadata={}, id='aa82bcac-0d31-4929-9aa6-27bdbd74766c'),
# HumanMessage(content="I'm C", additional_kwargs={}, response_metadata={}, id='6646e12e-5420-4b70-a34a-6e0daaa93bcc'), 
# HumanMessage(content="I'm E", additional_kwargs={}, response_metadata={}, id='37250af1-6520-4e27-bc99-a070b9170faf')], 
# 'fanout_values': [{'value': ["I'm C"], 'weight': 0.4}, {'value': ["I'm D"], 'weight': 0.9}], 'which': 'cd'}