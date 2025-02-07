from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json
from IPython.display import Image, display

# from messaging import mqtt_subscribe, send_alert  # MQTT, Slack 연동 함수
# from database import save_to_db  # DB 저장 함수
# from maintenance import create_maintenance_request  # 유지보수 API 호출


from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator

# State 정의
class GraphState(TypedDict):
    event: Annotated[List[Document], operator.add] # operator.add : list일때만 사용가능
    validation : Annotated[str, "데이터 검증"]
    action : Annotated[str, "후속 조치 결정"]
    alert : Annotated[str, "경고 알림 발송"]
    log : Annotated[str, "로그 저장"]




# 1. 데이터 수신 (MQTT / ESB)
def receive_anomaly_event(state:GraphState) -> GraphState:
    # 이상 감지 AI에서 Agent로 데이터 수신
    event = "이상감지 AI에서 수신된 이벤트"
    return GraphState(event = event)

# 2. 데이터 검증
def validate_data(state : GraphState) -> GraphState:
    validation = "데이터 검증"
    return GraphState(validation = validation)

# 데이터 검증 이후 분기문 처리 : 이상이 있으면 다음으로 노드로 넘어가고 이상이 없다면 종료

# 3. 후속 조치 결정
def decide_action(state : GraphState) -> GraphState:
    # "normal" : 정상 데이터 → 저장 없이 로그 기록 후 종료
    # "warning" : 경고 상태 → 알림 + 요약 데이터 저장
    # "critical" : 심각 상태 → 유지보수 요청 + 전체 데이터 저장
    # action = "후속 조치 결정"
    # return GraphState(action = action)
    action = state.get("action")
    if action == "critical":
        return ["send_critical_alert"]
    else:
        return ["send_warning"]

# 4. 경고 알림 발송
def send_warning(state: GraphState) -> GraphState:
    # "warning" 상태 → 알림 + 요약 데이터 저장
    alert = "경고 알림 발송"
    return GraphState(alert = alert)


# 5. 긴급 알림 발송 및 유지보수 자동화
def send_critical_alert(state: GraphState) -> GraphState:
    # "critical" 상태 → 유지보수 요청 알림 + 전체 데이터 저장) 
    alert = "긴급 알림 발송 및 유지보수 자동화"
    return GraphState(alert = alert)

# 지속적인 요청 후 유지보수 완료된다면 종료

# 6. 로그 저장 (정상 데이터는 저장 X)
def log_event(state: GraphState) -> GraphState:
    # print(f"✅ 정상 데이터 감지 (로그 기록만 수행) → {event['machine_id']}")
    log = "로그 저장"
    return GraphState(log = log)


# 7. LangGraph 노드 연결
graph = StateGraph(GraphState)

# create Nodes
graph.add_node("receive_anomaly_event", receive_anomaly_event)
graph.add_node("validate_data", validate_data)
graph.add_node("decide_action", decide_action)
graph.add_node("send_warning", send_warning)
graph.add_node("send_critical_alert", send_critical_alert)
graph.add_node("log_event", log_event)

# create Edges
graph.add_edge(START, "receive_anomaly_event")
graph.add_edge("receive_anomaly_event", "validate_data")
graph.add_edge("validate_data", "decide_action")

# decide_action의 조건부 분기 설정
graph.add_conditional_edges(
    "decide_action",
    decide_action,  # 조건 함수
    {
        "send_warning": "send_warning",
        "send_critical_alert": "send_critical_alert",
    }
)

graph.add_edge("send_warning", "log_event")

graph.add_edge("send_critical_alert", "log_event")

graph.add_edge("log_event", END)


# 8. 실행
memory = MemorySaver()

flow = graph.compile(checkpointer=memory)

# 그래프 시각화 
img = Image(flow.get_graph(xray=True).draw_mermaid_png())

with open("maintence_structure.png", "wb") as f:
    f.write(img.data)

display(Image("maintence_structure.png"))