from dotenv import load_dotenv
from langchain_teddynote.tools import TavilySearch
from langgraph.prebuilt import create_react_agent # 랭그래프 기반으로 만들어진 에이전트 : 에이전트를 하나의 노드로 추가 가능. 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import operator
from typing import Annotated, List, Tuple, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

tools = [TavilySearch(max_results=2)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer in Korean. "),
    ("human", "{messages}")
])

# LLM정의
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.76)

### ReAct 에이전트 생성 ###
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)
# state_modifier : 상태 수정자 함수(프롬프트 수정가능)



### 상태 정의 ###
# 상태 정의
class PlanExecute(TypedDict):
    input: Annotated[str, "User's input"] # 사용자 입력
    plan : Annotated[List[str], "Current plan"] # 현재 계획
    past_steps: Annotated[List[Tuple], operator.add] # 이전에 실행한 계획과 실행결과.
    response : Annotated[str, "Final response"] # 최종 응답
    

############################# 계획 단계 #############################
# `function_calling`을 사용하여 계획 수립.

# 모델 정의
class Plan(BaseModel):
    """Sorted steps to execute the plan"""
    steps : Annotated[List[str], "Different steps to follow, should be in sourted order"]



# 계획 수립을 위한 프롬프트 템플릿 생성
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """
For the given objective, come up with a simple  step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superflouous steps.\
The result of the final step shoud be the final answer. Make sure that each step has all the information needed - do not skip steps.\
Answer in Korean.
"""),
    ("placeholder", "{messages}") # 유저메시지를 바탕으로 플랜을 설계.
])

# 플랜 수립 모델 정의
planner = planner_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.76).with_structured_output(Plan)


############################# 플랜 수립 테스트 ##############################
# planner_result = planner.invoke({
#     "messages": [
#         ("user",
#          "Langgraph의 핵심 장단점과 LangGraph를 사용하는 이유는 무엇이니?")
#     ]
# })

# # 플랜 수립 결과 출력
# print(planner_result) 

# response : steps=['Langgraph의 정의를 조사한다.', 'Langgraph의 핵심 장점을 목록으로 정리한다.', 'Langgraph의 핵심 단점을 목록으로 정리한다.', 'Langgraph를 사용하는 이유를 정리한다.', '각 항목에 대해 설명을 추가하여 최종 답변을 완성한다.']
# 총 5개의 계획 생성.


############################# Re-Plan(재계획) 단계 ##############################

class Response(BaseModel):
    """Response to user."""
    # 사용자응답
    response: str

class Act(BaseModel):
    """Action to perform."""
    # 수행할 작업 : "Response", "Plan". 
    # 사용자에게 응답할 경우 Response 사용. 추가 도구 사용이 필요할 경우 Plan 사용.
    action : Union[Response, Plan] = Field(
        description = "Action to perform. If you want to respond to user, use Response."
        "If you need to further use tools to get the answer, use Plan."
    )

### 계획을 재수립하기 위한 프롬프트 정의 ###
# input : 사용자 입력
# plan : 원래 계획
# past_steps : 이전에 어디까지 실행했는가에 대한 여부
replanner_prompt = ChatPromptTemplate.from_template(
    """
For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this :
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. \
Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done.\
Do not return previously done steps as part of the plan.\

Answer in Korean.
"""
)

# replanner 생성
replanner = replanner_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.76).with_structured_output(Act)



############################# Graph 생성 ##############################

from langgraph.graph import END
from langchain_core.output_parsers import StrOutputParser

# 사용자 입력을 기반으로 계획을 생성하고 반환
def plan_step(state: PlanExecute):
    plan = planner.invoke({"mesaages":[("user", state["input"])]})
    # 생성된 계획의 단계 리스트 반환
    return {"plan": plan.steps} 

# 계획의 첫번째 단계를 실행하는 비동기 함수
# 에이전트 실행기를 사용하여 주어진 작업을 수행하고 결과를 반환
def execute_step(state: PlanExecute):
    plan = state["plan"]

    # 계획을 문자열로 변환하여 각 단계에 번호를 매김
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)) 

    # 항상 첫번째 단계를 실행(하나씩 실행하면서 지워지기때문에 다음단계 또한 첫번째가 됨)
    task = plan[0]  
    # 현재 실행할 작업을 포맷팅하여 에이전트에 전달.
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing [step 1. {task}]."""

    # 에이전트 실행기를 통해 작업 수행 및 결과 수신
    agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})

    # 이전 단계와 그 결과를 포함하는 딕셔너리 반환
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)]
    } 

# 이전 단계의 결과를 바탕으로 계획을 업데이트하거나 최종 응답을 반환
# 3가지의 분기문 처리: 
    # 1.최종 답변생성의 경우. 
    # 2. 계획 재조정중 더이상 수행할 계획이 없는 경우. 
    # 3. 계획 재조정중 수행할 계획이 있는 경우.
def replan_step(state: PlanExecute):
    output = replanner.invoke(state)

    # 응답이 사용자에게 반환될경우 action이 Response를 반환할경우.
    if isinstance(output.action, Response):
        return{"response": output.action.response}

    # 추가 단계가 필요할 경우 계획의 단계 리스트 반환
    else:
        next_plan = output.action.steps
        if len(next_plan) == 0:
            return{"response": "No more steps needed."}
        else:
            return {"plan": next_plan}

# 에이전트 실행종류 여부를 결정하는 함수.
def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return "final_report"
    else :
        return "execute"
    


# 최종 보고서 생성을 위한 프롬프트 정의
# arXiv style : 학술적인 문서를 작성하는 스타일.(논문, 보고서, 리포트 등)
final_report_prompt = ChatPromptTemplate.from_template(
    """
You are given the objective and the previously done steps. Your task is to generate a final report.
Final report should be professional and shoud be written in arXiv style.

Your objective was this:
{input}

Your previously done steps(question and answer pairs):
{past_steps}

Generate a final report in arXiv style. Answer in Korean.
"""
)

final_report = (
    final_report_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.76) | StrOutputParser()
)

def generate_final_report(state: PlanExecute):
    past_steps = "\n\n".join(
        [
            f"Question: {past_step[0]}\n\nAnswer: {past_step[1]}\n\n####"
            for past_step in state["past_steps"]
        ]
    )
    response = final_report.invoke({"input": state["input"], "past_steps": past_steps})
    return {"response": response}


######################## 그래프 생성 #########################
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 그래프 생성
workflow = StateGraph(PlanExecute)

# 노드 정의
workflow.add_node("planner", plan_step)
workflow.add_node("execute", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_node("final_report", generate_final_report)

# 엣지 정의
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "execute")
workflow.add_edge("execute", "replan")

workflow.add_conditional_edges(
    "replan",
    should_end,
    {"execute": "execute", "final_report": "final_report"}
)
workflow.add_edge("final_report", END)

app = workflow.compile(checkpointer=MemorySaver())

######################## 그래프 시각화 #########################
# from IPython.display import Image

# img = Image(app.get_graph(xray=True).draw_mermaid_png())
# with open("plan_and_execute.png", "wb") as f:
#     f.write(img.data)

######################## 그래프 실행 #########################

from langchain_teddynote.messages import invoke_graph, random_uuid
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(recursion_limit=50, configurable={"thread_id": random_uuid()})

inputs = {
    "input": "Langgraph의 핵심 장단점과 LangGraph를 사용하는 이유는 무엇이니?"
}

print(invoke_graph(app, inputs, config=config))

# snapshot = app.get_state(config).values
# print(snapshot["response"])

# from IPython.display import Markdown

# Markdown(snapshot["response"])