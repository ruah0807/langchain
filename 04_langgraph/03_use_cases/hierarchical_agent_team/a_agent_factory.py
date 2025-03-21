from typing import List

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gpt-4o"

################## 다중 에이전트 생성을 위한 유틸리티 함수 정의 ##################

# 작업을 간결하게 수행하기 위한 유틸리티 함수 생성.

# functools.partial을 사용하여 에이전트 노드 생성
# 1. worker agent 생성
# 2. sub-graph 의 supervisor 생성

# 에이전트 팩토리 클래스
class AgentFactory:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def create_agent_node(self, agent, name: str):
        # 노드 생성 함수
        def agent_node(state):
         
            result = agent.invoke(state)
            return{
                "messages":[
                    HumanMessage(content=result["messages"][-1].content, name=name)
                ]
            }
        return agent_node


# LLM 초기화
llm = ChatOpenAI(model = MODEL_NAME, temperature = 0)

# Agent Factory 인스턴스 생성
agent_factory = AgentFactory(MODEL_NAME)




############################## 팀 감독자 생성 함수 #################################

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal
import pprint
def create_team_supervisor(model_name, system_prompt, members) -> str:
    # 다음작업자 선택 옵션 목록 정의
    options_for_next=["FINISH"] + members

    # 작업자 선택 응답 모델 정의 : 다음작업자를 선태갛거나 작업완료를 나타냄
    class RouteResponse(BaseModel):
        next: Literal[*options_for_next] # type: ignore
    
    # ChatPromptTemplate 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"), # 히스토리 저장.
            (
                "system",
                "Given the conversation above, who should act next?"
                "Or should we FINISH? Selct one of : {options}"
            )
        ]
    ).partial(options=str(options_for_next))


    # LLM초기화
    llm = ChatOpenAI(model=model_name, temperature=0)

    # 프롬픝트와 LLM을 결합하여 체인구성
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    print("\n\n================ with_structured_output 구조==============\n\n")
    pprint.pprint(vars(llm.with_structured_output(RouteResponse)))
    print("\n\n==============================\n\n")

    return supervisor_chain
