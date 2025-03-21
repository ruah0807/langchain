##########################################################################################

### 스트리밍 모드 ###

# LangGraph 에서는 여러 스트리밍 모드를 지원한다.

# 1. `values` : 모든 그래프들의 값을 스트리밍 -> 각 노드가 호출된 후의 그래프의 전체 상태를 의미
# 2. `updates` : 그래프의 업데이트 내용을 스트리밍 -> 각 노드의 호출 후의 그래프 상태에 대한 업데이트
# 3. `messages` : 각 노드의 메시지를 스트리밍 -> LLM의 토큰단위 출력.

##########################################################################################
from dotenv import load_dotenv

load_dotenv()


######################## 그래프 정의 ########################
# 미리 정의해둔 기본 agent 그래프
from a_agent_graph import agent_graph

graph = agent_graph()


######################## 노드의 단계별 출력 ########################

##### 스트리밍 모드
# `values`: 각 단계의 현재 상태값 출력
# `updates`: 각 단계의 업데이트 내용 출력(기본값)
# `messages`: LLM의 토큰단위 메시지 출력

# 여기서 스트리밍의 의미는 LLM 출력시 토큰단위로 스트리밍 하는 개념이 아니라, 단계별로 출력하는 의미임 !!!!!!

##### `stream_mode = "values"`
# `values` 모드는 각 단계의 현재 상태값을 출력한다.

# `chunk.items()`
#   `key` : State의 key 값
#   `value` : State의 key에 대한 value


##################### 공통 질문과 config 설정 #####################

# 질문 입력
inputs = {"messages": [("human", "AI agent 관련 최신 뉴스를 검색해줘")]}
config = {"configurable": {"thread_id": "1"}}



###################### 동기 (Synchronous) 방식의 스트리밍 ######################
#   `chunk`는 dictionary 형태 (key:value)


# # 동기 스트림 처리(stream_mode="values")
for chunk in graph.stream(inputs, config, stream_mode="values"):
    # chunk는 dictionary 형태(key:value)
    for state_key, state_value in chunk.items():
        if state_key =="messages":
            state_value[-1].pretty_print()


###################### 비동기 (Asynchronous) 방식의 스트리밍 ######################
#  `astream()`메서드는 비동기 스트림 처리를 통해 그래프를 실행하고 값모드로 청크 단위 응답을 생성한다.
#  `async for` 문을 사용하여 비동기 스트림 처리를 수행.

async def values_stream(inputs, config):
    async for chunk in graph.astream(inputs, config, stream_mode="values"):
        # chunk는 dictionary 형태 (key:value)
        for state_key, state_value in chunk.items():
            if state_key == "messages":
                state_value[-1].pretty_print()

# # 비동기 함수 실행
import asyncio

async def main():
    await values_stream(inputs, config)

asyncio.run(main())

###################### `stream_mode`= "updates" ######################

# `updates` 모드는 각 단계의 업데이트된 State 만을 출력한다.

# 출력은 `노드 이름`을 `key`로, 없데이트된 값을 value로 하는 dictionary 형태.

# chunk.items()

# key: 노드(Node) 의 이름
# value: 해당 노드(Node) 단계에서의 출력 값(dictionary). 
#       즉, 여러 개의 key-value 쌍을 가진 dictionary.



######### 동기 Synchronous 방식의 updates스트리밍 
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    # chunk는 dictionary 형태 (key:노드, value:노드의 상태값)
    for node, value in chunk.items():
        if node:
            print(f"\n[ Node : {node}]\n")
        if "messages" in value:
            value["messages"][-1].pretty_print()

######### 비동기 Asynchronous 방식의 updates스트리밍 스트리밍 
async def updates_stream(inputs, config):
    async for chunk in graph.astream(inputs, config, stream_mode="updates"):
        # chunk는 dictionary 형태 (key:value)
        for node, value in chunk.items():
            if node:
                print(f"\n[ Node : {node}]\n")
            if "messages" in value:
                value["messages"][-1].pretty_print()

# # 비동기 함수 실행
import asyncio

async def main():
    await updates_stream(inputs, config)

asyncio.run(main())


###################### `stream_mode`= "messages" ######################
# `messages` 모드는 LLM의 토큰단위 메시지를 출력한다.

# `chunk` 는 두개의 요소를 가진 tuple 이다.
    # `chunk_msg` : token 단위 메시지
    # `metadata` : 노드 정보

##### LLM 응답만이 token 별로 streaming 모드가 가능하기때문에, 
#       노드가 "chatbot"일때만 messages 스트리밍 모드를 사용한다.


##### 동기 Synchronous 방식의 messages스트리밍 
# for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):

#     # chatbot 노드에서 출력된 메시지만 출력
#     if metadata["langgraph_node"] == "chatbot":
#         if chunk_msg.content:
#             print(chunk_msg.content, end="", flush=True)
#     else:
#         print(chunk_msg.content)
#         print(f"\n\nmetadata: \n{metadata}\n\n")


##### 비동기 Asynchronous 방식의 messages스트리밍 
# async def message_stream(inputs, config):
#     async for chunk_msg, metadata in graph.astream(inputs, config, stream_mode="messages"):
#         # chunk는 dictionary 형태 (key:value)
#         if metadata["langgraph_node"] == "chatbot":
#             if chunk_msg.content:
#                 print(chunk_msg.content, end="", flush=True)
#         else:
#             print(chunk_msg.content)
#             print(f"\n\nmetadata: \n{metadata}\n\n")

# # 비동기 함수 실행
# import asyncio

# async def main():
#     await message_stream(inputs, config)

# asyncio.run(main())


######################################################################################

###### 특정 노드만 스트리밍 하기 ######

# 특정 Node에 대해 출력하고 싶은 경우, `stream_mode="messages"`를 통해 설정가능하다.

# stream_mode="messages" 설정시, (chunk_msg, metadata) 형태로 메시지를 받는다.
#   `chunk_msg` : 실시간 출력 메시지
#   `metadata` : 노드 정보

# metadata["langgraph_node"] 를 통해 특정 노드에서 출력된 메시지만 출력할 수 있다.
    # 예 ) chatbot 노드에서 출력된 메시지만 출력하는 경우.
    # metadata["langgraph_node"] == "chatbot" 인 경우, 출력 

######################################################################################

from langchain_core.messages import HumanMessage

for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
    # HumanMessage 가 아닌 최종 노드의 유효한 컨텐츠만 출력처리
    if (
        chunk_msg.content
        and not isinstance(chunk_msg, HumanMessage)
        and metadata["langgraph_node"] == "chatbot"
    ):
        print(chunk_msg.content, end="", flush=True) 
        # end="" : 줄바꿈 없이 출력
        # flush=True : 출력 버퍼를 강제로 비워 즉시 출력. (줄바꿈없이 다음출력이 이어지도록 함.)
print(f"\n\nmetadata: {metadata}\n\n")

# 출력결과
# 다음은 "AI agent"에 대한 최신 뉴스입니다:
# 1. **AI 에이전트 AI Agents 란 무엇인가?** - 삼성SDS의 인사이트 리포트에서 AI 에이전트에 대한 설명과 관련 정보가 제공됩니다.
# 2. **AI Agent by RightBrain** - 브런치에서 AI 에이전트에 관한 콘텐츠를 소개합니다.
# 3. **MIT 테크놀로지 리뷰 매거진 Vol.19** - MIT 테크놀로지 리뷰에서 AI 및 관련 기술에 대한 최신 경향을 다룹니다.
# 4. **현명한 코인 거래를 돕는 AI 에이전트 마인드 오브 페페** - Cryptonews에서 AI 에이전트 "마인드 오브 페페"의 현재 사전판매 상태에 대한 소식이 보도되었습니다.
# 5. **'AI 에이전트 전용' 구인 게시판 등장** - AI타임스에서는 AI 에이전트를 위한 구인 게시판이 등장했지만, 인간 대체는 아직 불가능하다는 내용의 기사를 다룹니다.
# 더 궁금한 점이 있거나, 특정 뉴스에 대해 더 알고 싶으시면 말씀해 주세요!

# metadata: 
# {'thread_id': '1', 
# 'langgraph_step': 3, 
# 'langgraph_node': 'chatbot', 
# 'langgraph_triggers': ['tools'], 
# 'langgraph_path': ('__pregel_pull', 'chatbot'), 
# 'langgraph_checkpoint_ns': 'chatbot:429d9d94-a202-3dbb-0ec7-0d6aa8bb31b3', 
# 'checkpoint_ns': 'chatbot:429d9d94-a202-3dbb-0ec7-0d6aa8bb31b3', 
# 'ls_provider': 'openai', 
# 'ls_model_name': 'gpt-4o-mini', 
# 'ls_model_type': 'chat', 
# 'ls_temperature': 0.67}