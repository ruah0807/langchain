
###############################################################################

######################## 사용자 정의 스트리밍 ########################

# 사용자가 원하는 노드만 스트리밍하게 할수 있다.

# 1. tag 필터링 -> 마지막 메시지(지정한 tag)만 스트리밍

# 2. 도구 호출 스트리밍

# 3. Subgraphs 스트리밍

###############################################################################

from a_agent_graph import agent_graph
import asyncio

graph = agent_graph()

inputs = {"messages": [("human", "AI agent 관련 최신 뉴스를 검색해줘")]}

config = {"configurable": {"thread_id": "1"}}



######################## 사용자 정의 Tag 필터링된 스트리밍 ########################

# LLM의 출력이 여러군데에서 발생하는 경우, 특정노드에서 출력된 메시지만 출력하고 싶은 경우,
# `tags`를 추가하여 출력하고 싶은 노드만 선별 가능하다.

# LLM에 tags를 추가하는 방법 :
#   `llm.with_config(tags=["TAG_NAME"])`

# 위를 통해 이벤트를 더 정확하게 필터링하여 해당 모델에서 발생한 이벤트만 유지가 가능.

##### 예 ) `WANT_TO_STREAM` 태그가 있는 경우만 출력 : 
# llm_with_tools = llm.bind_tools(tools).with_config(tags=["WANT_TO_STREAM"])

### **`astream_events`** : tags로 필터링 해야하는경우 해당 함수를 사용해야함. ###

async def stream_events(inputs, config):
    # 비동기 이벤트 스트림 처리 (**astream_events**)
    async for event in graph.astream_events(inputs, config, version="v2"):
        #이벤트 종류와 태그 정보 추출
        kind = event["event"]
        tags = event.get("tags", [])
        # print(f"kind: {kind}, tags: {tags}")
        # 채팅 모델 스트림 이벤트 및 뢰종 노드 태그 필터링
        # on_chat_model_stream : 토큰 출력이 일어나는 단계를 말함.
        if kind == "on_chat_model_stream" and "WANT_TO_STREAM" in tags:
            # 이벤트 데이터 추출
            data = event["data"]
            # 출력 메시지
            if data["chunk"].content:
                print(data["chunk"].content, end="", flush=True)

async def main():
    await stream_events(inputs, config)

asyncio.run(main())

######################## 도구 호출에 대한 스트리밍 출력 #########################

# AIMessageChunk : 실시간 토큰 단위의 출력 메시지

# `tool_call_chunks` : 도구 호출 청크. 
#                    만일 tool_call_chunks가 존재할 경우, 도구 호출 청크를 누적하여 출력 
#                    (도구 토큰은 위 속성을 보고 판단하여 출력한다.)               

from langchain_core.messages import HumanMessage, AIMessageChunk
# 첫번재 메시지 처리 여부 플래그 설정
first  = True

# 비동기 스트림 처리를 통한 메시지 및 메타데이터 순차 처리
for msg, metadata in graph.stream(inputs, config, stream_mode = "messages"):
    # 사용자 메시지가 아닌 경우의 컨텐츠 출력 처리
    if msg.content and not isinstance(msg, HumanMessage):
        print(msg.content, end="", flush=True )

    # AI 메시지 청크 처리 및 누적
    if isinstance(msg, AIMessageChunk):
        if first:
            gathered_chunks = msg
            first = False
        else:
            gathered_chunks = gathered_chunks + msg
        
        # 도구 호출 청크 존재시 누적된 도구 호출 정보 출력
        if msg.tool_call_chunks:
            print(
                gathered_chunks.tool_calls[0]["name"], # 도구 이름
                " : ", 
                gathered_chunks.tool_calls[0]["args"] # 도구 인자
                ) 



######################## Subgraphs 스트리밍 #########################


