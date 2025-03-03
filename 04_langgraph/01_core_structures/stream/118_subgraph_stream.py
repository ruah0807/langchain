
###############################################################################

######################## Subgraphs 스트리밍 #########################

# 1. Subgraphs 출력을 '생략' 하는 경우
# 2. Subgraphs 출력도 '포함' 하는 경우

###############################################################################

from b_subgraph import subgraph
import asyncio

graph = subgraph()

inputs = {"messages": [("human", "AI agent 관련 최신 뉴스를 검색해줘")]}

config = {"configurable": {"thread_id": "1"}}


######################## Subgraphs 출력을 '생략' 하는 경우 #########################

# for chunk in graph.stream(inputs, config, stream_mode = "updates"):
#     # node_name : 현재 처리중인 노드명, node_chunk: 해당 노드의 청크 데이터
#     for node_name, node_chunk in chunk.items():
#         # 현재 처리 중인 노드 구분선 출력
#         print(f"\n========= Update from node '{node_name}' =========\n")
#         # 해당 노드의 업데이트된 데이터 출력
#         if "messages" in node_chunk:
#             node_chunk["messages"][-1].pretty_print()
#         else:
#             print(node_chunk)


######################## Subgraphs 출력을 '포함' 하는 경우 #########################

# `subgraphs=True` 를 통해 서브그래프 출력 포함
# (namespace, chunk) 형태로 출력된다.



# 네임스페이스 문자열을 보기 좋은 형식으로 변환하는 포맷팅 함수
# def format_namespace(namespace: str) :
#     return namespace[-1].split(":")[0] if len(namespace) > 0 else "parent graph"

# # subgraphs = True 를 통해 서브그래프의 출력도 포함(namespace, chunk) 형태로 출력됨.
# for namespace, chunk in graph.stream(inputs, stream_mode="updates", subgraphs=True):
#     # node_name : 현재 처리중인 노드명, node_chunk: 해당 노드의 청크 데이터
#     for node_name, node_chunk in chunk.items():
#         print(
#             f"\n========= Update from node [{node_name}] in [{format_namespace(namespace)}]=========\n"
#         )
#         # 노드의 청크 데이터 출력
#         if "messages" in node_chunk:
#             node_chunk["messages"][-1].pretty_print()
#         else:
#             print(node_chunk)





############ Subgraphs 안에서 LLM 출력 토큰 단위 스트리밍

# 비동기 함수를 써야함.

# async def subgraph_token_stream(inputs, config):

#     kind = None
#     async for event in graph.astream_events(inputs, config, version="v2", subgraphs=True):
#         kind = event["event"]

#         # 이벤트 종류와 태그 정보 추출
#         if kind == "on_chat_model_start":
#             print(f"\n=========== on_chat_model_start ===========\n")

#         # 채팅 모델 스트림 이벤트  및 최종 노드 태그 필터링
#         elif kind == "on_chat_model_stream":
#             data = event["data"]
            
#             # 토큰 단위의 스트리밍 출력
#             if data["chunk"].content:
#                 print(data["chunk"].content, end="", flush=True)
        
#         # 검색이 일어나는 시점
#         elif kind == "on_tool_start":
#             print(f"\n=========== on_tool_start ===========\n")
#             data = event["data"]
#             if "input" in data:
#                 tool_msg = data["input"]
#                 print(tool_msg)

#         # 검색된 결과가 나오는 시점
#         elif kind == "on_tool_end":
#             print(f"\n=========== on_tool_end ===========\n")
#             data = event["data"]
#             if "output" in data:
#                 tool_output = data["output"]
#                 print(tool_output)
            


######################## 특정 tags만 스트리밍 출력하는 경우 #########################

# `ONLY_STREAM_TAGS`를 통해 스트리밍 출력하고 싶은 tags만 설정하기.
# 도구에 태그 설정해놓은 "WANT_TO_STREAM"은 출력에서 배제하고, "SNS_POST" 태그만 출력하도록 설정.

async def print_specific_tags():
    def parse_namespace_info(info: tuple) -> tuple[str,str]:
            if len(info) > 1:
                namespace, node_name  = info
                return node_name.split(":")[0], namespace.split(":")[0]
            return info[0].split(":")[0], "parent graph"

    ONLY_STRAM_TAGS = ["WANT_TO_STREAM", "SNS_POST"]

    kind = None
    tags = None

    async for event in graph.astream_events(inputs, config, version="v2", subgraphs= True):
        kind = event["event"]
        tags = event.get("tags", [])

        # 이벤트 종류와 태그 정보 추출
        if kind == "on_chat_model_start":
            print(f"\n=========== on_chat_model_start ===========\n")

        # 채팅 모델 스트림 이벤트 및 최종 노드 태그 필터링
        elif kind == "on_chat_model_stream":
            for tag in tags:
                 if tag in ONLY_STRAM_TAGS:
                        # 이벤트 데이터 추출
                        data = event["data"]
                        # 출력 메시지 
                        if data["chunk"].content:
                            print(data["chunk"].content, end="", flush=True)

        # 검색이 일어나는 시점
        elif kind == "on_tool_start":
            print(f"\n=========== on_tool_start ===========\n")
            data = event["data"]
            if "input" in data:
                tool_msg = data["input"]
                print(tool_msg)

        # 검색된 결과가 나오는 시점
        elif kind == "on_tool_end":
            print(f"\n=========== on_tool_end ===========\n")
            data = event["data"]
            if "output" in data:
                tool_msg = data["output"]
                print(tool_msg.content)
                            
            




            
async def main():
    # await subgraph_token_stream(inputs, config)
    await print_specific_tags()
asyncio.run(main())
