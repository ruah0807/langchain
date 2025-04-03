from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatAnthropic(
    model_name="claude-3-7-sonnet-latest", temperature=0, max_tokens=2000
)

sse = {"weather": {"url": "http://localhost:8005/sse", "transport": "sse"}}


async def message_stream():
    async with MultiServerMCPClient(sse) as client:
        print(client.get_tools())

        agent = create_react_agent(model, client.get_tools())
        config = {"configurable": {"thread_id": "1"}}
        prev_node = ""

        async for chunk_msg, metadata in agent.astream(
            {"messages": "서울의 날씨는 어떠니?"}, config, stream_mode="messages"
        ):
            curr_node = metadata["langgraph_node"]
            final_result = {
                "node": curr_node,
                "content": chunk_msg,
                "metadata": metadata,
            }
            node_names = []
            callback = None

            # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
            if not node_names or curr_node in node_names:
                # 콜백 함수가 있는 경우 실행
                if callback:
                    result = callback({"node": curr_node, "content": chunk_msg})
                    if hasattr(result, "__await__"):
                        await result
                # 콜백이 없는 경우 기본 출력
                else:
                    # # 노드가 변경된 경우에만 구분선 출력
                    if curr_node != prev_node:
                        print("\n" + "=" * 50)
                        print(f"Node: {curr_node}")
                        print("- " * 25)

                    # Claude/Anthropic 모델의 토큰 청크 처리 - 항상 텍스트만 추출
                    if hasattr(chunk_msg, "content"):
                        # 리스트 형태의 content (Anthropic/Claude 스타일)
                        if isinstance(chunk_msg.content, list):
                            for item in chunk_msg.content:
                                if isinstance(item, dict) and "text" in item:
                                    print(item["text"], end="", flush=True)
                        # 문자열 형태의 content
                        elif isinstance(chunk_msg.content, str):
                            print(chunk_msg.content, end="", flush=True)
                    # 그 외 형태의 chunk_msg 처리
                    else:
                        print(chunk_msg, end="", flush=True)

                prev_node = curr_node


async def with_initialize():
    """
    Session을 생성하고 종료하는 코드

    Async Session을 유지하며 도구에 접근하는 방식
    """
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8005/sse",
                "transport": "sse",
            }
        }
    )
    await_client = await client.__aenter__()
    print(await_client.get_tools())

    await client.__aexit__(None, None, None)
    return await_client


if __name__ == "__main__":
    # 비동기 함수 실행
    import asyncio

    async def main():
        await with_initialize()

    asyncio.run(main())
