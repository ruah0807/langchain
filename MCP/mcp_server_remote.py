"""
- MCP host : MCP를 통해 데이터에 댁세스 하려는 Claude Desktop , IDE 또는 aI 도구와 같은 프로긂
    - Cursor AI, Claude Desktop, LangGraph, 그외 구현할 프로그램
- MCP Client : 서버와 1:1 연결을 유지하는 프로토콜 클라이언트
    - Host & Server 간에 연결을 위한 중개자
- MCP Server : 표준화 된 모델 컴텍스트 프로토콜을 통해 각각 특정 기능을 노출하는 경량 프로그램

# 통신방법
- SSE
    - Model Context Protocol(MCP) 을 Server-sent Events(SSE)를 통해 구현한 프로토콜. HTTP를 통해 원격 서비스에 연결.
    - 특징
        - MCP Server 에서 먼저 서버가 running 되고 있어야함
        - SSE MCP는 SSE 모드에서 잠재력이 크다는 평가를 받고 있음(외부와의 통신 가능성)
- Stdio
    - 통신을 위해 표준 입력/출력 사용
    - 로컬 프로세스에 이상적
    - 예 : Smithery 사이트에서 Json 형식의 내용으로 구동하는 방식은 대부분 Stdio 통신 방식을 사용
        - URL방식으로 접근하는것은 SSE
![](https://velog.velcdn.com/images/looa0807/post/8ef8f716-a541-4822-b0ae-82a0e1108c59/image.png)
"""

"""
# MultiServerMCPClient

`async with`로 일시적인 Session 연결을 생성 후 해제
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Weather",  # Name of MCP Server
    instructions="You are a weather assistant that can answer questions about the weather in a given location",
    host="127.0.0.0",  # host address(0.0.0.0 allows connections form any IP)
    port=8005,  # Port number for the server
)


@mcp.tool()
async def get_weather(location: str) -> str:
    """
    Get current weathe rinforamation for the specified location.

    This function simulates a weather service by returning a fixed response.
    In a production environment, this would connect to a real weather API.

    Args:
        location(str) : The name of the location (city, region, etc.) to get weather for

    Returns:
        str : A string containing the weathe rinformation for the specified location
    """
    return f"It's alwasy sunny in {location}"


if __name__ == "__main__":
    print("mcp remote server is running")
    mcp.run(transport="sse")
