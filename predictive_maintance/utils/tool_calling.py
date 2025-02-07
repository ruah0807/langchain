import json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from langchain_teddynote.tools.tavily import TavilySearch
# from langchain_core.messages import ToolMessage
# from a_element.tool_message import ToolMessage

from typing import Literal
from dataclasses import dataclass, field



@dataclass
class ToolMessage:
    content: str
    name: str
    tool_call_id: str
    type: Literal["tool"] = field(default="tool", init=False)
    status: Literal["success", "error"] = "success"

    def __str__(self) -> str:
        return f"ToolMessage(content={self.content}, name={self.name}, tool_call_id={self.tool_call_id})"


# 도구 호출 노드 생성
class BasicToolNode:
    """Run tools requested in the last AIMessage node"""

    def __init__(self, tools: list) -> None:
        # 도구 리스트
        self.tools_list = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # 메시지가 존재할 경우 가장 최근 메시지 1개 추출
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        # 도구 호출 결과 ***** 중요
        outputs = []
        for tool_call in message.tool_calls:
            # 도구 호출 후 결과 저장
            tool_result = self.tools_list[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                # 도구 호출 결과를 메시지로 저장
                ToolMessage(
                    content=json.dumps(
                        tool_result, ensure_ascii=False
                    ),  # 도구 호출 결과를 문자열로 변환
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}

if __name__ == "__main__":
        # 도구 메시지 생성
    tool_message = ToolMessage(
        content="검색 결과: 파이썬은 프로그래밍 언어입니다.",
        name="search",
        tool_call_id="search_123",
    )

    # 도구 메시지 출력
    print(tool_message)
    print(f"Content: {tool_message.content}")
    print(f"Tool Name: {tool_message.name}")
    print(f"Tool Call ID: {tool_message.tool_call_id}")
    print(f"Status: {tool_message.status}")

# # 검색 도구 생성
# tool = TavilySearch(max_results=3)

# # # 도구 목록에 추가
# tools = [tool]

# # 도구 노드 생성
# tool_node = BasicToolNode(tools=[tool])

# # 그래프에 도구 노드 추가
# graph_builder.add_node("tools", tool_node)