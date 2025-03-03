from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from a_agent_graph import define_tools
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image
class State(TypedDict):
    messages: Annotated[list, add_messages]


llm_with_tools, tools = define_tools()


def subgraph():
    def chatbot(state: State):
        # 메시지 호출 및 반환
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # SNS포스트 생성 함수 정의
    def create_sns_post(state: State):
        # SNS post 생성을 위한 프롬프트
        prompt = """
    이전 대화 내용을 바탕으로 SNS 게시글 형식으로 변환해주세요.
    다음형식을 따라주세요.
    - 해시태그 포함
    - 이모지 사용
    - 간결하고 흥미로운 문체 사용
    - 200자 이내로 작성
        """

        messages = state["messages"] + [("human", prompt)]
        sns_llm = ChatOpenAI(model="gpt-4o-mini").with_config(tags=["SNS_POST"])
        return { "messages" : [sns_llm.invoke(messages)]}



    # 서브그래프 생성
    def create_subgraph(tools):
        # 서브그래프용 상태 그래프 생성
        subgraph = StateGraph(State)

        # 챗봇 노드 추가
        subgraph.add_node("chatbot", chatbot)

        # SNS 포스트 생성 노드 추가
        tool_node = ToolNode(tools = tools)
        subgraph.add_node("tools", tool_node)

        # tools > chatbot
        subgraph.add_edge("tools", "chatbot")
        # start > chatbot
        subgraph.add_edge(START, "chatbot")

        # 조건부 엣지 추가
        subgraph.add_conditional_edges(
            "chatbot",
            tools_condition
        )

        # chatbot > end
        subgraph.add_edge("chatbot", END)

        return subgraph.compile()



    # 메인 그래프 생성
    graph_builder = StateGraph(State)

    # 서브 그래프 추가
    subgraph = create_subgraph(tools)
    graph_builder.add_node("news_subgraph", subgraph)

    # SNS post 생성 노드 추가
    graph_builder.add_node("sns_post", create_sns_post)

    graph_builder.add_edge(START, "news_subgraph")
    graph_builder.add_edge("news_subgraph", "sns_post")
    graph_builder.add_edge("sns_post", END)

    graph = graph_builder.compile()


    return graph







