from pickle import GET
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o"


#############################################################################################
# 데이터베이스 설정

#   SQLite 데이터베이스 생성 : 설정과 사용이 간편한 경량 데이터베이스
#   `chinook` DB Load : 디지털 미디어 스토어를 나타내는 샘플 데이터 베이스
#   [SQLite Sample Database](https://www.sqlitetutorial.net/sqlite-sample-database/)

#############################################################################################

# 데이터 베이스 다운로드
# import requests

# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

# response = requests.get(url)

# if response.status_code == 200:
#     with open("Chinook.db", "wb") as file:
#         file.write(response.content)
#     print("File downloaded and saved as Chinook.db")
# else:
#     print(f"Failed to download the file. Status code: {response.status_code}")


# 다운로드 받은 DB를 사용하여 SQLDatabase 도구를 생성하고 샘플 쿼리인 `"SELECT * FROM Artist LIMIT 5;`를 실행

from langchain_community.utilities.sql_database import SQLDatabase

# SqLite db 파일에서 SQLDatabase 인스턴트 생성
db = SQLDatabase.from_uri("sqlite:///04_langgraph/03_use_cases/sql_agent/db/chinook.db")

# DB dialect 출력(sqlite)
# print(db.dialect) 

# 데이터베이스에서 사용 가능한 테이블 이름 목록 출력
# print(db.get_usable_table_names())
    # ['albums', 'artists', 'customers', 'employees', 'genres', 'invoice_items', 'invoices', 'media_types', 'playlist_track', 'playlists', 'tracks']

# SQL 쿼리 실행
result = db.run("SELECT * FROM artists LIMIT 5;")
    # [(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains')]

# 결과 출력
print(result)

#############################################################################################

### 유틸리티 함수

# 에이전트 구현을 돕기 위해 몇 가지 유틸리티 함수를 정의한다.
# 특히, `ToolNode`를 오류 처리와 에이전트에 오류를 전달하는 기능을 포함하여 래핑.

from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode

# 오류처리 함수
def handle_tool_error(state) -> dict:
    # 오류 정보 조회
    error = state.get("error")
    #도구 정보 조회
    tool_calls = state["messages"][-1].tool_calls
    #ToolMessage 로 래핑 후 반환
    return {
        "messages": [
            ToolMessage(
                content=f"Here is the error: {repr(error)}\n\nPlease fix your mistakes.",
                tool_call_id= tc["id"],
            )
            for tc in tool_calls
        ]
    }

# 오류를 처리하고 에이전트에 오류를 전달하기 위한 ToolNode 생성
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    # 오류 발생 시 대체 동작을 정의하여 ToolNode에 추가
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key='error'
    )


#############################################################################################

#### SQL 쿼리 실행 도구

# 에이전트가 DB와 상호작용할 수 있도록 도구 정의

#   1. list_tables_tool : DB에서 사용가능한 table을 가져오기
#   2. get_schema_tool : 텝이블의 DDL을 가져오기
#   3. db_query_tool : 쿼리를 실해앟고 결과를 가져오거나 쿼리가 실패할 경우 오류 메시지 반환

# - DDL(데이터 정의 언어, Data Definition Language) : DB의 구조와 schema를 정의하거나 수정하는 SQL 명령어들.
    # - table, index, view, schema 등의 DB 객체 생성, 수정, 삭제 등의 작업 수행

# 주요 DDL 명령어
# CREATE : 객체 생성
    # EX) CREATE TABLE users (id INT, name VARCHAR(100));
# ALTER : 객체 수정
    # EX) ALTER TABLE users ADD COLUMN email VARCHAR(100);
# DROP : 객체 삭제
    # EX) DROP TABLE users;

# 그러나, LLM이 함부로 db객체를 수정하거나 삭제할 수 있기 때문에 ALTER, DROP 명령어는 제한적으로 사용


#### 데이터베이스 쿼리 관련 도구 ####

# `SQLDatabaseToolkit` 도구 목록

# Query SQLDataBaseTool
    # SQL 쿼리 실행 및 결과 반환
    # Input : 정확한 SQL query
    # output : DB 결과 또는 error message
    # Error 처리 :
        # Query 오류 발생 시 재작성 및 재시도
        # `Unknown column` 오류시 `sql_db_schema`로 정확한 table fields 확인

# InfoSqLDatabaseTool
    # Table schema 및 sample data 조회
    # Input : 콤마로 구분된 table 목록
    # ex ) table1, table2, table3
    # 주의 사항 : `sql_db_list_tables` 로 table 존재 여부 사전 확인 필요

# ListSQLDatabaseTool
    # Database 내 table 목록 조회

# QuerySQLCheckerTool
    # Query 실행 전 유효성 검사
    # 검사 항목:
        # NULL 값과 NOT IN 사용
        # UNION vs UNION ALL 적절성
        # BETWEEN 범위 설정
        # Data type 일치 여부
        # Identifier 인용 적절성
        # Function argument 수
        # Data type casting
        # Join column 정확성
    # 특징: GET-4 model 기반 검증 수행

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# SQLDatabaseToolkit 생성
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model=MODEL))

# SqlDatabaseToolkit 도구 목록
tools = toolkit.get_tools()
# print(tools) # 모든 도구 출력

# list_tables_tool과 get_schema_tool 에 대한 실행 예시
# 데이터베이스에서 사용가능한 테이블들을 나열하는 도구 선택
list_tables_tool = next(tool for tool in tools if tool.name =="sql_db_list_tables")

# 데이터베이스의 모든 테이블 목록 출력
# print(list_tables_tool.invoke(""))
    # 출력 결과
    # albums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks


# 특정 테이블의 DDL을 가져오는 도구 선택
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# Artist 테이블의 DDL 정보 출력
# print(get_schema_tool.invoke("artists"))
    # 출력 결과
    # CREATE TABLE artists (
        #         "ArtistId" INTEGER NOT NULL, 
        #         "Name" NVARCHAR(120), 
        #         PRIMARY KEY ("ArtistId")
        # )
    # /*
    # 3 rows from artists table:
    # ArtistId        Name
    # 1       AC/DC
    # 2       Accept
    # 3       Aerosmith
    # */

### db_query_tool 정의
# db_query_tool의 경우, 데이터베이스에 대해 쿼리를 실행하고 결과를 반환
# 만약, error 가 발생하면 오류 메시지를 반환

from langchain_core.tools import tool

# query 실행 도구
@tool
def db_query_tool(query: str) -> str:
    """
    Run SQL queries against a database and return results
    Returns an error message if the query is incorrect
    If an error is returned, rewrite the query, check, and retry
    """
    #쿼리 실행
    result = db.run_no_throw(query)

    # 오류 : 결과가 없으면 오류 메시지 반환
    if not result :
        return "Error : Query failed. Please rewrite your query and try again."
    
    return result 

### 정상실행된 경우

# Artist 테이블에서  상위 10개 행 선택 및 실행 결과 출력
# print(db_query_tool.invoke("SELECT * FROM artists LIMIT 10"))
# [(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains'), (6, 'Antônio Carlos Jobim'), (7, 'Apocalyptica'), (8, 'Audioslave'), (9, 'BackBeat'), (10, 'Billy Cobham')]


### 오류 발생 시

# artists 테이블에서 상위 10개 행 선택 및 실행 결과 출력
# print(db_query_tool.invoke("SELECT * FROM artists LIMITS 10"))
    # Error: (sqlite3.OperationalError) near "10": syntax error
    # [SQL: SELECT * FROM artists LIMITS 10]
    # (Background on this error at: https://sqlalche.me/e/20/e3q8)

#############################################################################################

#### SQL 쿼리 점검(SQL Query Checker)

# sql 쿼리에서 일반적인 실수를 점검하기 위한 LLM활용
# 이 후 워크 플로우에 노드로 추가

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# sql query의 일반적인 실수를 점검 하기위한 시스템 메시지 정의
query_check_system  = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""" 

# prompt 생성
query_check_prompt =ChatPromptTemplate.from_messages(
    [("system", query_check_system ), ("placeholder", "{messages}")]
)

# query checker 체인 생성
query_check = query_check_prompt | ChatOpenAI(
    model=MODEL,
    temperature=0
    ).bind_tools([db_query_tool], tool_choice="db_query_tool")

### TEST : 잘못된 쿼리 실행하여 결과가 잘 수정되었는지 확인.
# LIMIT -> LIMITS
# response = query_check.invoke(
#     {"messages": [("user", "SELECT * FROM artists LIMITS 10")]},
# )

# print(response.tool_calls[0])
    # {'name': 'db_query_tool', 'args': {'query': 'SELECT * FROM artists LIMIT 10'}, 'id': 'call_A3VxhwTf7lK87DzQTfoxuj7f', 'type': 'tool_call'}


#############################################################################################

##### 그래프 정의 ####

# 워크플로우 정의
# Agent는 먼저 `list_tables_tool`을 강제로 호출하여 데이터베이스에서 사용가능한 테이블을 가져온 후, 
# 튜토리얼 초반에 언급된 단계를 따른다.

from typing import Annotated, Literal
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
import operator

# 에이전트의 상태 정의
class State(TypedDict):
    # messages : Annotated[list[AnyMessage], add_messages]
    messages : Annotated[list[AnyMessage], operator.add]

# 새로운 그래프 정의
workflow = StateGraph(State)

# 첫번째 도구 호출을 위한 노드 추가
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args":{},
                        "id": "initial_tool_call_abc123"
                    }
                ]
            )
        ]
    }

# 쿼리의 정확성을 모델로 점검하기 위한 함수 정의
def model_check_query(state:State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to check that your query is correct before you run it.
    """
    checked_query = query_check.invoke({"messages": [state["messages"][-1]]})
    # print(checked_query)
    return {"messages": [checked_query]}



# 최종 상태를 나타내는 도구 설명
class SubmitFinalAnswer(BaseModel):
    """쿼리 결과를 기반으로 사용자에게 최종 답변 제출"""
    final_answer : str = Field(..., description = "The final answer to the user")

# 질문과 스키마를 기반으로 쿼리 생성을 위한모델 노드 추가
QUERY_GEN_INSTRUCTION = """ You are a SQL expert with a strong attention to detail.
You can define SQL queries, analyze queries results and interpretate query results to response an answer.
Read the messages bellow and identify the user question, table schemas, query statement and query result, or error if they exist.

1. If there's not any query result that make sense to answer the question, 
create a syntactically correct SQLite query to answer the user question.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

2. If you create a query, response Only the query statement. For example, "SELECT id, name FROM pets;"

3. If a query was already executed, but there was an error. Response with the same error message you found. 
For example: "Error: Pets table doesn't exist"

4. If a query was already executed successfully interpretate the response and answer the question following this pattern:
Answer : <<question answer>>. For Example: "Answer : There three cats registered as adopted"
"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", QUERY_GEN_INSTRUCTION), ("placeholder", "{messages}")]
)

query_gen = query_gen_prompt | ChatOpenAI(model=MODEL, temperature=0).bind_tools(
    [SubmitFinalAnswer, model_check_query]
)


# 조건부 엣지 생성
def should_continue(state:State) -> Literal[END, "correct_query", "query_gen"]: # type: ignore
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.content.startswith("Answer :"):
        return END
    if last_message.content.startswith("Error :"):
        return "query_gen"
    else:
        return "correct_query"
    
# 쿼리 생성 노드 정의
def query_gen_node(state: State) :
    message = query_gen.invoke(state)

    # LLM이 잘못된 도구를 호출할 경우 오류 메시지를 반환
    tool_messages = []
    message.pretty_print()
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content = f"""Error : The wrong tool was called : {tc["name"]}, \
                            Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. \
                            Generated queries should be outputted WITHOUT a tool call.""",
                        tool_call_id = tc["id"]
                    )
                )
                # print(tool_messages)
    else:
        tool_messages= []
    return {"messages": [message] + tool_messages}


#################################################################################################

##### 그래프 정의 #####

# 첫번째 도구 호출 노드 추가
workflow.add_node("first_tool_call", first_tool_call) # 도구 호출을 위한 노드
workflow.add_node("list_tables_tool", create_tool_node_with_fallback([list_tables_tool])) # 테이블 리스트 호출
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool])) # 선택한 테이블 schema 호출

model_get_schema = ChatOpenAI(model=MODEL, temperature=0).bind_tools(
    [get_schema_tool]
)

workflow.add_node("model_get_schema", lambda state:{"messages":[model_get_schema.invoke(state["messages"])]})
workflow.add_node("query_gen", query_gen_node) # 쿼리 생성 노드
workflow.add_node("correct_query", model_check_query ) # 쿼리 실행전 모델로 점검하는 노드 
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool])) # 쿼리 실행 노드


workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

app = workflow.compile(checkpointer=MemorySaver())

from IPython.display import Image

img = Image(app.get_graph(xray=True).draw_mermaid_png())
with open("sql_graph.png", "wb") as f:
    f.write(img.data)

#############################################################################################

#### 그래프 실행 ####
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid, invoke_graph, stream_graph
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError


def run_graph(
    message: str, recursive_limit: int = 30, node_names=[], stream: bool = False
):
    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(
        recursion_limit=recursive_limit, configurable={"thread_id": random_uuid()}
    )

    # 질문 입력
    inputs = {
        "messages": [HumanMessage(content=message)],
    }

    try:
        if stream:
            # 그래프 실행
            stream_graph(app, inputs, config, node_names=node_names)
        else:
            invoke_graph(app, inputs, config, node_names=node_names)
        output = app.get_state(config).values
        return output
    except GraphRecursionError as recursion_error:
        print(f"GraphRecursionError: {recursion_error}")
        output = app.get_state(config).values
        return output


# output = run_graph(
#     "Andrew Adam 직원의 인적정보를 조회해줘",
#     stream=False,
# )

output = run_graph(
    "2009년도에 어느 국가의 고객이 가장 많이 지출했을까요? 그리고 얼마를 지출했을까요? 한글로 답변하세요.",
    stream=False,
)
print(output)
    # 출력 결과
    # Answer : 2009년도에 미국(USA) 고객이 가장 많이 지출했으며, 총 지출 금액은 103.95입니다.



