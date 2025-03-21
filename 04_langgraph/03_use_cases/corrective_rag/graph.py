import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from retrieval import pdf_retriever, retrieval_grader
from rewrite_query import question_rewriter, web_search_tool
from generate import rag_chain
load_dotenv()



from typing import Annotated, List
from typing_extensions import TypedDict

class State(TypedDict):
    question : Annotated[str, "Teh question to answer"]
    generation : Annotated[str, "The generation from the LLM"]
    web_search : Annotated[str, "Whether to add search"]
    documents : Annotated[List[str], "The documents retrieved"]


#### Node

def retrieve(state : State):
    print("\n ============ Retrieve ============ \n")
    question = state["question"]
    # rag 사용한 답변 생성
    documents = pdf_retriever.invoke(question)

    return {"documents" : documents}

def generate(state: State) :
    print("\n ============ Generate ============ \n")
    question = state["question"]
    documents = state["documents"]

    # RAG를 사용한 답변 생성
    generation = rag_chain.invoke({"context" : documents, "question" : question})

    return {"generation" : generation}
# 문성 평가 노드 -> 관련성 있는 문서
# 
# 만 필터링
def grade_documents(state: State):
    print("\n ============ Check Document Relevance to question ============ \n")
    question = state["question"]
    documents = state["documents"]

    # 필터링된 문서
    filtered_docs = []
    relevant_doc_count = 0

    # 관련성 있는 문서만 필터링
    for d in documents:
        # Question-docuent 의 관련성 평가
        score = retrieval_grader.invoke(
            {"question": question, "document":d.page_content}
        ) 
        grade = score.binary_score

        if grade == "yes":
            print("======= Grade : Document Relevant =======")
            # 관련있는 문서를 filtered_docs에 추가
            filtered_docs.append(d)
            relevant_doc_count += 1
        else:
            print("======= Grade : Document Not Relevant =======")
            continue

    # 관련 문서가 3개 이하면 웹 검색 수행
    web_search = "Yes" if relevant_doc_count == 0 else "No"
    return {"documents":filtered_docs, "web_search" : web_search}


# 쿼리 재작성 노드
def query_rewrite(state: State):
    print("\n ============ Query Rewrite ============ \n")
    question = state["question"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})

    return {"question" : better_question}

# web search 노드
def web_search(state: State):
    print("\n ============ Web Search ============ \n")
    question = state["question"]
    documents = state["documents"]

    # 웹 검색 수행
    docs = web_search_tool.invoke({"query":question})

    # 검색 결과를 문서 형식으로 변환
    web_results = "\n".join(
        [d["content"] for d in docs]
    )

    documents.append(web_results)

    return {"documents" : documents}



############################ 조건부 엣지에 활용할 함수

# decide_to_generate 함수는 관련성 평가를 마친뒤, 웹 검색 여부에 따라 다음 노드로 라우팅하는 역할을 수행한다.

# web_search 가 yes 인 경우 : query_rewrite 노드에서 쿼리를 재작성한 뒤 웹 검색을 수행.
# 만일, 

def decide_to_generate(state: State):
    # 평가된 문서를 기반으로 다음 단계 결정
    print("\n ============ Assess Graded documents ============ \n")
    
    # 웹 검색 필요 여부
    web_search = state["web_search"]

    if web_search == "Yes":
        # 웹 검색으로 정보 보강이 필요한 경우
        print(
            "======= Dicision : Documents are NOT relevant to question, query rewrite ======"
        )
        return "query_rewrite"
    else:
        # 관련 문서가 존재하므로 답변 생성 단계(generate)로 진행
        print("====== Dicision : Generate ========")
        return "generate"
    


######################## 그래프 생성 ########################
from langgraph.graph import END, StateGraph, START

# 그래프 상태 초기화
workflow = StateGraph(State)

# 노드 정의
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("query_rewrite", query_rewrite)
workflow.add_node("web_search_node", web_search)


workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "query_rewrite": "query_rewrite",
        "generate": "generate"
    }
)
workflow.add_edge("query_rewrite", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

########## 그래프 실행 ##############
if __name__ == "__main__":
    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import invoke_graph, random_uuid    
    # config =- Runnableconfig
    config = RunnableConfig(
        recursion_limit=20, 
        configurable={"thread_id": random_uuid()}
        )

    input = {"question" : "삼성전자가 개발한 생성형 AI의 이름은?"}
    # input = {"question" : "2024년 노벨 문학상 수상자의 이름은?"}
    # 그래프 실행
    for chunk in app.stream(input, config, stream_mode="values"):
        for k, v in chunk.items():
            print(f"=================== {k} : \n{v}\n\n")

