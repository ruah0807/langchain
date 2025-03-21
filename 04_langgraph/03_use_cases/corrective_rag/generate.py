
############# 답변 생성 체인 #########

# 답변 생성 체인은 검색된 문서를 기반으로 답변을 생성하는 체인이다.

# 우리가 알고 잇는 일반적인 Naive RAG 체인임.

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from retrieval import MODEL_NAME, docs

# LangChain Hub 에서 RAG 프롬프트를 가져와 사용
prompt = hub.pull("teddynote/rag-prompt")

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(
        [
            f"""<document>
                <content>{doc.page_content}</content>
                <source>{doc.metadata["source"]}</source>
                <page>{doc.metadata["page"] + 1}</page>
            </document>"""
            for doc in docs
        ]
    )

# 체인 생성
rag_chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    question = "삼성전자가 개발한 생성 AI 에 대해 설명하세요."
    # 체인 실행 및 결과 출력 
    generation = rag_chain.invoke({"context": format_docs(docs), "question" : question})
    print(generation)

