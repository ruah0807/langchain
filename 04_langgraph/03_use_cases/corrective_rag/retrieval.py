import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _rag.pdf import PDFRetrievalChain

# pdf 문서로드
pdf = PDFRetrievalChain(
    [os.path.join(os.getcwd(),"../_docs/SPRI_AI_Brief_2023년12월호_F.pdf")]
    ).create_chain()

pdf_retriever = pdf.retriever
# Retrieval Chain 을 생성
pdf_chain = pdf.chain


#### 검색된 문서의 관련성 평가 (Question-Retrieval Evaluation)

# 검색된 문서의 관련성 평가는 검색된 문서가 질문과 관련이 있는지 여부를 평가하는 단계이다.
# 먼저 검색된 문서를 위한 평가기 (retreival-grader) 를 생성한다.

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

MODEL_NAME = "gpt-4o"

# 검색된 문서의 관련성 여부를 이진 점수로 평가하는 데이터 모델
class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved document."""

    # 문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no' 로 평가
    binary_score: str = Field(
        description= "Documents are relevant to the question, 'yes' or 'no'"
    )

# llm 초기화
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# GradeDocuments 데이터 모델을 사용하여 구조화도니 출력을 생성하는 LLM
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 시스템 프롬프트 정의
system_prompt = """You are a grader assessing relevance of a retrieved document to a suser question.\n
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n
Give a binary score 'yes' or 'no' score to  indicate whether the document is relevant to the question."""

# 채팅 프롬프트 탬플릿 생성
grade_prompt = ChatPromptTemplate.from_messages(
[
    ("system", system_prompt),
    ("human", "Retrieved document : \n\n{document}\n\nQuestion : {question}"),
]
)

# Retrieval 평가기 초기화
retrieval_grader = grade_prompt | structured_llm_grader

# retrieval_grader 를 사용해서 문서를 평가

# 문서의 집합이 아닌 1개의 단일문서에 대한 평가를 수행.
# 결과는 단일 문서에 대한 관련성 여부가 (yes/no)로 반환

# 질문 정의
question = "삼성전자가 개발한 생성 AI 에 대해 설명하세요."
docs = pdf_retriever.invoke(question)

if __name__ == "__main__":
    # 문서 검색
    # 검색된 문서 중 1번 index 문서의 페이지 내용을 추출
    doc_txt = docs[5].page_content
    # 검색된 문서와 질문을 사용하여 관련성 평가를 실행하고 결과 출력
    print(retrieval_grader.invoke({"question": question, "document":doc_txt}))






