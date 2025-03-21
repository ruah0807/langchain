from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
MODEL_NAME = "gpt-4o"

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


######################## Query 재작성

# 쿼리 재작성은 웹 검색을 최적화하기 위해 질문을 재작성 하는 단계이다.

query_rewrite_prompt = """You are a question rewriter that converts an input question to a better version that is optimized
for web search. Look at the input and try to reason about the underlying intent / meaning.
"""

# 프롬프트 정의 
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_rewrite_prompt),
        (
            "human",
            "Here is the initial question : \n\n {question} \n Formulate an improved question"
        )
    ]
)
# question re-writer 초기화
question_rewriter = rewrite_prompt | llm | StrOutputParser()

question = "삼성전자가 개발한 생성 AI 에 대해 설명하세요."
if __name__ == "__main__":

    question_invoke = question_rewriter.invoke({"question": question})
    
    print("[Rewritten Qustion] :", question_invoke)

    # 출력 결과 
    # [Rewritten Qustion] : 삼성전자가 개발한 생성 AI의 특징과 기능은 무엇인가요?




######################## 웹 검색 도구 ########################

from langchain_teddynote.tools.tavily import TavilySearch
# 최대 검색 결과를 3으로 설정
web_search_tool = TavilySearch(max_results=3)



if __name__ == "__main__":
    # 웹 검색 도구 실행
    results = web_search_tool.invoke({"query": question})

