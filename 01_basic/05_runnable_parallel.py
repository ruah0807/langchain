from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# API KEY를 환경변수로 관리하기 위한 설정파일
from dotenv import load_dotenv

# API 정보 로드
load_dotenv()
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# .env 파일에 LANGCHAIN_API_KEY를 입력합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("langchain_test")


####################################################################

### Runnable Parallel ###


# RunnableParallel 인스턴트 생성. 이 인스턴스는 여러 Runnable 인스턴스를 병렬로 실행가능.
runnable = RunnableParallel(
    # RunnablePassthrough 인스턴스를 'passed'키워드 인자로 전달. - 입력된 데이터를 그대로 통과시키는 역할.
    passed = RunnablePassthrough(),
    # 'extra' 키워드 인자로 runnablePassthrough.assign을 사용하여 'mult'람다 함수를 할당.  - 이 함수는 입력된 딕셔너리의 'num'키에 해당하는 값을 3배로 증가.
    extra = RunnablePassthrough.assign(mult = lambda x: x['num'] * 3),
    # 'modified' 키워드 인자로 람다함수 전달 - 입력된 딕셔너리의 'num'키에 해당하는 값에 1을 더함.
    modified= lambda x:x['num'] + 1
)

# runnable 인스턴스에 {'num' : 1 } 딕셔너리를 입력으로 전달하여 inboke 메소드를 호출.
# print(runnable.invoke({"num": 1}))

####################################################################


# Chain 도 RunnableParallel 을 적용가능.
chain1 = (
    {"country": RunnablePassthrough()} 
    | PromptTemplate.from_template("{country}의 수도는?")
    | ChatOpenAI() 
    | StrOutputParser()
)

chain2 = (
     {"country": RunnablePassthrough()} 
    | PromptTemplate.from_template("{country}의 면적은?")
    | ChatOpenAI() 
    | StrOutputParser() #  AIMessage(content='서울특별시입니다.', ...) <- content(LLM 응답) 부분만 가져옴.
)

combined_chain = RunnableParallel(capital = chain1, area = chain2)
# print(combined_chain.invoke("대한민국"))


####################################################################


from datetime import datetime
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

### Runnable Lambda ###

# Runnable Lambda 를 사용하여 코딩을 함수로 패키징 해놓고 결과값을 프롬프트에 입력할수 있음.

def get_today(a):
    # 오늘 날짜 가져오기
    print("입력 받은 변수 a의 값 : ", a)
    print(f"입력받은 n의 값: {a['n']}")
    return datetime.today().strftime("%b-%d")



# prompt 와 llm 생성
prompt = PromptTemplate.from_template(
    "{today}가 생일인 유명인 {n} 명을 나열하시오. 생년월일을 표기해 주 세요"
)
llm = ChatOpenAI(temperature = 0)

chain = (
                                          ### itemgetter : 딕셔너리 내의 "n"에 할당된 value값을 불러옴 
    {"today": RunnableLambda(get_today), "n": itemgetter("n")}
    | prompt
    | llm
    | StrOutputParser()
)

# print(chain.invoke({"n":3})) # '3'이라는 값이 RunnablePassthrough로 바로 들어가게됨.



####################################################################



from operator import itemgetter

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import ChatOpenAI

# 문장의 길이를 반환하는 함수
def length_function(text):
    return len(text)

# 두 문장의 길이를 곱한 값을 반환하는 함수
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

# _multiple_length_function 함수를 사용하여 두 문장의 길이를 곱한 값을 반환하는 함수
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("{a} + {b} 는 무엇인가요?")
model = ChatOpenAI()


# chain1 = prompt | model

chain = (
    {
        "a": itemgetter("word1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")}
        | RunnableLambda(multiple_length_function)
    }
    | prompt 
    | model
)

print(chain.invoke({"word1": "hello", "word2": "world"}))