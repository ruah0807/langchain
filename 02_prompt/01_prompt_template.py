from langchain_core.runnables import RunnableParallel, RunnablePassthrough
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



from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

####################################################################

### Prompt Template 를 사용하는 방법

#### 방법 1. from_template() 메소드를 이용하여 PromptTemplate 객체 생성
# - 치환될 변수를 {변수} 로 묶어 템플릿 정의

from langchain_core.prompts import PromptTemplate 

# template 정의 . {country}는 변수로, 이후에 값이 들어갈 자리를 의미
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)

# prompt 생성 : Format 메소드를 이용하여 변수에 값을 넣어줌.
prompt = prompt.format(country= "대한민국")

# print(prompt)

#######

# chain = prompt | llm 

# print(chain.invoke("미국").content)


####################################################################

#### 방법 2.  PromptTemplate  객체 생성과 동시에 prompt 생성


# 추가 유효성 검사를 위해 input_variables 를 명시적으로 지정
# 이러한 변수는 인스턴스화 중에 템플릿 문자열에 있는 변수와 비교하여 불일치하는 경우 예외를 발생시킨다.

# template 정의
template = "뮤지컬 {musical}의 가장 유명한 넘버는 무엇인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template = template,
    input_variables=['musical']
)

# print(prompt.format(musical="위키드"))

# chain = prompt | llm 

# print(chain.invoke("위키드").content)


######

template = "뮤지컬 {musical1}과 {musical2}의 가장 유명한 넘버는 각각 무엇인가요?"


# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template = template,
    input_variables=['musical1'],
    partial_variables={
        "musical2": "지킬앤하이드"  # dictionary 형태로 partial_variables를 채워줌
    }
)
# chain = prompt | llm 

# print(prompt.format(musical1 = "라이온킹"))
# print(chain.invoke("라이온킹").content)


prompt = PromptTemplate.from_template(template)

prompt = prompt.partial(musical2 = "마틸다")    # PromptTemplate.from_template().partial()
chain = prompt | llm 

# print(chain.invoke("위키드").content)
# print(chain.invoke({"musical1": "킹키부츠", "musical2" : "레베카"}).content)

#####################################################################

### partial_variables : 부분변수 채움

# partial 을 사용하는 일반적 용도는 함수르 ㄹ부분적으로 사용하는것. 이 사용 사례는 항상 공통된 방식으로 가져오고 싶은 변수가 있는 경우
# 대표적인 예가 날짜나 시간.

# 항상 현재 날짜가 표시되기를 원하는 프롬프트가 있다 가정.
# 프롬프트에 하드코딩 할수도 없고, 다른 입력 변수와 함께 전달하는 것도 번거롭다. 이 경우에 항상 현재 날짜를 반환하는 함수를 사용하여 프롬프트를 부분적으로 변경할수 있으면 매우 편리함.


#####################################################################
# 오늘의 날짜를 구하는 코드

from datetime import datetime

# 날짜를 반환하는 함수 정의
def get_today():
    return datetime.now().strftime("%b %d")


prompt = PromptTemplate(
    template="오늘의 날짜는 {today}입니다. 오늘이 생일인 유명 팝가수 {n}명을 나열해주세요. 생년월일도 함께 표기하세요 ",
    input_variables=["n"],
    partial_variables={
        "today": get_today  # 함수 실행 대기 
    }
)

# prompt.format(n=5)
chain = prompt | llm
# print(chain.invoke(5).content)

# 오늘날짜를 강제로 넣고 실행
print(chain.invoke({"today": "Oct 31", "n": 2}).content)

