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
# print(chain.invoke({"today": "Oct 31", "n": 2}).content)



#####################################################################


##### 파일로부터 template 읽어오기 #####

from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/fruit_color.yaml", encoding="utf-8")

prompt_with_variable = prompt.format(fruit="kakao")
# print(prompt_with_variable)

prompt2 = load_prompt("prompts/capital.yaml", encoding="utf-8")
# print(prompt2.format(country="Mexico"))


#####################################################################


####### ChatPromptTemplate #####

# ChatPromptTemplate은 대화 목록을 프롬프트로 주입하고자 할 때 활용 가능
# 메시지는 튜플(tuple)형식으로 구성하며, (role, message)로 구성하여 리스트로 생성 가능

### role
# "system": 시스템 설정 메시지. 주로 전역설정과 관련된 프롬프트
# "human": 사용자 입력 메시지
# "ai" : ai의 답변 메시지

from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template(
    "뮤지컬 {musical}의 주인공은 누구인가요?"
)

chat_prompt.format(musical="위키드")

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "당신은 뮤지컬전문가 AI입니다. 당신의 이름은 {name}입니다."),
        ("human", "안녕!"),
        ("ai", "안녕하세요, 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)
# chat message 를 생성
messages = chat_template.format_messages(
    name="Ryan", user_input="당신의 이름은 무엇입니까?"
)
# print(messages)

# 생성한 메시지를 바로 주입하여 결과 반환
message = llm.invoke(messages).content
# print(message)

# chain 생성
chain = chat_template | llm
message = chain.invoke(
    {"name": "Ryan", "user_input": "뮤지컬 위키드 주인공에 대해서 알려주실래요?"}
)

# print(message.content)


#####################################################################

##### Message Placeholder #####

# Langchain은 포맷하는 동안 랜더링할 메시지를 완전히 제어할 수있는 Message Placeholder를 제공
# 메시지 프롬프트 템플릿에 어떤 역할을 사용해야할지 확실하지 않거나, 서식 지정 중에 메시지 목록을 삽입하려는 경우 유용할 수 있다.

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 뮤지컬 배우 입시 전문 AI 선생님 입니다. 당신의 임무는 학생과 레슨 과정에서의 상황을 주요 키워드로 대화 요약을 하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "지금까지의 대화를 {word_count}단어로 요약합니다."),
    ]
)
# print(chat_prompt)

# conversation 대화 목록을 나중에 추가하고자 할때 MessagePlaceholder를 사용 가능

formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "어떻게 나한테 이럴수 있어요? 그럼 저는 괜찮을 거라 생각하세요????"),
        ("ai", "그게 아니지 ! 좀더 감정을 넣어야해. 진짜 괜찮지 않은 사람이어야해!"),
    ],
)

chain = chat_prompt | llm
messages = chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "어떻게 나한테 이럴수 있어요? 그럼 저는 괜찮을 거라 생각하세요????",
            ),
            (
                "ai",
                "그래 그거야 !!!!!! 감점을 더 넣어!!!!!!!!!",
            ),
        ],
    }
)
print(messages)

