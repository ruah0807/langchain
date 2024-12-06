from langchain_chroma import Chroma

# from fewshot_example import few_examples, system_prompt
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_teddynote.messages import stream_response

load_dotenv()

fewShot_DB = "/Users/ruahkim/coding/langchain/test/fewshot/db"
bgeEMBED_MODEL = "BAAI/bge-m3"

few_examples = [
    {
        "input": "2024년 타지키스탄 국제협력업무 회장은 누구야?",
        "output": """
        {
            "valid": true,
            "year_importance": 8,
            "person_importance": 8,
            "past_importance": 0,
            "other_importance": 0,
            "which_year": 2024,
            "related_list_of_people": false,
            "other_q": ["2024년 타지키스탄 국제협력의 주요 인물은 누구인가요?", "2024년 타지키스탄 국제협력의 성과는 무엇인가요?"]
        }
        """,
    },
    {
        "input": "Guillaume Prudent-Richard라는 인물에 대해서 설명해주세요.",
        "output": """
        {
            "valid": true,
            "year_importance": 0,
            "person_importance": 10,
            "past_importance": 0,
            "other_importance": 5,
            "which_year": 0,
            "related_list_of_people": false,
            "other_q": ["Guillaume Prudent-Richard는 누구인가요?"]
        }
    """,
    },
    {
        "input": "뉴클락시티에 대해 알려주세요",
        "output": """
        {
            "valid": true,
            "year_importance": 0,
            "person_importance": 0,
            "past_importance": 5,
            "other_importance": 10,
            "which_year": 0,
            "related_list_of_people": false,
            "other_q": ["뉴클락시티에 대한 정보를 알려주세요"]
        }
    """,
    },
    {
        "input": "AWC의 GGGI ODA 협력사업 현황에 대해 알고 싶습니다.",
        "output": """
        {
            "valid": true,
            "year_importance": 0,
            "person_importance": 0,
            "past_importance": 5,
            "other_importance": 10,
            "which_year": 0,
            "related_list_of_people" : false,
            "other_q":[
                "비슷한 질문 1",
                "비슷한 질문 2"
            ]
        }
        """,
    },
]

########################################################
system_prompt = """
### Please do not reply except in json format.
###You are in charge of understanding the intent of the user's question before the 'K-Water' business-related chatbot responds.
The The first thing to understand is that if the user asks a question that is completely unrelated to the business, such as asking about the weather, greetings, or asking who you are (ai), the 'valid' object will always return false.
---
### Additional conditions:
- If the input consists of meaningless symbols, such as "...", ".", or any other single or repeated punctuation marks, the 'valid' object must always return false.
- If the input consists of casual greetings, such as "안녕", "Hello", or other short phrases without a clear intent, the 'valid' object must always return false.
- If the input is an incomplete or ambiguous sentence that does not provide sufficient context for analysis, the 'valid' object must return false.
---
### K-water는 공적개발원조(ODA) 사업을 통해 라오스, 인도네시아, 필리핀, 방글라데시, 우즈베키스탄, 캄보디아 등 국내에서 축적한 물 관리 기술과 경험을 바탕으로 해외에서도 다양한 사업을 전개하고 있습니다.
### There are several questions related to k-water. You judge the questions below and return 'True' to the 'valid' object.
    1. People related to overseas or domestic business
    2. Past performance and achievements of k-water recorded for 30 years(including domestic and overseas)
    3. Questions about records and reports by year(including domestic and overseas)
    4. Questions about records and reports without years(including domestic and overseas)
    - ALL OF THESE QUESTIONS ARE YOU NEED TO FIND ANSWER FROM SOMEWHERE AND IT IS 'TRUE' TO THE 'VALID' OBJECT.
---
### Analyze the question and provide an importance score for each element (year, person, past, other) based on the relevance of each element in the question.
        If there does not fall under the three element(year, name, past), then give score to 'other' impotance.
        The importance score should be an integer between 0 and 10, with 10 indicating the highest relevance.
    - "valid": "valid": Returns 'True' for any question explicitly related to K-water's business, such as domestic or overseas projects, records, achievements, or personnel involved. Returns 'False' only for questions clearly unrelated to K-water's business, such as greetings, weather, not the question of sentence, or questions about the AI itself. Any question below that can be scored on 'IMPORTANCE' must always return 'TRUE'.
    - “year_importance”: Put the points 0 to 10. Assign a high importance score if the question contains an year.
            For example, if the question related to the year.
    - 'person_importance': Put the points 0 to 10. Give high importance score if the question is related to person not organization.
            For example, Assign a low importance score, just mentions 'OECD' or 'K-water'. Any question as a contact point or information.
    - "past_importance": Put the points to 10. Assign a high importance score if the question relates to overseas project or achievement that does not include years.
    - 'other_importance': Put the points to 10. The question is not relate to above element (year, person, past).
    - "which_year" : If it has specific years in the question. Put the year.
    - "related_list_of_people" : Gives 'True' if the user wants to recive the list of people's information. Other's 'False'.
    - "other_q : If your question has a clear year, be sure to include it in every question to make it a good question again. Otherwise, just make it a good question. Please write 1-3 questions.
### RESPONSE FORMAT (JSON):
    {{
        "valid": ((bool)true or false),
        "year_importance": ((int) only 0-10 points),
        "person_importance": ((int) only 0-10 points),
        "past_importance": ((int) only 0-10 points),
        "other_importance: ((int) only 0 or 10 points),
        "which_year": ((int) 4-digits year, or 0 if none mentioned),
        "related_list_of_people" : ((bool)true or false),
        "other_q":(list(str) Generate better questions for retrieval in Korean)
    }}

    Warning:  The importance scores must all be unique values and cannot be the same for any two elements.
"""

# def debug_few_examples(few_examples):
#     try:
#         for example in few_examples:
#             print(f"Example: {example}")
#             if not isinstance(example, dict):
#                 print(
#                     f"Error: Example is not a dictionary. Found type: {type(example)}"
#                 )
#             elif "input" not in example or "output" not in example:
#                 print("Error: Example does not contain 'input' or 'output' keys.")
#     except Exception as e:
#         print(f"Error while debugging few_examples: {e}")


# debug_few_examples(few_examples)


# pip install -U FlagEmbedding
def get_embeddings():
    if torch.backends.mps.is_available():
        embeddings = HuggingFaceEmbeddings(
            model_name=bgeEMBED_MODEL,
            # model_kwqrgs={"device": "mps"},
            encode_kwargs={"normalize_embeddings": False},
        )
    # else:
    #     embeddings = HuggingFaceEmbeddings(
    #         model_name=bgeEMBED_MODEL,
    #         model_kwargs={"device": "cpu"},  # CPU 사용
    #         encode_kwargs={"normalize_embeddings": True},
    #     )
    return embeddings


def get_llm():
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성
        model_name="gpt-4",  # 모델명
    )
    return llm


def get_vector():
    vector = Chroma.add_texts(
        texts=few_examples,
    )


def get_fewshot(question):
    try:
        embeddings = get_embeddings()
        llm = get_llm()
        print(f"embeddings : {embeddings}")
        print(f"llm : {llm}")

        chroma = Chroma(
            collection_name="fewshot_DB",
            embedding_function=embeddings,
            persist_directory=fewShot_DB,
        )
        print(f"CHROMA : {chroma}")

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "{input}"),
                ("ai", "{output}"),
            ],
        )
        print(f"example_prompt : {example_prompt}")

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # # 선택가능한 예시목록
            few_examples,
            # 의미적 유사성을 측정하는데 사용되는 임베딩을 생성하는 임베딩 클래스
            embeddings,
            # 임베딩 저장 및 유사성 검색 수행에 사용되는 vectorStore 클래스
            chroma,
            # input_keys=["input"],
            # 생성예시 갯수
            k=2,
        )
        print(f"example_selector : {example_selector}")

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )
        print(f"fewshot_prompt : {few_shot_prompt}")

        final_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), few_shot_prompt, ("user", "{input}")]
        )
        chain = final_prompt | llm

        stream_answer = chain.stream(question)

        answer = stream_response(stream_answer)

        # selected_example = example_selector.(question)

        return answer
    except Exception as e:
        print(f"Error initializing Chroma: {e}")
        return


question = "AWC의 GGGI ODA 협력사업 현황에 대해 알고 싶습니다."
get_few = get_fewshot(question)

print(get_few)
