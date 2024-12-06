from dotenv import load_dotenv

load_dotenv()

from fewshot_example import system_prompt, few_examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
import torch, json, os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document  # Document 객체를 가져옵니다.

bgeEMBED_MODEL = "BAAI/bge-m3"

fewShot_DB = "/Users/ruahkim/coding/langchain/test/fewshot/db"


# pip install -U FlagEmbedding
def get_embeddings():
    if torch.backends.mps.is_available():
        embeddings = HuggingFaceEmbeddings(
            model_name=bgeEMBED_MODEL,
            # model_kwqrgs={"device": "mps"},
            encode_kwargs={"normalize_embeddings": False},
        )
    return embeddings


embeddings = get_embeddings()


from langchain.vectorstores import Chroma


# def get_vector():

#     # 각 예시를 Document 객체로 변환
#     document_objects = [
#         Document(page_content=example["input"], metadata={"output": example["output"]})
#         for example in few_examples
#     ]
#     # Chroma 데이터베이스 생성
#     vector = Chroma.from_documents(
#         documents=document_objects,  # Document 객체 리스트
#         collection_name="fewshot_DB",  # 컬렉션 이름
#         embedding=embeddings,  # 임베딩 함수
#         persist_directory=fewShot_DB,  # 데이터베이스 저장 경로
#     )
#     vector_db = vector.persist()
#     return vector_db


# vector = get_vector()


def get_llm():
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성
        model_name="gpt-4",  # 모델명
    )
    return llm


def get_fewshot(question):
    try:
        llm = get_llm()
        # print(f"embeddings : {embeddings}")
        # print(f"llm : {llm}")

        chroma = Chroma(
            # collection_name="fewshot_DB",
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

        example_selector = SemanticSimilarityExampleSelector.from_examples(
            few_examples,
            embeddings,
            # 초기화 된 vector db
            chroma,
            # 생성예시 갯수
            k=2,
        )
        # selected_example = example_selector.select_examples({"input": question})
        # print(f"selected_example : {selected_example}")

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )
        print(f"fewshot_prompt : {few_shot_prompt}")

        final_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), few_shot_prompt, ("user", "{input}")]
        )

        print(f"Selected FEW SHOT : {few_shot_prompt}")

        chain = final_prompt | llm | StrOutputParser()

        # stream_answer = chain.stream(question)
        # answer = stream_response(stream_answer)

        answer = chain.invoke(question)

        return answer
    except Exception as e:
        print(f"Error initializing Chroma: {e}")
        return


# question = "AWC의 GGGI ODA 협력사업 현황에 대해 알고 싶습니다."
question = "이 비즈니스에 어떤 사람들이 참여했나요?"
# question = "뉴클락시티에 대해 알려주세요"
get_few = get_fewshot(question)

print(get_few)
