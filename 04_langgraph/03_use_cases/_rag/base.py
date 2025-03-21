import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda
from abc import ABC, abstractmethod
from operator import itemgetter
from langchain import hub
from yaml import safe_load

class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 10

    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        return OpenAIEmbeddings(model="text-embedding-3-small")

    def create_vectorstore(self, split_docs):
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        # MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    

    def create_prompt(self):
        print(os.getcwd())
        # YAML 파일에서 프롬프트를 로드
        with open(os.path.join(os.getcwd(),"../prompt/rag-prompt.yaml"), "r") as file:
            prompt_config = safe_load(file)
        
        # PromptTemplate 객체 생성
        return PromptTemplate(
            template=prompt_config.get("template", ""),
            input_variables=prompt_config.get("input_variables", [])
        )

    # def create_prompt(self):
    #     return hub.pull("teddynote/rag-prompt-chat-history")
    

    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)
    


    def create_chain(self):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()

        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt  # PromptTemplate 객체를 직접 사용
            | model
            | StrOutputParser()
        )
        return self
    

if __name__ == "__main__":
    def test_create_prompt():
        retrieval_chain = RetrievalChain()
        prompt = retrieval_chain.create_prompt()
        print("Template:", prompt["template"])
        print("Input Variables:", prompt["input_variables"])

    # 테스트 실행
    test_create_prompt()