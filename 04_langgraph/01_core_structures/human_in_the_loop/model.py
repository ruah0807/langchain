from ollama import Client
from ollama import AsyncClient
from langchain_ollama import ChatOllama
import logging
import asyncio

LLAMA_MODEL = "llama3.1:latest"

class OllamaLLM:
    def __init__(self):
        self.client = Client(
            host="http://localhost:11434",
            headers={"Content-Type": "application/json"}
        )
        self.tools = []
        self.logger = logging.getLogger(__name__)

    def invoke(self, question):
        try:
            # question을 messages 형식으로 변환
            messages = [{"role": "user", "content": question}]

            response = self.client.chat(
                model=LLAMA_MODEL,
                messages=messages
            )
            if response.get("error"):
                raise ValueError(response["error"])

            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error during API call: {e}")
            print(f"Error during API call: {e}")
            raise

    def bind_tools(self, tools):
        self.tools = tools
        # 도구 사용 로직 추가
        # print(f"Tools bound: {tools}")
        return self
    





# def ollama_client():
#     client =Client(
#         host="http://localhost:11434",
#         headers = {"Content-Type": "application/json"}
#     )
#     return client

# async def ollama_llm(messages):
#     client = ollama_client()
#     response = client.chat(
#         model=LLAMA_MODEL,
#         messages=messages
#     )
#     return response.message.content
#     # print(response["message"]["content"])
#     # print(response.message.content)



# async def stream_ollama_api(question:str):
   
#     messages = {'role':'user', 'content':question}
#     async for chunk in await AsyncClient().chat(
#         model=LLAMA_MODEL, 
#         messages=[messages], 
#         stream=True
#         ):
#         # print(chunk["message"]["content"], end="", flush=True)
#         print(chunk.message.content, end="", flush=True)


# if __name__ == "__main__":
#     # print(query_ollama_api("hello"))
#     asyncio.run(stream_ollama_api("hello"))

# # source : https://github.com/ollama/ollama-python