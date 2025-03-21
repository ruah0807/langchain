### Research Team 도구 설정

# 웹에서 정보를 찾기 위해 검색 엔진과 URL 스크래퍼를 사용. 

from typing import List
from langchain_community.document_loaders import WebBaseLoader # 웹 스크래핑 도구
from langchain_community.tools import TavilySearchResults # 웹 검색 도구
from langchain_core.tools import tool

# 검색 도구 정의 (TavilySearchResults)
@tool
def search_web(query: str) -> str:
    """Tavily Web Search"""
    tavily_tool = TavilySearchResults(
        max_results=5
    )
    return tavily_tool.invoke({"query": query})

# 웹 페이지에서 세부 정보를 스크래핑 하기 위한 도구 정의
@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information"""
    # 텍스트 정리 함수: 줄바꿈과 탭을 공백으로 대체
    def clean_text(text):
        # 줄바꿈과 탭을 공백으로 대체
        cleaned_text = text.replace('\n', ' ').replace('\t', ' ')
        # 여러 개의 공백을 하나의 공백으로 줄임
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text
        
    # 주어진 url  목록을 사용하여 웹 페이지 로드
    loader = WebBaseLoader(
        web_path=urls,
        header_template = {
            # 정적 웹사이트만 가능. 동적 웹사이트는 스크래핑 불가(로그인, 자바스크립트 등이 포함된 웹사이트.)
            # 웹 페이지 로드 시 헤더 정보 설정 : 웹서버 차단 방지, 브라우저로 인식하도록 호환성유지 등.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        }
    )
    docs = loader.load()

    return "\n\n".join(
        [
            f'<Document name = "{doc.metadata.get("title","")}"> {clean_text(doc.page_content)}</Document>'
            for doc in docs
        ]
    )