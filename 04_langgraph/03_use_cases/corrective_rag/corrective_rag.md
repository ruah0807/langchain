# CRAG : Corrective RAG

> Corrective RAG(CRAG) 전략을 사용하여 RAG 기반 시스템을 개선하는 방법

CRAG는 검색된 문서들에 대한 자기반성(Self-reflection) 및 자기 평가(self-evaluation) 단계를 포함하여, 검색-생성 파이프라인을 정교하게 다루는 접근법이다.

![langgraph-crag](https://raw.githubusercontent.com/teddylee777/langchain-kr/1999da031d689326fc7db9596b4a29b10076e290/17-LangGraph/03-Use-Cases/assets/langgraph-crag.png)

## CRAG 란
- Corrective-RAG는 검색 과정에서 찾아온 문서를 평가, 지식을 정제(refine)하는 단계를 추가한 방법론이다.

- 생성에 앞서 검색 결과를 점검하고 필요하면 보조적인 검색을 수행.
- 최종적으로 품질 높은 답변을 생성하기 위한 일련의 프로세스를 포함.

### CRAG의 핵심 아이디어
![Corrective Retrieval Augmented Generation Paper Link](https://arxiv.org/pdf/2401.15884)

1. 검색된 문서 중 하나 이상이 사전 정의된 관련성 임계값(retrieval validation score)를 초과하면 생성 단계로 진입
2. 생성 전 지식 정제 단계 수행
3. 문서를 "knowledge strips"로 세분화. -> `k` :문석 검색 결과수
4. 각 지식 스트립을 관련성 여부(score)로 평가. -> 문서는 'chunk'단위로 평가
5. 문서가 관련성 임계값 이하이거나 평과 결과 신뢰도가 낮을 경우, 추가 데이터 소스(예: 웹검색)으로 보강.
6. 웹 검색을 통한 보강시, 쿼리 재작성(query-rewrite)을 통해 검색 겨로가 최적화


### 주요 내용
- LangGraph를 활용하여 CRAG 접근법의 일부 아이디어를 구현.
- 지식 정제 단계는 생략(필요하다면 노드로 추가할 수 있는 형태로 설계)
- 관련있는 문서가 하나도 없다면 웹검색 실행
- Tavily Search 사용, 검색 최적화를 위해 질문 재작성(Question Re-writing) 도입.

### 주요 단계 개요
- Retrieval Grader : 검색된 문서의 관련성 평가
- Generate : LLM을 통한 답변 생성
- Question Re-writer : 질문 재작성을 통한 검색 질의 최적화
- Web Search Tool : Tavily Search 를 통한 웹 검색 활용
- Create Graph : LangGraph를 통한 CRAG 전략 그래프 생성
- Use the graph : 생성된 그래프를 활용하는 방법.