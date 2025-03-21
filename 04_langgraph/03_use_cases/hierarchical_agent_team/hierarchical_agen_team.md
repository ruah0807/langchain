# 계층적 에이전트 팀

해당 튜토리얼에서는 계층적 에이전트 팀을 구성하는 방법을 살펴본다.

단일 에이전트나 단일 수준의 supervisor(감독자)로는 대응하기 힘든 복잡한 작업을 **계층적(hierarchial)**구조를 통해 분할하고, 각각의 하위 수준 감독자가 해당 영역에 특화된 작업자 에이전트를 관리하는 방식을 구현한다.

이러한 계층적 접근 방식은 작업자가 너무 많아질 경우나, 단일 작업자가 처리하기 힘든 복잡한 작업을 효율적으로 해결하는데 도움이 된다.

Idea Source : [AutoGen 논문](https://arxiv.org/abs/2308.08155)



![Hierarchial Agent Team Flow](https://raw.githubusercontent.com/teddylee777/langchain-kr/b67071a05cf3b3bfd18eca7d8892a3cda335e039/17-LangGraph/03-Use-Cases/assets/langgraph-multi-agent-team-supervisor.png)


## Why?

- 작업의 복잡성 증가 : 단일 supervisor로는 한번에 처리할 수 없는 다양한 하위 영역의 전문 지식 필요
- 작업자 수 증가 : 많은 수의 작업자를 관리할 때, 단일 supervisor가 모든 작업자에게 직접 명령을 내리면 관리 부담이 커짐.

## 다룰 내용
1. 도구 생성 : web research 및 documentation을 위한 에이전트 도구 정의
2. 에이전트 팀 정의 : 연구 팀 및 문서 작성팀을 계층적으로 정의하고 구성
3. 계층 추가 : 상위 수준 그래프와 중간 수준 감독자를 통해 전체 작업을 계층적으로 조성
4. 결합 : 모든 요소를 통합하여 최종적인 계층적 에이전트 팀 구축

[Multi-Agent systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)

