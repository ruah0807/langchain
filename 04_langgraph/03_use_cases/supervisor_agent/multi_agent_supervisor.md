# Multi-Agent Supervisor

> Langgraph를 활용하여 다중 에이전트 시스템을 구축, 에이전트 간 작업을 효율적으로 조정하고 감독자를 통해 관리하는 방법
> 여러 에이전트를 동시에 다루며, 각 에이전트가 자신의 역할을 수행하도록 관리하고, 작업 완료시 이를 적절히 처리하는 과정.

Agent가 여러기로 늘어나고, 이들을 조정해야할 경우, 단순한 분기 로직만으로는 한계가 있다.

여기서는 [LLM을 활용한 Supervisor](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#network)를 통해 에이전트들을 관리하고, 각 에이전트 노드의 결과를 바탕으로 팀 전체를 조율하는 방법을 살펴본다.

### 중점 사항 : 
- supervisor 는 다양한 전문 에이전트를 한 데 모아 하나의 팀으로 운영하는 역할을 한다.
- supervisor 에이전트는 팀의 진행 상황을 관찰하고, 각 단계별로 적절한 에이전트를 호출하거나 작업을 종료하는 등의 로직을 수행한다.

![langgraph-multi-agent-supervisor](https://raw.githubusercontent.com/teddylee777/langchain-kr/1999da031d689326fc7db9596b4a29b10076e290/17-LangGraph/03-Use-Cases/assets/langgraph-multi-agent-supervisor.png)

### 다룰 내용
- 설정(Setup): 필요한 패키지 설치 및 API 키 설정 방법
- 도구 생성(Tool Creation): 웹 검색 및 플롯(plot) 생성 등, 에이전트가 사용할 도구 정의
- 도우미 유틸리티(Helper Utilities): 에이전트 노드 생성에 필요한 유틸리티 함수 정의
- 에이전트 감독자 생성(Creating the Supervisor): 작업자(Worker) 노드의 선택 및 작업 완료 시 처리 로직을 담은 Supervisor 생성
- 그래프 구성(Constructing the Graph): 상태(State) 및 작업자(Worker) 노드를 정의하여 전체 그래프 구성
- 팀 호출(Invoking the Team): 그래프를 호출하여 실제로 다중 에이전트 시스템이 어떻게 작동하는지 확인

이러한 과정에서 Langgraph 의 사전 구축된 [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/) 함수를 활용하여 각 에이전트 노드를 간소화한다.

이러한 고급 agent 사용방식은 LangGraph에서 특정 디자인 패턴을 시연하기 위한 것이고, 필요에 따라 다른 기본 패턴과 결합하여 최적의 결과를 얻을 수 있다. 