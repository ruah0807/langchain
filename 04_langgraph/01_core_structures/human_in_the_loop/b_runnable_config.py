from typing import Any, Optional
from typing_extensions import TypedDict
import uuid
class RunnableConfig(TypedDict, total=False):
    """Configuration for a Runnable."""

    tags: list[str]
    """
    이 호출 및 모든 하위 호출(예: LLM을 호출하는 체인)에 대한 태그입니다.
    이를 사용하여 호출을 필터링할 수 있습니다.
    """

    metadata: dict[str, Any]
    """
    이 호출 및 모든 하위 호출(예: LLM을 호출하는 체인)에 대한 메타데이터입니다.
    키는 문자열이어야 하며, 값은 JSON 직렬화 가능해야 합니다.
    """

    # callbacks: Callbacks
    # """
    # 이 호출 및 모든 하위 호출(예: LLM을 호출하는 체인)에 대한 콜백입니다.
    # 태그는 모든 콜백에 전달되며, 메타데이터는 handle*Start 콜백에 전달됩니다.
    # """

    run_name: str
    """
    이 호출에 대한 트레이서 실행의 이름입니다. 기본값은 클래스의 이름입니다.
    """

    max_concurrency: Optional[int]
    """
    만들 수 있는 최대 병렬 호출 수입니다. 제공되지 않으면 ThreadPoolExecutor의 기본값을 사용합니다.
    """

    recursion_limit: int
    """
    호출이 재귀할 수 있는 최대 횟수입니다. 제공되지 않으면 기본값은 25입니다.
    """

    configurable: dict[str, Any]
    """
    이 Runnable 또는 하위 Runnable에서 .configurable_fields() 또는 .configurable_alternatives()를 통해
    이전에 구성 가능하게 만든 속성에 대한 런타임 값입니다.
    구성 가능하게 만든 속성에 대한 설명은 .output_schema()를 확인하세요.
    """

    run_id: Optional[uuid.UUID]
    """
    이 호출에 대한 트레이서 실행의 고유 식별자입니다. 제공되지 않으면 새 UUID가 생성됩니다.
    """