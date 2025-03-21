### 문서 작성 팀 도구
# 에이전트가 파일 시스템에 접근할수있도록 한다. (안전하지 않을 수 있기 때문에 사용에 주의 필요.)

from pathlib import Path
from typing import Dict, Optional, List
from typing_extensions import Annotated
from langchain_core.tools import tool

# 임시 디렉토리 생성 및 작업 디렉토리 설정
WORKING_DIRECTORY = Path("./temp_dir")

# tmp_dir 폴더가 없으면 생성
WORKING_DIRECTORY.mkdir(exist_ok=True)

# 아웃라인 생성 및 파일로 저장
@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections"],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file"]:
    
    """Create and save an outline."""
    
    # 주어진 파일 이름으로 아웃라인 저장.
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"    

# 문서 읽기
@tool
def read_document(
    file_name: Annotated[str, "File path to read the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"]= None
) -> str:
    
    """ Read the specified document."""
    print(f"\n\nfile_name: \n\n{file_name}\n\n")

    # 주어진 파일 이름으로 문서 읽기
    with (WORKING_DIRECTORY / file_name).open("r", encoding="utf-8") as file:
        lines = file.readlines()
        # 시작 줄이 지정되지 않은 경우 기본값 설정
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


# 문서 쓰기
@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name : Annotated[str, "File path to save the document."]
) -> Annotated [str, "Path of the saved document file."]:
    
    """Create and save a text document."""

    # 주어진 파일 이름으로 문서 저장
    with(WORKING_DIRECTORY / file_name).open("w", encoding="utf-8") as file:
        file.write(content)
    return f"Document Saved to {file_name}"


# 문서 편집
@tool
def edit_document(
    file_name : Annotated[str, "file path of the document to be edited"],
    inserts: Annotated[
        Dict[int,str],
        "Dictionary where key is the line number(1-indexed) and value is the text to be inserted at that line."
    ],
) -> Annotated[str, "File path of the edited document."]:
    
    """Edit a document by inserting text at specific line neumbers."""

    # 주어진 파일 이름으로 문서 읽기
    with(WORKING_DIRECTORY / file_name).open("r", encoding="utf-8") as file:
        lines = file.readlines()

    # 삽입할 텍스트를 정렬하여 처리
    sorted_inserts = sorted(inserts.items())

    # 지정된 줄 번호에 텍스트 삽입
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number -1, text + "\n")
        else:
            return f"Error : Line Number {line_number} is out of range."
        
    # 편집된 문서를 파일에 저장
    with (WORKING_DIRECTORY / file_name).open("w", encoding="utf-8") as file:
        file.writelines(lines)


    return f"Document edited and saved to {file_name}"


from langchain_experimental.tools import PythonREPLTool

# 코드 실행도구구 도구 생성
# 나중에 차트 generation할때 쓸 도구
python_repl_tool = PythonREPLTool()
    
