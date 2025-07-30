from abc import ABC, abstractmethod
from typing import Dict, Any
import re
import time
from .state import GraphState

"""

Tool là class abstract để tạo ra các tool trong pipeline.
Các tool là object thực hiện một bước nhất định trong pipeline.

Các file lưu dưới dạng `pipeline_*.py` là code chứa object kế thừa từ Tool
mỗi object này sẽ thực hiện một công việc nhất định trong pipeline.
Các object này ghép lại để tạo ra 1 pipeline hoàn chỉnh.
Ouput của object này sẽ là input của object tiếp theo.

Thực ra input và output của mỗi object là một object GraphState.
"""
class Tool(ABC):

    def __init__(self):
        self.tool_name = camel_to_snake(self.__class__.__name__)

    def __call__(self, state: GraphState) -> GraphState:
        # try:
        self._run(state)
        # except Exception as e:
        #     print(f"{type(e)}: <{e}>")
            # state.errors[self.tool_name] = f"{type(e)}: <{e}>"

        return state

    @abstractmethod
    def _run(self, state: GraphState) -> Dict:
        pass

def camel_to_snake(name):
    # Insert an underscore before each uppercase letter (excluding the first letter)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert an underscore before each uppercase letter followed by lowercase or digits
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    # Convert the entire string to lowercase
    return s2.lower()