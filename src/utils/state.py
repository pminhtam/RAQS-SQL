
from typing import Dict, TypedDict, Callable, List, Any, Optional
from .item import Item

"""
GraphState là class để tạo ra object state cho pipeline tạo ra ở `workflow_builder.py`

Các giá trị lưu ở trong GraphState sẽ được truyền qua các tool trong pipeline.

"""

### Graph State ###
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """
    # keys: Dict[str, any]
    keys: Optional[Dict[str, Any]] = None
    item: Item
    pipeline_log: List[Dict[str, Any]] = [] # Store logs of each step in the pipeline


