from typing import Optional, Any, Dict, List
from pydantic import BaseModel

class Task(BaseModel):
    """
    Represents a task with question and database details.

    Attributes:
        question_id (int): The unique identifier for the question.
        db_id (str): The database identifier.
        question (str): The question text.
        evidence (str): Supporting evidence for the question.
        SQL (Optional[str]): The SQL query associated with the task, if any.
        difficulty (Optional[str]): The difficulty level of the task, if specified.
    """
    question_id: int
    db_id: str
    question: str
    evidence: str
    extra_map: Optional[Dict[str, Any]] = {}
    extra_table_map: Optional[Dict[str, Any]] = {}
    SQL: Optional[str] = None
    difficulty: Optional[str] = None
    orig_query: Optional[str] = None
    orig_example_idx: Optional[Any] = None
    queries: Optional[List] = []
