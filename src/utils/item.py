from typing import Optional, Any, Dict, List
from pydantic import BaseModel

"""
Class chứa thông tin của một item bao gồm câu hỏi và thông tin về cơ sở dữ liệu.
Đây là object truyền vào khi bắt đầu chạy pipeline.

"""
class Item(BaseModel):
    """
    Represents a task with question and database details.
    Chứa thông tin của một item bao gồm câu hỏi và thông tin về cơ sở dữ liệu.
    Một item là 1 câu hỏi và thông tin về cơ sở dữ liệu.

    Attributes:
        question_id (int): The unique identifier for the question.
        db_id (str): The database identifier.
        question (str): The question text.
        evidence (str): Supporting evidence for the question.
        schema_text (str): The schema of the database. With text format.
    """
    question_id: Optional[int] = None
    db_id: str
    question: str
    evidence: Optional[str] = None
    schema_text_wo_content : str
    schema_text_with_content : str
    ori_query: Optional[str] = None
    amb_query: Optional[str] = None