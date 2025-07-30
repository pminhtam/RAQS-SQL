"""
Componet in pipelines for SpiderBird dataset.

Why need: because SpiderBird dataset need output top-1 query, so we need to use a different pipeline to generate the query.


"""

from .pipeline_gen_one_query import GenOneSQL
from .pipeline_gen_many_queries import GenManySQL
from .pipeline_rewrite_queries import ReWriteSpiderbird
from .pipeline_select_final_query import SelectFinal