from typing import Dict, Any
from langgraph.graph import END, StateGraph
import logging
from src.pipelines import (InitialSQL, IntendDetectionDummySQL, SchemaLinkingLLMAPI, GenAmbiqtSQL,
                       EmbeddingSearch,RelationalClassificationPLM, ReWrite, SchemaLinkingPLM)
from src.pipelines.spiderbird_pipelines import GenOneSQL, GenManySQL, ReWriteSpiderbird, SelectFinal
from .state import GraphState

AGENT_CLASSES = {
    "initial_sql": InitialSQL,
    "intend_detection_dummy_sql": IntendDetectionDummySQL,
    "schema_linking_llm_api": SchemaLinkingLLMAPI,
    "schema_linking_plm": SchemaLinkingPLM,
    "gen_ambiqt_sql": GenAmbiqtSQL,
    "embedding_search": EmbeddingSearch,
    "relational_classification_plm": RelationalClassificationPLM,
    "rewrite": ReWrite,
    "gen_one_sql": GenOneSQL,
    "gen_many_sql": GenManySQL,
    "rewrite_spiderbird": ReWriteSpiderbird,
    "select_final": SelectFinal,
}


class WorkflowBuilder:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.workflow = StateGraph(GraphState)
        self.config = config
        logging.info("Initialized WorkflowBuilder")

    def build(self):
        agents = {agent_name: agent_config for agent_name, agent_config in self.config["agents"].items()
                  if agent_name in AGENT_CLASSES}
        self._add_agents(agents)
        agents_keys = list(agents.keys())
        self.workflow.set_entry_point(agents_keys[0])
        connections = [(agents_keys[i], agents_keys[i+1])
                       for i in range(len(agents_keys)-1)]
        connections += [(agents_keys[-1], END)]
        self._add_connections(connections)
        # import pdb; pdb.set_trace()

    def _add_agents(self, agents: Dict[str, Dict[str, Any]]) -> None:
        """
        Adds agents to the team.
        Là các node trong graph
        Node có thể là function hoặc class
        Nếu là class thì phải có method __call__ để chạy
        Khi chạy thì sẽ gọi method __call__ của class đó
        Args:
            agents (list): A list of agent names.
        """
        for agent_name, agent_config in agents.items():
            agent = AGENT_CLASSES[agent_name](**agent_config)
            self.workflow.add_node(agent_name, agent)
            logging.info(f"Added agent: {agent_name}.")


    def _add_connections(self, connections: list) -> None:
        """
        Adds connections between agents in the team.

        Args:
            connections (list): A list of tuples representing the connections.
        """
        for src, dst in connections:
            self.workflow.add_edge(src, dst)
            logging.info(f"Added connection from {src} to {dst}")

def build_workflow(config: Dict[str, any]) -> StateGraph:
    """
    Builds and compiles the pipeline based on the provided tools.

    Args:
        pipeline_tools (str): A string of pipeline tool names separated by '+'.

    Returns:
        StateGraph: The compiled team.
    """

    builder = WorkflowBuilder(config)
    builder.build()
    workflow = builder.workflow.compile()
    logging.info("Team built and compiled successfully")
    return workflow