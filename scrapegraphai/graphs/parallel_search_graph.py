"""
ParallelSearchGraph Module
"""
from copy import copy, deepcopy
from typing import Optional
from pydantic import BaseModel

from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph

from ..nodes import (
    GraphIteratorNode,
    ReRankNode,
    MergeExploreGraphsNode
)

from ..graphs.explore_graph import ExploreGraph


class ParallelSearchGraph(AbstractGraph):
    """
    SmartScraper is a scraping pipeline that automates the process of 
    extracting information from web pages
    using a natural language model to interpret and answer prompts.

    Attributes:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (str): The schema for the graph output.
        llm_model: An instance of a language model client, configured for generating answers.
        embedder_model: An instance of an embedding model client, 
        configured for generating embeddings.
        verbose (bool): A flag indicating whether to show print statements during execution.
        headless (bool): A flag indicating whether to run the graph in headless mode.

    Args:
        prompt (str): The prompt for the graph.
        source (str): The source of the graph.
        config (dict): Configuration parameters for the graph.
        schema (str): The schema for the graph output.

    Example:
        >>> smart_scraper = ParallelSearchGraph(
        ...     "List me all the attractions in Chioggia.",
        ...     "https://en.wikipedia.org/wiki/Chioggia",
        ...     {"llm": {"model": "gpt-3.5-turbo"}}
        ... )
        >>> result = smart_scraper.run()
        )
    """

    def __init__(self, prompt: str, source: str, config: dict, schema: Optional[BaseModel] = None):
        super().__init__(prompt, config, source, schema)

        self.input_key = "url" if source.startswith("http") else "local_dir"

        if all(isinstance(value, str) for value in config.values()):
            self.copy_config = copy(config)
        else:
            self.copy_config = deepcopy(config)
        self.copy_schema = deepcopy(schema)

        super().__init__(prompt, config, schema)

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for web scraping.

        Returns:
            BaseGraph: A graph instance representing the web scraping workflow.
        """
         
        explore_graph_instance  = ExploreGraph(
            prompt="",
            source="",
            config=self.copy_config,
        )
        
        rerank_link_node  = ReRankNode(
              input="user_prompt & urls",
            output=["results"],
            node_config={
                "graph_instance": explore_graph_instance ,
            }
        )

        graph_iterator_node = GraphIteratorNode(
            input="user_prompt & urls",
            output=["results"],
            node_config={
                "graph_instance": explore_graph_instance ,
            }
        )

        merge_explore_graphs_node = MergeExploreGraphsNode(input="user_prompt & results",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "schema": self.schema
            }
        )


        return BaseGraph(
            nodes=[
                rerank_link_node,               
                graph_iterator_node,
                merge_explore_graphs_node,
            ],
            edges=[
                (rerank_link_node, graph_iterator_node),
                (graph_iterator_node, merge_explore_graphs_node),
            ],
            entry_point=rerank_link_node
        )

    def run(self) -> str:
        """
        Executes the scraping process and returns the answer to the prompt.

        Returns:
            str: The answer to the prompt.
        """

        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        return self.final_state.get("answer", "No answer found.")
