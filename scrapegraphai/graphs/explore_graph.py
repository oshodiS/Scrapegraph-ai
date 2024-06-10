""" 
ExploreGraph Module
"""

from copy import copy, deepcopy
from typing import Optional
from pydantic import BaseModel

from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph
from .smart_scraper_graph import SmartScraperGraph

from ..nodes import (
    FetchNode,
    ParseNode,
    RAGNode,
    GenerateAnswerNode,
    SearchLinkNode
)


class ExploreGraph(AbstractGraph):
    """ 
    ExploreGraph is a scraping pipeline that searches the internet for answers to a given prompt.
    It only requires a user prompt to search the internet and generate an answer.

    Attributes:
        prompt (str): The user prompt to search the internet.
        llm_model (dict): The configuration for the language model.
        embedder_model (dict): The configuration for the embedder model.
        headless (bool): A flag to run the browser in headless mode.
        verbose (bool): A flag to display the execution information.
        model_token (int): The token limit for the language model.

    Args:
        prompt (str): The user prompt to search the internet.
        config (dict): Configuration parameters for the graph.
        schema (Optional[str]): The schema for the graph output.

    Example:
        >>> search_graph = ExploreGraph(
        ...     "What is Chioggia famous for?",
        ...     {"llm": {"model": "gpt-3.5-turbo"}}
        ... )
        >>> result = search_graph.run()
    """

    def __init__(self, prompt: str, config: dict, schema: Optional[BaseModel] = None):

        self.max_results = config.get("max_results", 3)

        if all(isinstance(value, str) for value in config.values()):
            self.copy_config = copy(config)
        else:
            self.copy_config = deepcopy(config)
        
        self.copy_schema = deepcopy(schema)

        super().__init__(prompt, config, schema)

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for web scraping and searching.

        Returns:
            BaseGraph: A graph instance representing the web scraping and searching workflow.
        """

        # ************************************************
        # Create a SmartScraperGraph instance
        # ************************************************

        fetch_node = FetchNode(
            input="url | local_dir",
            output=["doc", "link_urls", "img_urls"],
            node_config={
                "loader_kwargs": self.config.get("loader_kwargs", {}),
            }
        )
        parse_node = ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={
                "chunk_size": self.model_token
            }
        )
        rag_node = RAGNode(
            input="user_prompt & (parsed_doc | doc)",
            output=["relevant_chunks"],
            node_config={
                "llm_model": self.llm_model,
                "embedder_model": self.embedder_model
            }
        )
        generate_answer_node = GenerateAnswerNode(
            input="user_prompt & (relevant_chunks | parsed_doc | doc)",
            output=["answer"],
            node_config={
                "llm_model": self.llm_model,
                "schema": self.schema,
            }
        )

        search_link_node = SearchLinkNode(
            input="doc",
            output=[{"link": "description"}],
            node_config={
                "llm_model": self.llm_model,
            }
        )

        return BaseGraph(
            nodes=[
                fetch_node,
                parse_node,
                rag_node,
                generate_answer_node,
            ],
            edges=[
                (fetch_node, parse_node),
                (parse_node, rag_node),
                (rag_node, generate_answer_node),
                (generate_answer_node, search_link_node)
            ],
            entry_point=fetch_node
        )

    def run(self) -> str:
        """
        Executes the web scraping and searching process.

        Returns:
            str: The answer to the prompt.
        """
        inputs = {"user_prompt": self.prompt}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        return self.final_state.get("answer", "No answer found.")
