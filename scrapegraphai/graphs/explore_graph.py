"""
ExploreGraph Module
"""

from typing import Optional
from pydantic import BaseModel

from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph

from ..nodes import (
    FetchNode,
    ParseNode,
    RAGNode,
    GenerateAnswerNode,
    SearchLinkNode
)


class ExploreGraph(AbstractGraph):
    """
    ExploreGraph is a web scraping pipeline that automates the extraction of information
    from web pages using natural language models to interpret and respond to prompts.

    Attributes:
        prompt (str): The prompt for the graph.
        source (str): The source URL or local directory for the graph.
        config (dict): Configuration parameters for the graph.
        schema (str): The schema for the graph output.
        llm_model: An instance of a language model client for generating answers.
        embedder_model: An instance of an embedding model client for generating embeddings.
        verbose (bool): A flag indicating whether to show print statements during execution.
        headless (bool): A flag indicating whether to run the graph in headless mode.

    Args:
        prompt (str): The prompt for the graph.
        source (str): The source URL or local directory for the graph.
        config (dict): Configuration parameters for the graph.
        schema (Optional[BaseModel]): The schema for the graph output.

    Example:
        >>> explore_graph = ExploreGraph(
        ...     "List me all the attractions in Chioggia.",
        ...     "https://en.wikipedia.org/wiki/Chioggia",
        ...     {"llm": {"model": "gpt-3.5-turbo"}}
        ... )
        >>> result = explore_graph.run()
        >>> print(result)
    """

    def __init__(self, prompt: str, source: str, config: dict, schema: Optional[BaseModel] = None):
        super().__init__(prompt, config, source, schema)
        self.input_key = "url" if source.startswith("http") else "local_dir"

    def _create_graph(self) -> BaseGraph:
        """
        Creates the graph of nodes representing the workflow for web scraping.

        Returns:
            BaseGraph: A graph instance representing the web scraping workflow.
        """
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
            output=[{"relevant_links"}],
            node_config={
                "llm_model": self.llm_model,
            }
        )

        return BaseGraph(
            nodes=[
                fetch_node,
                parse_node,
                rag_node,
                search_link_node,
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

    def run(self) -> tuple[str, dict]:
        """
        Executes the scraping process and returns the answer to the prompt.

        Returns:
            str: The answer to the prompt.
        """
        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        return (self.final_state.get("answer", "No answer found."),
                self.final_state.get("relevant_links", dict()))
