"""
MergeExploreGraphsNode Module
"""

# Imports from standard library
from typing import List, Optional
from functools import reduce
import operator
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm
from ..models import Ollama
from .base_node import BaseNode


class MergeExploreGraphsNode(BaseNode):
    """
    A node that generates an answer using a large language model (LLM) based on the user's input
    and the content extracted from a webpage. It constructs a prompt from the user's input
    and the scraped content, feeds it to the LLM, and parses the LLM's response to produce
    an answer.

    Attributes:
        llm_model: An instance of a language model client, configured for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "GenerateAnswer".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "MergeExploreGraph",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)

        self.llm_model = node_config["llm_model"]

        if isinstance(node_config["llm_model"], Ollama):
            self.llm_model.format="json"

        self.verbose = (
            True if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        """
        Generates an answer by constructing a prompt from the user's input and the scraped
        content, querying the language model, and parsing its response.

        Args:
            state (dict): The current state of the graph. The input keys will be used
                            to fetch the correct data from the state.

        Returns:
            dict: The updated state with the output key containing the generated answer.

        Raises:
            KeyError: If the input keys are not found in the state, indicating
                      that the necessary information for generating an answer is missing.
        """

        self.logger.info(f"--- Executing {self.node_name} Node ---")

         # merge the answers in one string
     

        # Initialize the output parser
        if self.node_config.get("schema", None) is not None:
            output_parser = JsonOutputParser(pydantic_object=self.node_config["schema"])
        else:
            output_parser = JsonOutputParser()

        format_instructions = output_parser.get_format_instructions()

        template_answer = """
        You are a website scraper and you have just scraped some content from multiple websites.\n
        You are now asked to provide an answer to a USER PROMPT based on the content you have scraped.\n
        You need to merge the content from the different websites into a single answer without repetitions (if there are any). \n
        The scraped contents are in a JSON format and you need to merge them based on the context and providing a correct JSON structure.\n
        OUTPUT INSTRUCTIONS: {format_instructions}\n
        USER PROMPT: {user_prompt}\n
        WEBSITE CONTENT: {website_content}
        """

        input_keys = self.get_input_keys(state)
        # Fetching data from the state based on the input keys
        input_data = [state[key] for key in input_keys]

        user_prompt = input_data[0]
        #answers is a list of strings
        answers, relevant_links = zip(*input_data[1])

        answers_str = ""
        for i, answer in enumerate(answers):
            answers_str += f"CONTENT WEBSITE {i+1}: {answer}\n"

        answers = str(state.get("answer"))
        relevant_links = str(state.get("relevant_links"))
        answer = {}

        merge_prompt = PromptTemplate(
                template=template_answer,
                input_variables=["context", "question"],
                partial_variables={"format_instructions": format_instructions},
            )

        answer = merge_prompt.invoke({"question": user_prompt})

        state.update({"relevant_links": reduce(operator.ior, relevant_links, {})})
        state.update({"answer": answer})
        return state
