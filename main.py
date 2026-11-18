import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ibm import ChatWatsonx
from tavily import TavilyClient

# Load configuration from .env
load_dotenv(override=True)


tavily = TavilyClient()

# Check that the required Watsonx variables exist
required_variables = [
    "WATSONX_URL",
    "WATSONX_PROJECT_ID",
    "WATSONX_API_KEY",
]

missing_variables = [
    variable
    for variable in required_variables
    if not os.getenv(variable)
]

if missing_variables:
    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing_variables)
    )


@tool
def search(query: str) -> str:
    """Search for current information on the internet.

    Args:
        query: The search query.

    Returns:
        The search result.
    """

    print(f"Searching for: {query}")

    # Temporary hard-coded result for testing
    return tavily.search(query=query)


# Configure the Watsonx language model
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=os.environ["WATSONX_URL"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    apikey=os.environ["WATSONX_API_KEY"],
    params={
        "temperature": 0,
        "max_tokens": 500,
    },
)


# Give the search tool to the agent
agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the available tools when current information is required. "
        "Call each tool no more than once for a user request."
    ),
)

def main():
    print("Running the Watsonx agent...")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the current weather in Tokyo?"
                )
            ]
        }
    )

    final_message = result["messages"][-1]
    print(final_message.content)


if __name__ == "__main__":
    main()