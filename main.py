import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ibm import ChatWatsonx
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

# Load configuration from .env
load_dotenv(override=True)


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source.")


class Job(BaseModel):
    """Schema for a job posting found by the agent."""

    title: str = Field(description="The job title.")
    company: str = Field(description="The hiring company.")
    location: str = Field(description="The job location.")
    details: str = Field(description="A short description of the role and requirements.")
    url: str = Field(description="The URL of the job posting.")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(description="The agent's answer to the query.")
    jobs: List[Job] = Field(
        default_factory=list,
        description="Job postings found, if the query is about jobs.",
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer.",
    )


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


# Configure the Watsonx language model
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=os.environ["WATSONX_URL"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    apikey=os.environ["WATSONX_API_KEY"],
    params={
        "temperature": 0,
        "max_tokens": 1000,
    },
)

tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "Use the search tool to find current information. "
        "Put the complete answer in the answer field: for each item found, "
        "include all relevant details such as title, company, location and "
        "a short description. Do not only write an introduction."
    ),
    response_format=AgentResponse,
)


def main():
    print("Running the Watsonx agent...")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
                )
            ]
        }
    )

    response = result["structured_response"]
    print(response.answer)
    for job in response.jobs:
        print(f"\n{job.title} at {job.company} ({job.location})")
        print(f"  {job.details}")
        print(f"  {job.url}")
    print("\nSources:")
    for source in response.sources:
        print(f"- {source.url}")


if __name__ == "__main__":
    main()
