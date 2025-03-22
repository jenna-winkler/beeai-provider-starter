import asyncio

from acp.server.highlevel import Server, Context
from beeai_sdk.providers.agent import run_agent_provider
from beeai_sdk.schemas.metadata import UiDefinition, UiType
from beeai_sdk.schemas.text import TextInput, TextOutput

from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput


async def run():
    server = Server("beeai-agents")

    @server.agent(
        name="travel-advisor",
        description="An agent that provides travel recommendations",
        input=TextInput,
        output=TextOutput,
        ui=UiDefinition(
        type=UiType.hands_off,
        userGreeting="Enter your destination and travel dates (e.g., 'Boston, MA, Mar 23-25, 2025')"
    ),
    )
    async def travel_advisor(input: TextInput, ctx: Context) -> TextOutput:
        parts = input.text.split(',', 1)
        destination = parts[0].strip() if len(parts) > 0 else "Boston, MA"
        travel_dates = parts[1].strip() if len(parts) > 1 else "Mar 23-25, 2025"
        
        llm = ChatModel.from_name("ollama:llama3.1")
        workflow = AgentWorkflow(name="Travel Advisor")
        
        workflow.add_agent(
            name="Weather Forecaster",
            role="A diligent weather forecaster",
            instructions="You specialize in reporting on the weather.",
            tools=[OpenMeteoTool()],
            llm=llm,
        )
        
        workflow.add_agent(
            name="Activity Planner",
            role="An expert in local attractions",
            instructions="You know about interesting activities and would like to share.",
            tools=[DuckDuckGoSearchTool()],
            llm=llm,
        )
        
        workflow.add_agent(
            name="Travel Advisor",
            role="A travel advisor",
            instructions="""You can synthesize travel details such as weather and recommended activities and provide a coherent summary.""",
            llm=llm,
        )
        
        response = await workflow.run(
            inputs=[
                AgentWorkflowInput(
                    prompt=f"Provide a comprehensive weather summary for {destination} from {travel_dates}.",
                    expected_output="Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
                ),
                AgentWorkflowInput(
                    prompt=f"Search for a set of activities close to {destination} from {travel_dates} that are appropriate in light of the weather conditions.",
                    expected_output="A list of activities including location and description that are weather appropriate.",
                ),
                AgentWorkflowInput(
                    prompt=f"Consider the weather report and recommended activities for the trip to {destination} from {travel_dates} and provide a coherent summary.",
                    expected_output="A summary of the trip that the traveler could take with them. Break it down by day including weather, location and helpful tips.",
                ),
            ]
        )
        
        return TextOutput(text=f"Travel Recommendations for {destination}\n\n{response.state.final_answer}")

    await run_agent_provider(server)


def main():
    asyncio.run(run())
