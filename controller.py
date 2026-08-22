from orchestration.orchestrator import Orchestrator
from orchestration.router import Router

from agent.data_agent import DataAgent
from agent.anomaly_agent import AnomalyAgent
from agent.forecasting_agent import ForecastingAgent

from tools.registry import ToolRegistry

from chat.llm import LLMClient
from chat.prompts import Prompt_Builder
from agent.memory import Memory

from orchestration.state import AgentState


class Controller:
    """
    Controls and initializes the Multi-Agent System.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.prompt_builder = Prompt_Builder()
        self.memory = Memory()

        self.data_tools = ToolRegistry()
        self.anomaly_tools = ToolRegistry()
        self.forecasting_tools = ToolRegistry()

        self.data_agent = DataAgent(llm=self.llm, tools=self.data_tools,
                                    prompt_builder=self.prompt_builder,
                                    memory=self.memory)

        self.anomaly_agent = AnomalyAgent(llm=self.llm,
                                          tools=self.anomaly_tools,
                                          prompt_builder=self.prompt_builder,
                                          memory=self.memory)

        self.forecasting_agent = ForecastingAgent(llm=self.llm,
                                                  tools=self.forecasting_tools,
                                                  prompt_builder=self.
                                                  prompt_builder,
                                                  memory=self.memory)

        self.router = Router(llm=self.llm, prompt_builder=self.prompt_builder)

        self.orchestrator = Orchestrator(router=self.router,
                                         data_agent=self.data_agent,
                                         forecasting_agent=self.
                                         forecasting_agent,
                                         anomaly_agent=self.anomaly_agent,
                                         max_iterations=10)

    def run(self, task):
        """
        Start the Multi-Agent System.

        Creates a shared AgentState and gives it to
        the Orchestrator.
        """
        state = AgentState(task)
        response = self.orchestrator.run(
            task=task,
            state=state
        )
        return response
