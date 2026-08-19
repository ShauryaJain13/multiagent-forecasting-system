from agent.base_agent import BaseAgent


class ForecastingAgent(BaseAgent):
    """
    This is an agent that enables the actual forecasting of the data
    """

    def __init__(self, llm, tools, prompt_builder, memory):
        super().__init__(name="Forecasting Agent", llm=llm, tools=tools,
                         system_prompt="You are a forecasting agent",
                         prompt_builder=prompt_builder, memory=memory)
