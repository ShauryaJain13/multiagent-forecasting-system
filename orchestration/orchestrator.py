from orchestration.state import AgentState


class Orchestrator:
    """
    This class is the functionality behind the MAS. It orchestrates which data
    is passed to which agent, when it is passed, and what to do next
    """
    def __init__(self, llm, prompt_builder, data_agent, forecasting_agent,
                 anomaly_agent, max_iterations=10):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.agents = {"data_agent": data_agent,
                       "forecasting_agent": forecasting_agent,
                       "anomaly_agent": anomaly_agent}
        self.max_iterations = max_iterations
        self.final_response = None

    def run(self, task, state=None):
        """
        The actual running of the function
        """
        if state is None:
            state = AgentState(task)

        for _ in range(self.max_iterations):
            if self.is_task_complete(state):
                break

            next_agent_name = self.choose_agent(task, state)
            if next_agent_name not in self.agents:
                state.add_error({"component": "orchestrator",
                                 "error": f"unknown agent {next_agent_name}"})
                break

            next_agent = self.agents[next_agent_name]
            state.set_current_agent(next_agent)

            try:
                next_agent.run(task, state)
            except Exception as e:
                state.add_error({"agent": next_agent_name,
                                 "error": str(e)})
                break

            # if self.is_task_complete(state) is True:
            #     self.is_complete = True
            #     break

        state.add_error({"component": "orchestrator",
                         "error": (f"Maximum iterations ({self.max_iterations}"
                                   ") reached before the task was completed.")}
                        )

        # while not self.is_complete(state):
        #     next_agent = self.choose_agent(task, state)
        #     result = next_agent.run(task, state)

        # self.final_response = result
        # return self.final_response(state)

    def choose_agent(self, state):
        """
        LLM-driven, choosing the appropriate agent
        """
        context = state.to_dict()

        messages = self.prompt_builder.build_messages(
            memory=[],
            system_prompt=(
                "You are the orchestrator of a multi-agent "
                "data forecasting system. Your job is to "
                "decide which specialist agent should act "
                "next based on the current task and shared "
                "state.\n\n"

                "Available agents:\n"
                "- data_agent: understands and prepares data\n"
                "- forecasting_agent: evaluates models and "
                "produces forecasts\n"
                "- anomaly_agent: detects and evaluates "
                "anomalies\n\n"

                "Return ONLY the name of the next agent."
            ),
            context=context)

        response = self.llm.generate(messages, tools=None)

        if response is None:
            raise ValueError("Orchestrator LLM returned no response.")

        agent_name = response.content.strip()
        return agent_name

    def is_task_complete(self, state):
        """
        Determine whether enough work has been completed
        to produce a final answer.
        """
        if state.data is None or state.forecast is None:
            return False
        # if state.forecast is None:
        #     return False

        return True

    def generate_final_response(self, task, state):
        """
        Generate the final response to the user using
        the information accumulated in shared state.
        """

        context = state.to_dict()

        messages = self.prompt_builder.build_messages(
            memory=[],
            system_prompt=(
                "You are the final response generator for "
                "a data forecasting system. Use the results "
                "contained in the shared state to answer "
                "the user's request clearly. Explain the "
                "selected model, forecast, and any relevant "
                "warnings or anomalies."),
            context=context)

        response = self.llm.generate(messages, tools=None)
        if response is None:
            return "Unable to generate a final response."

        return response.content
