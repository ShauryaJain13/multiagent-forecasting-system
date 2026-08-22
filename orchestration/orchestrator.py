from orchestration.state import AgentState


class Orchestrator:
    """
    This class is the functionality behind the MAS. It orchestrates which data
    is passed to which agent, when it is passed, and what to do next
    """
    def __init__(self, router, llm, prompt_builder, data_agent,
                 forecasting_agent, anomaly_agent, max_iterations=10):
        self.router = router
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

            # CHANGED: router.route() can raise (invalid/malformed JSON
            # from the LLM, missing fields, unknown agent name, LLM
            # request failure). Previously this call wasn't wrapped, so
            # any routing failure would crash the whole request with no
            # graceful fallback -- now it's logged to state.errors and the
            # loop breaks, still letting generate_final_response report
            # back whatever was accomplished so far.
            try:
                decision = self.router.route(task, state)
            except Exception as e:
                state.add_error({"component": "orchestrator",
                                 "error": f"routing failed: {e}"})
                break

            next_agent_name = decision["agent"]

            if next_agent_name not in self.agents:
                state.add_error({"component": "orchestrator",
                                 "error": f"unknown agent {next_agent_name}"})
                break

            agent_task = decision["task"]
            next_agent = self.agents[next_agent_name]
            state.set_current_agent(next_agent.name)

            try:
                next_agent.run(agent_task, state)
            except Exception as e:
                state.add_error({"agent": next_agent_name,
                                 "error": str(e)})
                break
        else:
            state.add_error({
                "component": "orchestrator",
                "error": (f"Maximum iterations ({self.max_iterations}"
                          ") reached before the task was completed.")})

        # CHANGED: this is the main fix. Every code path above (early
        # break on completion, break on error, or the for-else exhaustion
        # branch) now converges here, so run() always returns a real
        # answer built from whatever ended up in shared state, instead of
        # implicitly returning None.
        self.final_response = self.generate_final_response(task, state)
        return self.final_response

    # CHANGED: removed choose_agent(). It was dead code -- an earlier,
    # simpler "LLM returns just an agent name" approach that got fully
    # superseded by Router (router.py), which returns structured JSON
    # with an agent name, task, AND reason. run() only ever calls
    # self.router.route(), never self.choose_agent(), so this method was
    # never executed. Removed to avoid confusion about which one is live.

    def is_task_complete(self, state):
        """
        Determine whether enough work has been completed
        to produce a final answer.

        NOTE (not fixed, flagging for awareness): this requires BOTH
        state.data and state.forecast to be set, regardless of what the
        user actually asked for. A request like "are there any anomalies
        in this dataset?" doesn't need a forecast, but the loop will keep
        routing to forecasting_agent anyway until max_iterations is hit,
        since this check has no way to know the task didn't require
        forecasting. Worth revisiting -- e.g. having the Router signal
        "done" explicitly, or giving is_task_complete per-task criteria --
        once you're ready to handle non-forecasting requests well.
        """
        if state.data is None or state.forecast is None:
            return False

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