from orchestration.state import AgentState

class Orchestrator:
    """
    This class is the functionality behind the MAS. It orchestrates which data
    is passed to which agent, when it is passed, and what to do next
    """
    def __init__(self):
        self.is_complete = False
        self.final_response = None

    def run(self, task, state):
        """
        The actual running of the function
        """
        state = AgentState(task)

        while not self.is_complete(state):
            next_agent = self.choose_agent(task, state)
            result = next_agent.run(task, state)

        self.final_response = result
        return self.final_response(state)

    def choose_agent():
        """
        LLM-driven, choosing the appropriate agent
        """
        