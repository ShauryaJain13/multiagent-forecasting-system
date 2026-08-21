class AgentState:
    """
    This class monitors the 'state' of the model
    """

    def __init__(self, user_request):
        self.user_request = user_request
        self.data = None
        self.data_summary = None
        self.forecast = None
        self.forecast_metrics = None
        self.anomalies = []
        self.current_agent = None
        self.completed_agents = []
        self.errors = []

    def add_error(self, error):
        """
        Adds the error to the current state
        """
        return self.errors.append(error)

    def mark_agent_complete(self, agent):
        """
        Marking agent after it finished its task
        """
        if agent not in self.completed_agents:
            self.completed_agents.append(agent)

        self.current_agent = None

    def set_current_agent(self, agent):
        """
        Setting current agent
        """
        self.current_agent = agent

    def to_dict(self):
        """
        Converts the state to a dictionary for easy readability.
        This also allows the state to be included in the context in an easier
        manner
        """
        return {"user_request": self.user_request,
                "data": self.data,
                "data_summary": self.data_summary,
                "forecast": self.forecast,
                "forecast_metrics": self.forecast_metrics,
                "anomalies": self.anomalies,
                "current_agent": self.current_agent,
                "completed_agents": self.completed_agents,
                "errors": self.errors}
