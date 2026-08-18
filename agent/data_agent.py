class DataAgent(BaseAgent):
    """
    This agent class is specifically meant for the Data of the files
    """

    def __init__(self, llm, tools):
        super.__init__()
        super.name = "Data Agent"
        super.