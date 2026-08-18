from dataclasses import dataclass, field


@dataclass
class State:
    """
    This class is to work on the "State" of the LLM.
    Much like the memory, this keeps track of things like
    number of requests, current iteration, etc. It is a core aspect
    of the ReAct loop.
    """

    messages: list = field(default_factory=list)
    iterations: int = 0
    tool_calls: list = field(default_factory=list)
    status: str = "running"
    error: str | None = None

    def record_tool_call(self, call):
        """
        Records what tool has been called and will add it to the list
        """
        self.tool_calls.append(call)

    def increment_iteration(self):
        """
        Iterating the loop of ReAct, depending on the current status of
        the model
        """
        self.iterations += 1

    def set_error(self, err):
        """
        If there is an error, then we are setting the error
        """
        self.error = err

    def complete(self):
        """
        Marking the current state as complete
        """
        self.status = "complete"

    # def __init__(self, history, tool_calls, status="running", iteration=0):
    #     self.history = history
    #     self.tool_calls = tool_calls
    #     self.status = status
    #     self.iteration = iteration
