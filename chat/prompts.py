from agent.memory import Memory


class Prompt_Builder:
    """
    This class accepts the history of the chat, the system prompt
    (how the model is supposed to behave) and the current message
    that the user just inputted. It is supposed to create a model
    prompt that the LLM can process and generate an answer of.
    """

    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def build_messages(self, history: Memory = None) -> list:
        """
        This method builds the actual prompt from the given information
        """
        messages = []
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        if history is not None:
            messages.extend(history.get_messages())

        return messages

    def get_system_prompt(self):
        """
        This function accepts the system prompt that tells the model
        how to behave
        """
        return self.system_prompt

    def set_system_prompt(self, system_prompt):
        """
        This function sets the system prompt
        """
        self.system_prompt = system_prompt
