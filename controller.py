class Controller:
    """
    This class serves as the 'brain' or the controller of the entire operation.
    It coordinates with the other classes to work and control the flow of the
    entire chatbot
    """

    def __init__(self, agent):
        self.agent = agent

    def handle_message(self, message: str):
        """
        This function accepts an input from the user
        """
        return self.agent.run(message)

    def handle_exit(self):
        """
        Terminating application
        """
        print("Exiting agent...")

    def loop(self):
        """
        Keeps running the application over and over until the user
        decides to exit
        """
        print("Agent has started. Type 'exit' or 'quit' to quit")

        while True:
            try:
                message = input("\nYou: ")

                if message.lower() in {"exit", "quit"}:
                    self.handle_exit()
                    break

                response = self.handle_message(message)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                self.handle_exit()
                break

            except Exception as e:
                print(f"Error: {e}")
