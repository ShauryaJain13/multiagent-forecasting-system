from agent.agent import Agent
from agent.memory import Memory
from chat.llm import LLMClient
from chat.prompts import Prompt_Builder
from tools.registry import ToolRegistry, Tool
from controller import Controller
from tools.calculator import Calculator
from tools.read_file import ReadFile
from tools.read_csv import ReadCSV


memory = Memory()

prompt_builder = Prompt_Builder(
    system_prompt="""You are a helpful AI assistant.
    Use tools when necessary to answer the user's questions.
    If you are using a tool, mention the tool you are using explicitly""")

llm_client = LLMClient()
tools = ToolRegistry()

calculator = Calculator()
calculator_tool = Tool(
    name="calculator",
    description="Evaluate a mathematical expression.",
    function=calculator.execute,
    arguments={"type": "object",
               "properties": {
                   "expression": {
                       "type": "string",
                       "description": ("The mathematical"
                                       "expression to evaluate.")}},
                "required": ["expression"]})

read_file = ReadFile()
read_file_tool = Tool(name="file_reader",
                      description="Reads a file that has been entered and"
                      "returns the contents",
                      function=read_file.execute,
                      arguments={"type": "object",
                                 "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": ("The path of the file"
                                                        "to read, relative to"
                                                        "the directory")}},
                                 "required": ["file_path"]})

read_csv = ReadCSV()
read_csv_tool = Tool(name="csv_reader",
                     description="Reads a csv file that has been entered and"
                     "returns the contents, including the column names and"
                     "titles",
                     function=read_csv.execute,
                     arguments={"type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": ("The path of the csv"
                                                        "file to read,"
                                                        "relative to the"
                                                        "directory")}},
                                "required": ["file_path"]})

tools.register(calculator_tool)
tools.register(read_file_tool)
tools.register(read_csv_tool)

agent = Agent(llm=llm_client, tools=tools, memory=memory,
              prompt_builder=prompt_builder)

controller = Controller(agent)
controller.loop()
