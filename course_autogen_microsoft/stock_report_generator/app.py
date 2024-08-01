import os

from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent, GroupChat, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
import yfinance
import matplotlib.pyplot as plt
import datetime

load_dotenv()

# Debug statements to print environment variables
# print("Model:", os.getenv("model"))
print("OpenAI API Key:", os.getenv("openai_api"))
# print("Seed:", os.getenv("seed"))
# print("Context Window:", os.getenv("context_window"))


config_list = [
    {
        "model": os.getenv("model"),
        "api_key": os.getenv("openai_api"),
        
    },
]

llm_config = {"config_list": config_list, "seed": int(os.getenv("seed"))}

task = "Write a blogpost about the stock price performance of "\
"Nvidia in the past month. Today's date is 2024-04-23."


# Build a group chat
# This group chat will include these agents:

# User_proxy or Admin: to allow the user to comment on the report and ask the writer to refine it.
# Planner: to determine relevant information needed to complete the task.
# Engineer: to write code using the defined plan by the planner.
# Executor: to execute the code written by the engineer.
# Writer: to write the report.

user_proxy =  ConversableAgent(
    name="Admin",
    system_message="Give the task, and send "
    "instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="ALWAYS",
)

planner = ConversableAgent(
    name="Planner",
    system_message="Given a task, please determine "
    "what information is needed to complete the task. "
    "Please note that the information will all be retrieved using"
    " Python code. Please only suggest information that can be "
    "retrieved using Python code. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps. If a step fails, try to "
    "workaround",
    description="Planner. Given a task, determine what "
    "information is needed to complete the task. "
    "After each step is done by others, check the progress and "
    "instruct the remaining steps",
    llm_config=llm_config,
)


engineer = AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    description="An engineer that writes code based on the plan "
    "provided by the planner.",
)


executor = ConversableAgent(
    name="Executor",
    system_message="Execute the code written by the "
    "engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },
)

writer = ConversableAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="Writer."
    "Please write blogs in markdown format (with relevant titles)"
    " and put the content in pseudo ```md``` code block. "
    "You take feedback from the admin and refine your blog.",
    description="Writer."
    "Write blogs based on the code execution results and take "
    "feedback from the admin to refine the blog."
)

# groupchat =GroupChat(
#     agents=[user_proxy, engineer, writer, executor, planner],
#     messages=[],
#     max_round=10,
# )

groupchat = GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions={
        user_proxy: [engineer, writer, executor, planner],
        engineer: [user_proxy, executor],
        writer: [user_proxy, planner],
        executor: [user_proxy, engineer, planner],
        planner: [user_proxy, engineer, writer],
    },
    speaker_transitions_type="allowed",
)

manager = GroupChatManager(
    groupchat=groupchat, llm_config=llm_config
)

groupchat_result = user_proxy.initiate_chat(
    manager,
    message=task,
)