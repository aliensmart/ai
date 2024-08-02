"""
we demonstrate how to utilize RetrieveChat to generate code and answer questions 
based on customized documentations that are not present in the LLM’s training dataset.
RetrieveChat uses the RetrieveAssistantAgent and RetrieveUserProxyAgent, 
which is similar to the usage of AssistantAgent and UserProxyAgent in other notebooks
"""


import json
import os

import chromadb

from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

load_dotenv()


config_list = [
    {"model": os.getenv("model"),
        "api_key": os.getenv("openai_api"),
        "api_type": "openai"
    },
]

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])


print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)

# 1. create an RetrieveAssistantAgent instance named "assistant"
assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)


# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
# `docs_path` is the path to the docs directory. It can also be the path to a single file, or the url to a single file. By default,
# it is set to None, which works only if the collection is already created.
# `task` indicates the kind of task we're working on. In this example, it's a `code` task.
# `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.
# `custom_text_types` is a list of file types to be processed. Default is `autogen.retrieve_utils.TEXT_FORMATS`.
# This only applies to files under the directories in `docs_path`. Explicitly included files and urls will be chunked regardless of their types.
# In this example, we set it to ["non-existent-type"] to only process markdown files. Since no "non-existent-type" files are included in the `websit/docs`,
# no files there will be processed. However, the explicitly included urls will still be processed.
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
            os.path.join(os.path.abspath(""), "..", "website", "docs"),
        ],
        "custom_text_types": ["non-existent-type"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        # "client": chromadb.PersistentClient(path="/tmp/chromadb"),  # deprecated, use "vector_db" instead
        "vector_db": "chroma",  # to use the deprecated `client` parameter, set to None and uncomment the line above
        "overwrite": False,  # set to True if you want to overwrite an existing collection
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)


"""
Example 1: Generate code based off docstrings w/o human feedback
Use RetrieveChat to help generate sample code and automatically run the code and fix errors if there is any.

Problem: Which API should I use if I want to use FLAML for a classification task and I want to train the model in 30 seconds. Use spark to parallel the training. Force cancel jobs if time limit is reached.
"""
# reset the assistant. Always reset the assistant before starting a new conversation.
# assistant.reset()

# given a problem, we use the ragproxyagent to generate a prompt to be sent to the assistant as the initial message.
# the assistant receives the message and generates a response. The response will be sent back to the ragproxyagent for processing.
# The conversation continues until the termination condition is met, in RetrieveChat, the termination condition when no human-in-loop is no code block detected.
# With human-in-loop, the conversation will continue until the user says "exit".
# code_problem = "How can I use FLAML to perform a classification task and use spark to do parallel training. Train 30 seconds and force cancel jobs if time limit is reached."
# chat_result = ragproxyagent.initiate_chat(
#     assistant, message=ragproxyagent.message_generator, problem=code_problem, search_string="spark"
# )  # search_string is used as an extra filter for the embeddings search, in this case, we only want to search documents that contain "spark".


"""
Use RetrieveChat to answer a question that is not related to code generation.

Problem: Who is the author of FLAML?

Example 2: Answer a question based off docstrings w/o human feedback
"""

# reset the assistant. Always reset the assistant before starting a new conversation.
# assistant.reset()

# qa_problem = "Who is the author of FLAML?"
# chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)



"""
Example 3: Generate code based off docstrings w/ human feedback
Use RetrieveChat to help generate sample code and ask for human-in-loop feedbacks.
"""
# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

# set `human_input_mode` to be `ALWAYS`, so the agent will ask for human input at every step.
ragproxyagent.human_input_mode = "ALWAYS"
code_problem = "how to build a time series forecasting model for stock price using FLAML?"
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=code_problem)



"""
Example 4: Answer a question based off docstrings w/ human feedback
Use RetrieveChat to answer a question and ask for human-in-loop feedbacks.
"""
# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

# set `human_input_mode` to be `ALWAYS`, so the agent will ask for human input at every step.
ragproxyagent.human_input_mode = "ALWAYS"
qa_problem = "Is there a function named `tune_automl` in FLAML?"
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=qa_problem
)  # type "exit" to exit the conversation
