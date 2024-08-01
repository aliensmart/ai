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

