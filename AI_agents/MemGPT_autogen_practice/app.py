import autogen
import os
import openai
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config
from dotenv import load_dotenv

load_dotenv()

# Debug statements to print environment variables
print("Model:", os.getenv("model"))
print("OpenAI API Key:", os.getenv("openai_api"))
print("Seed:", os.getenv("seed"))
print("Context Window:", os.getenv("context_window"))

# Ensure that environment variables are correctly loaded
assert os.getenv("model") is not None, "Model environment variable is not set"
assert os.getenv("openai_api") is not None, "OpenAI API key environment variable is not set"
assert os.getenv("seed") is not None, "Seed environment variable is not set"
assert os.getenv("context_window") is not None, "Context window environment variable is not set"
openai.api_key = os.getenv("openai_api")
# This config is for AutoGen agents that are not powered by MemGPT
config_list = [
    {
        "model": os.getenv("model"),
        "api_key": os.getenv("openai_api"),
        
    },
]

print("config_list:", config_list)  # Debug print

llm_config = {"config_list": config_list, "seed": int(os.getenv("seed"))}

print("llm_config:", llm_config)  # Debug print

config_list_memgpt = [
    {
        "model": "gpt-4o",
        "preset": "memgpt_chat",
        "model_wrapper": None,
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1",
        "context_window": int(os.getenv("context_window")),
        "openai_key": os.getenv("openai_api"),
    },
]

print("config_list_memgpt:", config_list_memgpt)  # Debug print

llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

print("llm_config_memgpt:", llm_config_memgpt)  # Debug print



# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "work_dir": "agents-workspace",
        "use_docker": False,
    },
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
)

# The agent playing the role of the product manager (PM)
# Let's make this a non-MemGPT agent
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)



# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent
USE_MEMGPT = False

if not USE_MEMGPT:
    # In the AutoGen example, we create an AssistantAgent to play the role of the coder
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
    )
    
    tester = autogen.AssistantAgent(
        name="Tester",
        system_message="You are a tester that should make sure that the codes are working fine.",
        llm_config=llm_config,
    )
    

else:
    # In our example, we swap this AutoGen agent with a MemGPT agent
    # This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.

    # We can use interface_kwargs to control what MemGPT outputs are seen by the groupchat
    interface_kwargs = {
        "debug": True,
        "show_inner_thoughts": True,
        "show_function_outputs": False,
    }

    # Debug print to inspect config being passed to MemGPT agent
    print("Creating MemGPT agent with the following config:")
    print("llm_config_memgpt:", llm_config_memgpt)
    print("system_message:", f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
                             f"(which I make sure to tell everyone I work with).\n"
                             f"You are participating in a group chat with a user ({user_proxy.name}) "
                             f"and a product manager ({pm.name}).")
    print("interface_kwargs:", interface_kwargs)

    coder = create_memgpt_autogen_agent_from_config(
        name="MemGPT_coder",
        system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
                       f"(which I make sure to tell everyone I work with).\n"
                       f"You are participating in a group chat with a user ({user_proxy.name}) "
                       f"and a product manager ({pm.name}).",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        interface_kwargs=interface_kwargs,
        llm_config=llm_config_memgpt,
    )

# Initialize the group chat between the user and two LLM agents (PM and coder)
groupchat = autogen.GroupChat(agents=[user_proxy, pm, coder, tester], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message="I want to make AI-Powered Market Intelligence, Platforms providing insights into market trends and competitor analysis using AI.",
)
