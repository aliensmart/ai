import os

from dotenv import load_dotenv
from autogen import ConversableAgent
import pprint

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

# the agent is created with the config and the human input mode is set to NEVER to make the agent fully autonomous and not require any human input
# agent = ConversableAgent(
#     name="chatbot",
#     llm_config=llm_config,
#     human_input_mode="NEVER"
# )

# the agent is asked to generate a reply to the user input "Tell me a joke" and the reply is printed to the console 
# reply = agent.generate_reply(messages=[{
#     "role": "user",
#     "content": "Tell me a joke"
# }])
# print(reply)

# cathy = ConversableAgent(
#     name="cathy",
#     llm_config=llm_config,
#     human_input_mode="NEVER",
#     system_message="You are Cathy, a stand-up comedian. You are performing a stand-up comedy show. You have to make the audience laugh. You can tell jokes, funny stories, or anything that you think will make the audience laugh. You can also interact with the audience. Remember, your goal is to make the audience laugh. You can start your performance now."
#     )


# joe = ConversableAgent(
#     name="joe",
#     llm_config=llm_config,
#     human_input_mode="NEVER",
#     system_message=
#     "You are Joe, a stand-up comedian"
#     "Start the next joke with the punchline of the previous joke, and try to make the audience laugh. You can also interact with the audience. Remember, your goal is to make the audience laugh. You can start your performance now."  
#     )

# chat_result = joe.initiate_chat(
#     recipient=cathy,
#     message="I'm Joe. Cathy, do you want to start the show?",
#     max_turns=5
#     )


# pprint.pprint(chat_result.chat_history)

# pprint.pprint(chat_result.cost)

# pprint.pprint(chat_result.summary)


cathy = ConversableAgent(
    name="cathy",
    system_message=
    "Your name is Cathy and you are a stand-up comedian. "
    "When you're ready to end the conversation, say 'I gotta go'.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"],
)

joe = ConversableAgent(
    name="joe",
    system_message=
    "Your name is Joe and you are a stand-up comedian. "
    "When you're ready to end the conversation, say 'I gotta go'.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "I gotta go" in msg["content"] or "Goodbye" in msg["content"],
)

chat_result = joe.initiate_chat(
    recipient=cathy,
    message="I'm Joe. Cathy, let's keep the jokes rolling."
)

# cathy.send(message="What's, the last joke we talked about?")