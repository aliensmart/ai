import os

from dotenv import load_dotenv
from autogen import ConversableAgent, initiate_chats
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


# this agent is the first agent in the conversation, it is responsible for collecting personal information from the customer to create an account
onboarding_personal_information_agent = ConversableAgent(
    name="Onboarding Personal Information Agent",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message='''
    You are a helpful customer service agent. You are helping a new customer onboard to your platform. 
    You need to collect some personal information from the customer to create an account for them. 
    You can ask the customer for their name, email address, phone number, and any other information you need to create an account for them. 
    Remember, your goal is to make the onboarding process as smooth as possible for the customer. 
    Remember to be polite and professional at all times. Our company name is Alienmoore, and we are software development company.
    Return 'TERMINATE' when you have collected all the necessary information.
    ''',
    code_execution_config=False,
    )

# this agent is the second agent in the conversation, it is responsible for collecting information about the customer's interests and preferences to customize their experience
onboarding_topic_preference_agent = ConversableAgent(
    name="Onboarding Topic preference Agent",
    system_message='''You are a helpful customer onboarding agent,
    you are here to help new customers get started with Alienmoore.
    You need to collect information about the customer's interests and preferences to customize their experience. 
    You can ask the customer about their interests, hobbies, and preferences to personalize their experience.''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# this agent is the third agent in the conversation, it is responsible for engaging with the customer to provide customer service and support
customer_engagement_agent = ConversableAgent(
    name="Customer Engagement Agent",
    system_message='''You are a customer engagement agent at Alienmoore. 
    You are responsible for engaging with customers who have signed up for our platform. 
    Your goal is to provide excellent customer service and support to ensure customer satisfaction. 
    You can answer questions, provide assistance, and engage with customers in a friendly and professional manner. 
    Remember, the customer's satisfaction is our top priority. 
    Return 'TERMINATE' when the conversation is complete.''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)


# this agent is a proxy agent that acts as the customer in the conversation, it receives messages from the onboarding agents and forwards them to the customer engagement agent
customer_proxy_agent = ConversableAgent(
    name="customer_proxy_agent",
    llm_config=False,
    code_execution_config=False,
    human_input_mode="ALWAYS",
    is_termination_msg=lambda msg: "terminate" in msg.get("content").lower(),
)


chats = [
    {
        "sender": onboarding_personal_information_agent, # this is the first agent in the conversation that collects personal information from the customer
        "recipient": customer_proxy_agent, # this is the proxy agent that acts as the customer in the conversation
        "message": 
            "Hello, I'm here to help you get started you started with Alienmoore. "
            "I need to collect some information from you to create an account. "
            "Can you please provide me with your name, email address, and phone number?", # this is the message sent by the agent to the customer
            
        "summary_method": "reflection_with_llm", # this is the method used to generate a summary of the conversation
        "summary_args": {
            "summary_prompt" : "Return the customer information "
                             "into as JSON object only: "
                             "{'name': '', 'location': ''}", # this is the prompt used to generate the summary
        },
        "max_turns": 2, # this is the maximum number of turns allowed in the conversation
        "clear_history" : True # this flag indicates whether to clear the conversation history after the conversation is complete
    },
    {
        "sender": onboarding_topic_preference_agent, # this is the second agent in the conversation that collects information about the customer's interests and preferences
        "recipient": customer_proxy_agent, # this is the proxy agent that acts as the customer in the conversation
        "message": 
                " Welcome to Alienmoore! I'm here to help you get started. "
                "Can you tell me about your interests and preferences? " # this is the message sent by the agent to the customer
                ,
        "summary_method": "reflection_with_llm", # this is the method used to generate a summary of the conversation
        "max_turns": 1, # this is the maximum number of turns allowed in the conversation
        "clear_history" : False # this flag indicates whether to clear the conversation history after the conversation is complete
    },
    {
        "sender": customer_proxy_agent, # this is the proxy agent that acts as the customer in the conversation
        "recipient": customer_engagement_agent, # this is the third agent in the conversation that engages with the customer to provide customer service and support
        "message": "I'm ready to engage with you.", # this is the message sent by the agent to the customer
        "max_turns": 1, # this is the maximum number of turns allowed in the conversation
        "summary_method": "reflection_with_llm",
    },
]

chat_results = initiate_chats(chats) # this function initiates the conversation between the agents and returns the results

for chat_result in chat_results:
    print(chat_result.summary)
    print("\n")
    
for chat_result in chat_results:
    print(chat_result.cost)
    print("\n")