import os  # Import the os module to interact with the operating system

from dotenv import load_dotenv  # Import the load_dotenv function from the dotenv module
from autogen import AssistantAgent  # Import the AssistantAgent class from the autogen module
import pprint  # Import the pprint module for pretty-printing data structures

# Load environment variables from a .env file into the environment
load_dotenv()

# Print the OpenAI API key from the environment variables (for debugging purposes)
print("OpenAI API Key:", os.getenv("openai_api"))

# Create a list to store configuration settings for the language model
config_list = [
    {
        "model": os.getenv("model"),  # Get the model name from environment variables
        "api_key": os.getenv("openai_api"),  # Get the OpenAI API key from environment variables
    },
]

# Create a dictionary to store the full language model configuration
llm_config = {
    "config_list": config_list,  # Add the configuration list
    "seed": int(os.getenv("seed"))  # Get the seed value from environment variables and convert it to an integer
}

# Define the task for the writer as a string containing the instructions for the blog post
task = '''
    Write a concise but engaging blogpost about self-love.
    Make sure the blog post is within 200 words.
'''

# Create an instance of AssistantAgent for the writer
writer = AssistantAgent(
    name="Writer",  # Name the agent as "Writer"
    system_message=(
        "You are a writer. You write engaging and concise blog posts (with title) on given topics. "
        "You must polish your writing based on the feedback you receive and give a refined version. "
        "Only return your final work without additional comments."
    ),  # Set the system message describing the writer's role
    llm_config=llm_config,  # Pass the language model configuration
)

# Create an instance of AssistantAgent for the critic
critic = AssistantAgent(
    name="Critic",  # Name the agent as "Critic"
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,  # Define when the chat should terminate
    llm_config=llm_config,  # Pass the language model configuration
    system_message=(
        "You are a critic. You review the work of the writer and provide constructive feedback to help improve the quality of the content."
    ),  # Set the system message describing the critic's role
)

# Create an instance of AssistantAgent for the SEO reviewer
SEO_reviewer = AssistantAgent(
    name="SEO Reviewer",  # Name the agent as "SEO Reviewer"
    llm_config=llm_config,  # Pass the language model configuration
    system_message=(
        "You are an SEO reviewer, known for your ability to optimize content for search engines, ensuring that it ranks well and attracts organic traffic. "
        "Make sure your suggestion is concise (within 3 bullet points), concrete, and to the point. Begin the review by stating your role."
    ),  # Set the system message describing the SEO reviewer's role
)

# Create an instance of AssistantAgent for the legal reviewer
legal_reviewer = AssistantAgent(
    name="Legal Reviewer",  # Name the agent as "Legal Reviewer"
    llm_config=llm_config,  # Pass the language model configuration
    system_message=(
        "You are a legal reviewer, known for your ability to ensure that content is legally compliant and free from any potential legal issues. "
        "Make sure your suggestion is concise (within 3 bullet points), concrete, and to the point. Begin the review by stating your role."
    ),  # Set the system message describing the legal reviewer's role
)

# Create an instance of AssistantAgent for the ethics reviewer
ethics_reviewer = AssistantAgent(
    name="Ethics Reviewer",  # Name the agent as "Ethics Reviewer"
    llm_config=llm_config,  # Pass the language model configuration
    system_message=(
        "You are an ethics reviewer, known for your ability to ensure that content is ethically sound and free from any potential ethical issues. "
        "Make sure your suggestion is concise (within 3 bullet points), concrete, and to the point. Begin the review by stating your role."
    ),  # Set the system message describing the ethics reviewer's role
)

# Create an instance of AssistantAgent for the meta reviewer
meta_reviewer = AssistantAgent(
    name="Meta Reviewer",  # Name the agent as "Meta Reviewer"
    llm_config=llm_config,  # Pass the language model configuration
    system_message=(
        "You are a meta reviewer, you aggregate and review the work of other reviewers and give a final suggestion on the content."
    ),  # Set the system message describing the meta reviewer's role
)

# Define a function to create a reflection message for review
def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

# Create a list of review chat configurations for different reviewers
review_chats = [
    {
        "recipient": SEO_reviewer,  # SEO reviewer agent
        "message": reflection_message,  # Message function for review
        "summary_method": "reflection_with_llm",  # Summarization method
        "summary_args": {
            "summary_prompt": (
                "Return review into as JSON object only: {'Reviewer': '', 'Review': ''}. Here Reviewer should be your role"
            ),  # Prompt for summarizing the review
        },
        "max_turns": 1  # Maximum number of interaction turns
    },
    {
        "recipient": legal_reviewer,  # Legal reviewer agent
        "message": reflection_message,  # Message function for review
        "summary_method": "reflection_with_llm",  # Summarization method
        "summary_args": {
            "summary_prompt": (
                "Return review into as JSON object only: {'Reviewer': '', 'Review': ''}."
            ),  # Prompt for summarizing the review
        },
        "max_turns": 1  # Maximum number of interaction turns
    },
    {
        "recipient": ethics_reviewer,  # Ethics reviewer agent
        "message": reflection_message,  # Message function for review
        "summary_method": "reflection_with_llm",  # Summarization method
        "summary_args": {
            "summary_prompt": (
                "Return review into as JSON object only: {'reviewer': '', 'review': ''}"
            ),  # Prompt for summarizing the review
        },
        "max_turns": 1  # Maximum number of interaction turns
    },
    {
        "recipient": meta_reviewer,  # Meta reviewer agent
        "message": (
            "Aggregate feedback from all reviewers and give final suggestions on the writing."
        ),  # Message function for aggregation
        "max_turns": 1  # Maximum number of interaction turns
    },
]

# Register nested chats for the critic agent to handle the review process
critic.register_nested_chats(
    review_chats,
    trigger=writer,  # Trigger the review process when the writer completes their task
)

# Initiate the chat process with the critic agent, starting the review of the writer's task
res = critic.initiate_chat(
    recipient=writer,  # Start the chat with the writer agent
    message=task,  # The initial task to be performed
    max_turns=2,  # Maximum number of interaction turns for the chat
    summary_method="last_msg"  # Method to summarize the chat by the last message
)
