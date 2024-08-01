import os

from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent
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

executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
)

# # Agent with code executor configuration
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)
# # Agent with code writing capability
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_writer_agent_system_message = code_writer_agent.system_message

# print(code_writer_agent_system_message)


today = datetime.datetime.now().date()
# message = f"Today is {today}. "\
# "Create a plot showing stock gain YTD for NVDA and TLSA. "\
# "Make sure the code is in markdown code block and save the figure"\
# " to a file ytd_stock_gains.png."""

# chat_result = code_executor_agent.initiate_chat(
#     code_writer_agent,
#     message=message,
# )

# User-Defined Functions
# Instead of asking LLM to generate the code for downloading stock data and plotting charts each time, you can define functions for these two tasks and have LLM call these functions in the code.
def get_stock_prices(stock_symbols, start_date, end_date):
    """Get the stock prices for the given stock symbols between
    the start and end dates.

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format 
        'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
    
    Returns:
        pandas.DataFrame: The stock prices for the given stock
        symbols indexed by date, with one column per stock 
        symbol.
    """
    

    stock_data = yfinance.download(
        stock_symbols, start=start_date, end=end_date
    )
    return stock_data.get("Close")


def plot_stock_prices(stock_prices, filename):
    """Plot the stock prices for the given stock symbols.

    Args:
        stock_prices (pandas.DataFrame): The stock prices for the 
        given stock symbols.
    """
    

    plt.figure(figsize=(10, 5))
    for column in stock_prices.columns:
        plt.plot(
            stock_prices.index, stock_prices[column], label=column
                )
    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig(filename)
    

executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    functions=[get_stock_prices, plot_stock_prices],
    )

code_writer_agent_system_message += executor.format_functions_for_prompt()
print(code_writer_agent_system_message)


code_writer_agent = ConversableAgent(
    name="code_writer_agent",
    system_message=code_writer_agent_system_message,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=f"Today is {today}."
    "Download the stock prices YTD for NVDA and TSLA and create"
    "a plot. Make sure the code is in markdown code block and "
    "save the figure to a file stock_prices_YTD_plot.png.",
)