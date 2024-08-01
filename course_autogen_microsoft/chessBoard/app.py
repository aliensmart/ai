import os  # Import the os module to interact with the operating system

from dotenv import load_dotenv  # Import the load_dotenv function from the dotenv module
from autogen import ConversableAgent, register_function  # Import necessary classes and functions from the autogen module
import pprint  # Import the pprint module for pretty-printing data structures
import chess  # Import the chess module for handling chess game logic
import chess.svg  # Import the chess.svg module for rendering chess boards as SVG
from typing_extensions import Annotated  # Import Annotated for type hinting

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

# Initialize a chess board
board = chess.Board()
made_move = False  # Flag to indicate if a move has been made

# Function to get legal moves in UCI format
def get_legal_moves() -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Possible moves are: " + ",".join(
        [str(move) for move in board.legal_moves]
    )

# Function to make a move in UCI format
def make_move(
    move: Annotated[str, "A move in UCI format."]
) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)  # Convert the move from UCI format
    board.push(move)  # Push the move onto the board
    global made_move
    made_move = True  # Set the made_move flag to True

    # Display the board with the move highlighted
    display(
        chess.svg.board(
            board,
            arrows=[(move.from_square, move.to_square)],
            fill={move.from_square: "gray"},
            size=200
        )
    )

    # Get the piece name and symbol
    piece = board.piece_at(move.to_square)
    piece_symbol = piece.unicode_symbol()
    piece_name = (
        chess.piece_name(piece.piece_type).capitalize()
        if piece_symbol.isupper()
        else chess.piece_name(piece.piece_type)
    )
    return f"Moved {piece_name} ({piece_symbol}) from "\
           f"{chess.SQUARE_NAMES[move.from_square]} to "\
           f"{chess.SQUARE_NAMES[move.to_square]}."

# Create an instance of ConversableAgent for the white player
player_white = ConversableAgent(
    name="Player White",
    system_message="You are a chess player and you play as white. "
                   "First call get_legal_moves() to get a list of legal moves. "
                   "Then call make_move(move) to make a move. "
                   "After a move is made, chitchat to make the game fun.",
    llm_config=llm_config,
)

# Create an instance of ConversableAgent for the black player
player_black = ConversableAgent(
    name="Player Black",
    system_message="You are a chess player and you play as black. "
                   "First call get_legal_moves() to get a list of legal moves. "
                   "Then call make_move(move) to make a move. "
                   "After a move is made, chitchat to make the game fun.",
    llm_config=llm_config,
)

# Function to check if a move has been made
def check_made_move(msg):
    global made_move
    if made_move:
        made_move = False
        return True
    else:
        return False

# Create an instance of ConversableAgent for the board proxy
board_proxy = ConversableAgent(
    name="Board Proxy",
    llm_config=False,
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="NEVER",
)

# Register functions for getting legal moves and making moves with the player agents
for caller in [player_white, player_black]:
    register_function(
        get_legal_moves,
        caller=caller,
        executor=board_proxy,
        name="get_legal_moves",
        description="Get legal moves.",
    )

    register_function(
        make_move,
        caller=caller,
        executor=board_proxy,
        name="make_move",
        description="Call this tool to make a move.",
    )

# Set tools configuration for player agents
player_black.llm_config["tools"]

# Register nested chats to handle the interaction between player agents
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_white,
            "summary_method": "last_msg",
            "silent": True,
        }
    ],
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[
        {
            "sender": board_proxy,
            "recipient": player_black,
            "summary_method": "last_msg",
            "silent": True,
        }
    ],
)

# Initialize the chess board
board = chess.Board()

# Start the chat with player_black initiating the game
chat_result = player_black.initiate_chat(
    player_white,
    message="Let's play chess! Your move.",
    max_turns=2,
)
