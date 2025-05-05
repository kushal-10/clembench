"""
Post Processing methods after instances are generated
Util functions for mm_mapworld....
"""
from typing import Dict, Tuple
import ast

def get_direction(start_pos: Tuple, next_pos: Tuple) -> str:
    """
    Get the direction of next move
    Args:
        start_pos: current node of the agent inside mapworld
        next_pos: next possible node of the agent inside mapworld

    Returns:
        direction: direction of the next move as a string item
    """
    if next_pos[0] == start_pos[0] and next_pos[1] == start_pos[1] + 1:
        return "north"
    elif next_pos[0] == start_pos[0] and next_pos[1] == start_pos[1] - 1:
        return "south"
    elif next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0] + 1:
        return "east"
    elif next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0] - 1:
        return "west"
    else:
        raise ValueError("Invalid move! Check the node positions!")


def get_node(start_pos: Tuple, move: str) -> Tuple:
    """
    Get the node of a move
    Args:
        start_pos: current node of the agent inside mapworld
        move: move as a string item

    Returns:
        node: node of the move as a string item
    """
    if move == "north":
        return start_pos[0], start_pos[1] + 1
    elif move == "south":
        return start_pos[0], start_pos[1] - 1
    elif move == "east":
        return start_pos[0] + 1, start_pos[1]
    elif move == "west":
        return start_pos[0] - 1, start_pos[1]
    else:
        raise ValueError("Invalid move! Check the parsed response!")


def get_next_moves(game_instance: Dict, agent_pos: Tuple) -> str:
    """
    Return a list of next possible rooms given the current room.

    Args:
        game_instance: a dictionary containing the map information/ game instance
        agent_pos: current node of the agent inside mapworld
    Returns:
        moves: list of next possible rooms as a string item
    """

    moves = []
    edges = [ast.literal_eval(e) for e in game_instance['edges']]
    for e in edges:
        if e[0] == agent_pos:
            direction = get_direction(agent_pos, e[1])
            room = game_instance['mapping'][str(e[1])]
            moves.append((room, direction))

    return str(moves)


def get_next_image(game_instance: Dict, agent_pos: Tuple, move: str) -> str:
    """
    Return the next image path - given the current room and next move.
    Args:
        game_instance: a dictionary containing the map information/ game instance
        agent_pos: current node of the agent inside mapworld
        move: next move from - {north, south, east, west}

    Returns:
        New image path as a string item after the move
    """
    next_node = get_node(agent_pos, move)
    return game_instance['images'][str(next_node)]


if __name__ == '__main__':
    game_inst = {
        "game_id": 0,
        "nodes": [
            "(0, 0)",
            "(0, 1)",
            "(1, 2)",
            "(1, 1)",
            "(2, 1)"
        ],
        "edges": [
            "((0, 0), (0, 1))",
            "((0, 1), (0, 0))",
            "((0, 1), (1, 1))",
            "((1, 1), (0, 1))",
            "((1, 1), (2, 1))",
            "((2, 1), (1, 1))",
            "((1, 1), (1, 2))",
            "((1, 2), (1, 1))"
        ],
        "images": {
            "(0, 0)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/urban/street/ADE_train_00016858.jpg",
            "(0, 1)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/reception/ADE_train_00015716.jpg",
            "(1, 1)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/bedroom/ADE_train_00000526.jpg",
            "(1, 2)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/urban/casino__outdoor/ADE_train_00005212.jpg",
            "(2, 1)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/shower/ADE_train_00016280.jpg"
        },
        "mapping": {
            "(0, 0)": "street",
            "(0, 1)": "reception",
            "(1, 1)": "bedroom",
            "(1, 2)": "casino",
            "(2, 1)": "shower"
        },
        "start": "(0, 0)",
        "target": "(1, 2)",
        "explorer_prompt": "You are stuck in a mapworld environment. Your task is to explore this world and reach an escape room.\nStart by describing the image given to you that represents the current room you are in.\nYou can also make moves to the following rooms - $ROOMS, here the first part is the room name and the second part is the direction which leads to\nthe room. You are allowed to respond only in the following format.\n\n{\"description\": A one line description of the current room you are in, \"moves\": [list of tuples of possible moves to rooms]}\n\nYou are to return this JSON format/dict format text as a string. DO NOT RESPOND with anything else.",
        "explorer_reprompt": "Now we made a move to this room, and you can move to these rooms $ROOMS, provide the description and moves.",
        "guide_prompt": "I need your help, I am stuck in a mapworld environment. Your task is to help me reach an escape room.\nI do not know what the escape room looks like. But fortunately, you have an image of the escape room with you.\nI will explore each room here and give you a description and possible moves in the following format:\n\n{\"description\": A one line description of the current room I am in, \"rooms\": [list of tuples of possible moves to rooms]}\n\nYour task is to compare the description of my room with the image of the room you have been given. Then you have two options\nOption 1) If my description matches the image of the room that you have respond with - {'move': 'escape'} as a string\nOption 2) If my description does not match the image that you have been given, then respond in the following format as a string\n\n{'move': 'possible_move'}, Here possible_move can be one of the {north, south, east, west}\n\nHere is my initial Description:\n$DESCRIPTION\n"
    }

    move = get_next_moves(game_inst, (1,2))
    print(move)