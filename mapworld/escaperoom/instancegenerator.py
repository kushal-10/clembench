from clemcore.clemgame import GameInstanceGenerator
from engine.ade_maps import ADEMap

import os
import numpy as np
from typing import Tuple

# TODO: Move this to initial_prompt.template under resources
EXPL_PROMPT = "You are stuck in a mapworld environment. Your task is to explore this world and reach an escape room.\nStart by describing the image given to you that represents the current room you are in.\nYou can also make moves to the following rooms - $ROOMS, here the first part is the room name and the second part is the direction which leads to\nthe room. You are allowed to respond only in the following format.\n\n{\"description\": A one line description of the current room you are in, \"moves\": [list of tuples of possible moves to rooms]}",
EXPL_REPROMPT = "Now we made a move to this room, and you can move to these rooms $ROOMS, provide the description and moves.",
GUIDE_PROMPT = "I need your help, I am stuck in a mapworld environment. Your task is to help me reach an escape room.\nI do not know what the escape room looks like. But fortunately, you have an image of the escape room with you.\nI will explore each room here and give you a description and possible moves in the following format:\n\n{\"description\": A one line description of the current room I am in, \"rooms\": [list of tuples of possible moves to rooms]}\n\nYour task is to compare the description of my room with the image of the room you have been given. Then you have two options\nOption 1) If my description matches the image of the room that you have respond with - {'move': 'escape'} as a string\nOption 2) If my description does not match the image that you have been given, then only respond in the following format \n\n{'move': 'possible_move'}, Here possible_move can be one of the {north, south, east, west}\n\nHere is my initial Description and possible moves :\n$DESCRIPTION\n"

N = 10 # Number of instances per experiment

SEED = 10 # LM10
# np.random.seed(SEED)

class EscapeRoomInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(os.path.abspath(__file__)))


    def on_generate(self):
        experiments = {
            "beginner": {"size": 3, "rooms":4, "cycle": False, "ambiguity": None},
            "semi-pro": {"size": 4, "rooms": 5, "cycle": True, "ambiguity": [2]},
            "pro": {"size": 5, "rooms": 10, "cycle": True, "ambiguity": [2,2]},
            # "expert": {"size": 6, "rooms": 15, "cycle": True, "ambiguity": [3,3]},
        }

        for exp in experiments.keys():
            experiment = self.add_experiment(exp)
            game_id = 0
            size = experiments[exp]["size"]
            rooms = experiments[exp]["rooms"]
            cycle = experiments[exp]["cycle"]
            ambiguity = experiments[exp]["ambiguity"]
            for i in range(N):
                ade_map = ADEMap(size, size, rooms)

                if cycle:
                    ade_graph = ade_map.create_cyclic_graph()
                else:
                    ade_graph = ade_map.create_acyclic_graph()
                ade_graph = ade_map.assign_types(ade_graph, ambiguity=ambiguity)
                ade_graph = ade_map.assign_images(ade_graph)

                map_metadata = ade_map.metadata(ade_graph)
                map_metadata["explorer_prompt"] = EXPL_PROMPT
                map_metadata["guide_prompt"] = GUIDE_PROMPT
                map_metadata["explorer_reprompt"] = EXPL_REPROMPT

                escape_room_instance = self.add_game_instance(experiment, game_id)

                for orig_k, orig_v in map_metadata.items():
                    # convert tuple keys to strings
                    if isinstance(orig_k, tuple):
                        k = str(orig_k)
                    else:
                        k = orig_k

                    # convert tuple values to strings (if you want)
                    if isinstance(orig_v, tuple):
                        v = str(orig_v)
                    else:
                        v = orig_v

                    escape_room_instance[k] = v

                game_id += 1



if __name__ == '__main__':
    EscapeRoomInstanceGenerator().generate()
