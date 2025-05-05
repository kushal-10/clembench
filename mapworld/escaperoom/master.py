"""
Game Master for Escape Room
Implementing a base variant for now...2-player game only
"""

import ast
from clemcore.clemgame import Player, GameMaster, GameBenchmark, DialogueGameMaster, GameScorer, GameSpec
from clemcore.clemgame import metrics as ms
from clemcore.backends import Model
from clemcore.utils import file_utils
from engine.utils import get_next_moves, get_next_image, get_node

from typing import List, Dict, Tuple, Union
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("escaperoom.master")

class Explorer(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self.response: str = ""
        self.tag: str = "Explorer"

    def _custom_response(self, context: Dict) -> str:
        return "{'description': 'A room with a dining table and 4 chairs', 'moves': [('living_room', 'north'), ('shower', 'east')]}"
    
class Guide(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self.response: str = ""
        self.tag: str = "Guide"


    def _custom_response(self, context: Dict) -> str:
        return "{'move': 'north'}"

class EscapeRoom(DialogueGameMaster):

    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(name, path, experiment, player_models)

        # Initialize Experiment/Prompts/Scorer flags
        self.experiment: str = experiment["name"]

        # Scorers
        self.aborted = False
        # Analyse each player individually
        self.explorer_aborted = False
        self.guide_aborted = False


    def _on_setup(self, **game_instance):
        # Initialize Game Instance
        self.game_instance = game_instance

        # Prompts
        self.explorer_prompt: str = self.game_instance["explorer_prompt"]
        self.explorer_reprompt: str = self.game_instance["explorer_reprompt"]
        self.guide_prompt: str = self.game_instance["guide_prompt"]

        # Initialize Players
        # Player 1 (Explorer) is in the mapworld
        # Player 2 (Guide) is outside the world
        self.explorer = Explorer(self.player_models[0])
        self.guide = Guide(self.player_models[1])

        # Setup for Explorer/Player1
        self.explorer_pos = self.game_instance["start"]
        self.explorer_image = self.game_instance["images"][self.explorer_pos]
        # Keep the nodes and edges as str in master (straightforward mapping) but pass as Tuples to the mapworld engine
        moves = get_next_moves(self.game_instance, ast.literal_eval(self.explorer_pos))
        # Name of the room category - bedroom, for example
        self.explorer_room = self.game_instance["mapping"][self.explorer_pos]
        # Set possible Moves for Explorer
        # TODO: Do this via API (for each possible direction, get the response and let the explorer build the list)
        self.explorer_prompt = self.explorer_prompt.replace("$ROOMS", moves)
        self.explorer_target = self.game_instance["target"]

        # Setup for Guide/Player2
        self.escape_pos = self.game_instance["target"]
        self.escape_room = self.game_instance["mapping"][self.escape_pos]
        self.guide_image = self.game_instance["images"][self.escape_pos]

        # Add players
        # NOTE: Player calls will be made in the order below
        self.add_player(self.explorer)
        self.add_player(self.guide)

    def _on_before_game(self):
        """
        Pass initial message - first player (Explorer), first turn
        """

        # Add initial prompt to Explorer in (Explorer's) history
        self.set_context_for(self.explorer, self.explorer_prompt, image=[self.explorer_image])
        stdout_logger.info(f"First message for Explorer: {self.explorer_prompt}")
        stdout_logger.info(f"First Room image path for Explorer: {self.explorer_image}")

    def _does_game_proceed(self) -> bool:
        """
        Fail cases for each turn, use init_flags/scorers etc...
        """
        if self.aborted or self.current_round==10:
            return False
        else:
            return True

    @staticmethod
    def clean_agent_response(response: str) -> str:
        """
        Remove leading and trailing markdown wrappers from response
        """
        response = response.strip()
        response = response.replace("```json", "")
        response = response.replace("```", "")
        return response


    def _validate_player_response(self, player, utterance: str) -> bool:
        """
        Check Correct format/ tag etc... in each Player's response
        Args:
            player: Player object - Explorer or Guide type
            utterance: str - response from Player
        Returns:
            True if response format is valid, False otherwise
        """

        stdout_logger.info(f"Player response {player.tag}: {utterance}")
        utterance = self.clean_agent_response(utterance)
        stdout_logger.info(f"Cleaned Player response {player.tag}: {utterance}")

        if not utterance.startswith('{') or not utterance.endswith('}'):
            self.aborted = True
            self.log_to_self(f"invalid format", f"{player.tag}")
            return False

        response_dict = ast.literal_eval(utterance)

        if type(response_dict) != dict:
            self.aborted = True
            if type(player) == Explorer:
                self.explorer_aborted = True
            else:
                self.guide_aborted = True
            stdout_logger.info(f"Invalid Response Type: Expected Dictionary, got {type(response_dict)}: {response_dict}")
            self.log_to_self(f"invalid format", f"{player.tag}")
            return False

        keys = response_dict.keys()
        if type(player) == Explorer:
            """
            Explorer should respond only in the following format
            {'description': 'A room with a table and 4 chairs', 'moves': [('living_room', 'north'), ('shower', 'east')]}.
            Check for each key, value and value type 
            """

            # Check Keys
            if "description" not in keys or "moves" not in keys:
                self.aborted = True
                stdout_logger.info(f"Invalid response format for Explorer - Invalid Keys: {utterance}")
                self.log_to_self("invalid format", "explorer") # Violated request count
                return False

            # Check Value types
            if type(response_dict["description"]) != str or type(response_dict["moves"]) != list:
                self.aborted = True
                stdout_logger.info(f"Invalid response format for Explorer - Invalid Value Types: {utterance}")
                self.log_to_self("invalid format", "explorer") # Violated request count
                return False

            # Check each move value
            for m in response_dict["moves"]:
                if type(m) != tuple or len(m) != 2 or m[1].lower() not in ['north', 'south', 'east', 'west']:
                    self.aborted = True
                    stdout_logger.info(f"Invalid response for Explorer - Invalid Move: {utterance}")
                    self.log_to_self("invalid value", "explorer") # Parsed request count
                    return False

            # Success case
            self.log_to_self("success", "explorer")
            return True

        else:
            """
            Guide should respond only in the following format
            {'move': 'north'} 
            """
            # Check if 'move' key exists
            if 'move' not in response_dict.keys():
                self.aborted = True
                stdout_logger.info(f"Invalid Response Key for Guide: Expected 'move', got {list(response_dict.keys())}")
                self.log_to_self("invalid format", "guide") # Violated request count
                return False

            # Check if 'move' key has a valid value
            # HACK: This should also handle any parsing errors
            if response_dict["move"].lower() not in ['north', 'south', 'east', 'west']:
                self.aborted = True
                stdout_logger.info(f"Invalid Response Value for Guide: Expected one of ['north', 'south', 'east', 'west'], got {response_dict['move']}")
                self.log_to_self("invalid value", "guide") # parsed request count
                return False

            # Success case
            self.log_to_self("success", "guide")
            return True

    def _parse_response(self, player: Union[Explorer, Guide], utterance: str) -> str:
        """
        Modify the response from Guide and send it to the Explorer
        Args:
            player: Player object - Explorer or Guide type
            utterance: str - response from current Player

        NOTE: _validate_player_response is called before _parse_response in step()
        """
        # TODO: If player is guide - pass the reprompt version with rooms to Guide - Do this in parse resp
        # FIXME: Does not seem to be required, Is the utterance stored before or after this call
        # FIXME: If after, then the utterance can be overridden here as reprompt for explorer
        # if type(player) == Explorer:
        #     # Return utterance as is, and forward to Guide
        #     return utterance
        # elif type(player) == Guide:
        #     move_dict = ast.literal_eval(utterance)
        #     move = move_dict['move']
        #     self.explorer_image = get_next_image(self.game_instance, self.explorer_pos, move)
        #     next_node = get_node(ast.literal_eval(self.explorer_pos), move)
        #     next_moves = get_next_moves(self.game_instance, next_node)
        #     self.explorer_reprompt = self.explorer_reprompt.replace("$ROOMS", next_moves)
        return utterance


    def _on_valid_player_response(self, player: Union[Explorer, Guide], utterance: str):
        """
        Send Explorer's response to Guide and vice versa

        Args:
            player: Player object - Explorer or Guide type
            utterance: str - response from current Player
        """

        # First explorer turn is done, the response from explorer always goes into guide, unchanged
        # The guide response never goes into the Explorer, rather the reprompt of explorer is fixed
        # and the next possible moves are interpreted based on the guide's response

        if type(player) == Explorer:
            stdout_logger.info(f"Current Round index: {self.current_round}")
            utterance = self.clean_agent_response(utterance)
            if self.current_round==0: # First Guide Prompt
                self.guide_prompt = self.guide_prompt.replace("$DESCRIPTION", utterance)
                stdout_logger.info(f"First prompt for Guide: {self.guide_prompt}")
                stdout_logger.info(f"Image for Guide: {self.guide_image}")
                # Pass the response from Explorer to Guide - wo image
                self.set_context_for(self.guide, utterance, image=[self.guide_image])
            else:
                # Pass the response from Explorer as is
                self.set_context_for(self.guide, utterance)
        else:
            move = ast.literal_eval(utterance)['move'] # Parse validation is already handled in _validate_player_response
            self.explorer_image = get_next_image(self.game_instance, self.explorer_pos, move)
            next_node = get_node(ast.literal_eval(self.explorer_pos), move)
            next_moves = get_next_moves(self.game_instance, next_node)
            self.explorer_reprompt = self.explorer_reprompt.replace("$ROOMS", next_moves)
            self.set_context_for(self.explorer, self.explorer_reprompt, image=[self.explorer_image]) # Pass the updated str
            # Update explorer pos
            self.explorer_pos = str(next_node)


    def _on_after_game(self):
        # record final results once game episode has ended:
        pass


class EscapeRoomScorer(GameScorer):
    """
    Scorer class for Escape Room Game
    """

    def __init__(self, game_name:str, experiment:Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """
        Temp method to compute scores for Escape Room Game - Binary success rate
        """

        aborted = True
        # skip player specific eval for now
        success_explorer = False
        success_guide = False
        all_turn_scores = []
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score_dict = {
                "request_count": 0,
                "violated_request_count": 0,
                "parsed_request_count": 0,
            }

            # Walk through log_to_self items from DGM
            for event in turn:
                action = event["action"]
                turn_score_dict["request_count"] += 1
                if action["type"] == "invalid format":
                    turn_score_dict["violated_request_count"] += 1
                elif action["type"] == "invalid value":
                    turn_score_dict["parsed_request_count"] += 1
                elif action["type"] == "success":
                    aborted = False
                    turn_score_dict["parsed_request_count"] += 1

            # log turn request scores
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score_dict["request_count"])
            all_turn_scores.append(turn_score_dict)

        # Log episodic scores
        ep_request_count = 0
        ep_violated_request_count = 0
        ep_parsed_request_count = 0
        for s in all_turn_scores:
            ep_request_count += s["request_count"]
            ep_violated_request_count += s["violated_request_count"]
            ep_parsed_request_count += s["parsed_request_count"]

        self.log_episode_score(ms.METRIC_REQUEST_COUNT, ep_request_count)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, ep_violated_request_count)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, ep_parsed_request_count)

        if aborted:
            self.log_episode_score(ms.METRIC_ABORTED, 1)
            self.log_episode_score(ms.METRIC_SUCCESS, 0)
            self.log_episode_score(ms.METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(ms.BENCH_SCORE, np.nan)  # metric not applicable
            self.log_episode_score("Player Score", np.nan)
        else:
            self.log_episode_score(ms.METRIC_ABORTED, 0)
            self.log_episode_score(ms.METRIC_SUCCESS, 1)
            self.log_episode_score(ms.METRIC_LOSE, 0)
            self.log_episode_score(ms.BENCH_SCORE, 100)
            self.log_episode_score("Player Score", 100)


class EscapeRoomBenchmark(GameBenchmark):
    """
    Integrate this game in the overall benchmark runs
    """

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return EscapeRoom(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return EscapeRoomScorer(self.game_name, experiment, game_instance)