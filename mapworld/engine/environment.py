## Working Grid environment

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import Dict, Tuple

import os

from engine.ade_maps import ADEMap

class MapWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str = "human", size: int = 5, map_metadata: Dict = None,
                 agent_pos: np.array = None, target_pos: np.array = None):
        """
        Initialize mapworld as a Gymnasium environment.

        Args:
            render_mode: Type of rendering mode to use - human (renders the observation as a plot), rgb_array (returns an array without any plots)
            size: Grid size of the mapworld environment
            map_metadata: metadata from the ADEMap class for a graph
            agent_pos: Agent start room on the mapworld environment, assigns a random outdoor room if agent_pos is None
            target_pos: Target room on the mapworld environment (Think Escaperoom Base version),
                        assigns a random outdoor room if target_pos is None
        """

        self.size = size  # The size of the square grid
        self.window_size = 500  # The size of the PyGame window
        self.map_metadata = map_metadata

        self.agent_pos = agent_pos if agent_pos is not None else np.array(self.map_metadata["start_node"])
        self.target_pos = target_pos if target_pos is not None else np.array(self.map_metadata["target_node"])

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2
        # Ref - https://gymnasium.farama.org/introduction/create_custom_env/
        # Ref - https://gymnasium.farama.org/introduction/basic_usage/#action-and-observation-spaces
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 5 actions, corresponding to "east", "north", "west", "south", "explore", and "escape"
        self.action_space = spaces.Discrete(6)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "down" etc.
        """

        self._action_to_direction = {
            0: np.array([1, 0]),  # East
            1: np.array([0, 1]),  # South
            2: np.array([-1, 0]), # West
            3: np.array([0, -1]), # North
            4: np.array([0, 0]),  # Explore
            5: np.array([1, 1])   # Escape 
        }

        self._action_to_move = {
            0: "east",
            1: "south",
            2: "north",
            3: "west",
            4: "<explore>",
            5: "<escape>"
        }

        self.move_to_action = {
            "east": 0,
            "south": 1,
            "north": 2,
            "west": 3,
            "<explore>": 4,
            "<escape>": 5
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self._agent_location = self.agent_pos
        self._target_location = self.target_pos

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        
        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        # An episode is done if the guide agent has generated the <escape> token
        terminated = 0
        reward = 0
        if self._action_to_move[action] == "<escape>":
            terminated = 1
            reward = 1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_rect(self, canvas, color, pos, pix_square_size, room_ratio, label):

        # Define rectangle dimensions
        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(
                (pos*pix_square_size + ((1-room_ratio)/2)*pix_square_size), # (left, top)
                (room_ratio*pix_square_size, room_ratio*pix_square_size), # (width, height)
            ),
        )


        text_pos = pos*pix_square_size + pix_square_size/2
        text_pos = [text_pos[0], text_pos[1] - pix_square_size/2 + 10]
        text_surf = self.font.render(str(label), True, (0, 0, 0))
        # center it in the cell
        text_rect = text_surf.get_rect(center=text_pos)
        canvas.blit(text_surf, text_rect)


    def _draw_line(self, canvas, color, edge, pix_square_size, room_ratio):
        start_pos = np.array(edge[0])*pix_square_size + pix_square_size/2
        end_pos = np.array(edge[1])*pix_square_size + pix_square_size/2

        if edge[0][0] < edge[1][0]:
            # Horizontal edges
            start_pos[0] = start_pos[0] + room_ratio/2
            end_pos[0] = end_pos[0] - room_ratio/2
        elif edge[0][0] > edge[1][0]:
            start_pos[0] = start_pos[0] - room_ratio / 2
            end_pos[0] = end_pos[0] + room_ratio / 2
        elif edge[0][1] < edge[1][1]:
            # Vertical Edges
            start_pos[1] = start_pos[1] + room_ratio/2
            end_pos[1] = end_pos[1] - room_ratio/2
        else:
            start_pos[1] = start_pos[1] - room_ratio / 2
            end_pos[1] = end_pos[1] + room_ratio / 2

        pygame.draw.line(canvas, color, start_pos, end_pos)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 10)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = int(
            self.window_size / self.size
        )  # The size of a single grid square in pixels
        room_ratio = 0.6 # Ratio of a visible room pixel wrt pix_square_size

        # Draw edges
        for edge in self.map_metadata["unnamed_edges"]:
            self._draw_line(canvas, (0, 0, 0), edge, pix_square_size, room_ratio)

        # Draw Pieces
        for node in self.map_metadata["unnamed_nodes"]:
            self._draw_rect(canvas, (255,0,0), np.array(node), pix_square_size,
                            room_ratio, self.map_metadata["node_to_category"][node])

        # Now we draw the agent
        self.robot_img = pygame.image.load(os.path.join("engine", "resources", "robot.png")).convert_alpha()
        # scale it to cell size
        self.robot_img = pygame.transform.smoothscale(
            self.robot_img, (room_ratio*pix_square_size, room_ratio*pix_square_size)
        )

        # draw the robot
        canvas.blit(self.robot_img, self._agent_location*pix_square_size + 0.2*pix_square_size)

        self.window.blit(canvas, (0, 0))
        pygame.display.flip()

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    n, m = 3, 3
    rooms = 4
    ademap = ADEMap(m, n, rooms)
    graph_a = ademap.create_acyclic_graph()
    graph_a = ademap.assign_types(graph_a, ambiguity=[2], use_outdoor_categories=False)
    graph_a = ademap.assign_images(graph_a)
    metadata = ademap.metadata(graph_a, "random", "random")

    env = MapWorldEnv(render_mode="human", size=5, map_metadata=metadata)

    env.reset()
    env.render()
    print(env._agent_location)
    for i in range(10):
        env.render()
        random_action = env.action_space.sample()
        env.step(random_action)
        print(env._agent_location, random_action)
    env.close()