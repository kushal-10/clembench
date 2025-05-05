from engine.maps import BaseMap

import random
import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple


class ADEMap(BaseMap):

    def __init__(self, m: int = 3, n: int = 3, n_rooms: int = 9, seed: int = 42):
        """
        Set up a base 2-D graph whose nodes are based on ADE20k dataset.

        Args:
            m: Number of rows in the graph.
            n: Number of columns in the graph
            n_rooms: Required number of rooms. Should be less than n*m

        Raises:
            ValueError: If any value is unset
            AssertionError: If `n_rooms` > `n*m`
        """
        random.seed(seed)
        np.random.seed(seed)
        super().__init__(m, n, n_rooms)

    def assign_types(self, G,
                     json_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "categories.json"),
                     ambiguity: list[int] = None, use_outdoor_categories: bool = True):
        """
        Assign room categories and images to the nodes in the generated graph.
        Example G.nodes[room] = {
            'base_type': 'indoor', # or 'outdoor'
            'type': 'k/kitchen',
            'ambiguous': True - for all ambiguous cases or a random room in central area if ambiguity is None - else False
        }

        Args:
            G: Generated graph. (via BaseMap.create_acyclic_graph()/BaseMap.create_cyclic_graph())
            json_path: Path to a json file containing "targets", "outdoors" and "distractors" categories
            ambiguity: List of integers to control ambiguity. Example: [3,2] means - at
            least two types of potential target categories, and that
            the first will occur three times, and the second twice. Only applies to indoor rooms
            use_outdoor_categories: If true, include `outdoor` room categories

        Raises:
            ValueError: if possible indoor rooms (rooms with more than 1 neighbor) < sum of ambiguity
        """
        # Fix ambiguity = None case
        if not ambiguity:
            ambiguity = [1]

        outdoor_nodes = []  # Rooms with only one neighbour - Most likely will be used as Entry/Exit points
        indoor_nodes = []  # Most likely will be used as target/distractor rooms
        # Assign nodes based on number of neighbours
        for room in G.nodes():
            if G.degree(room) == 1:
                outdoor_nodes.append(room)
            elif G.degree(room) > 1:
                indoor_nodes.append(room)
            else:
                raise ValueError(f"Check Graph Generation!! Found a node with no neighbors!")

        if len(indoor_nodes) < sum(ambiguity):
            raise ValueError(
                f"Ambiguity passed is {ambiguity}, But number of indoor rooms in the generated graph is {len(indoor_nodes)}"
                f"Try decreasing ambiguity such that sum of ambiguity is <= {len(indoor_nodes)}"
                f"If ambiguity is strictly required and possible with the initialized Map, try generating another random graph.")

        
        with open(json_path, 'r') as f:
            categories = json.load(f)

        outdoor_nodes_assigned = []
        # Using distractors here, change to "targets" if required
        category_key = "outdoors" if use_outdoor_categories else "distractors"
        for room in outdoor_nodes:
            # ambiguity=None for outdoor rooms
            # TODO : Add ambiguity for outdoor rooms?

            random_outdoor_room = np.random.choice(categories[category_key])
            while random_outdoor_room in outdoor_nodes_assigned:
                random_outdoor_room = np.random.choice(categories[category_key])

            outdoor_nodes_assigned.append(random_outdoor_room)
            G.nodes[room]['base_type'] = "outdoor" # Used as an identifier for rooms with one neighbour
            G.nodes[room]['type'] = random_outdoor_room
            G.nodes[room]['ambiguous'] = False

            # TODO: Pass 'type' as argument as well, to pre set certain type of rooms
            # At least 2 home_office and 2 bedrooms, for example


        indoor_nodes_assigned = []
        room_type_assigned = [] if category_key!="targets" else outdoor_nodes_assigned
        
        for room_type in ambiguity:
            # TODO: Add check to see length of ambiguity <= len(categories["targets"])
            # Pick a random unique room category
            random_room_type = np.random.choice(categories["targets"])
            while random_room_type in room_type_assigned:
                random_room_type = np.random.choice(categories["targets"])
            room_type_assigned.append(random_room_type)

            room_count = 1
            for r in range(room_type):
                # Pick a random node from the list of indoor nodes and assign the selected room_category
                room = indoor_nodes[np.random.randint(len(indoor_nodes))]
                while room in indoor_nodes_assigned:
                    room = indoor_nodes[np.random.randint(len(indoor_nodes))]
                indoor_nodes_assigned.append(room)

                G.nodes[room]['base_type'] = "indoor"
                # Make the room_type unique
                # (so instead of assigning 'kitchen' to multiple nodes, assign 'kitchen1' and 'kitchen2')
                G.nodes[room]['type'] = str(random_room_type) + " " + str(room_count)
                room_count += 1
                G.nodes[room]['ambiguous'] = True # Set True for all ambiguous rooms

        # Remaining nodes - as distractors
        distractor_rooms = list(set(indoor_nodes) - set(indoor_nodes_assigned))

        # Collect remaining room_types from targets and aa categories["distractors"] to it
        all_targets = set(categories["targets"])
        all_distros = set(categories["distractors"])
        chosen_rooms = set(room_type_assigned)
        dist_categories = sorted((all_targets - chosen_rooms) | all_distros)

        distractor_assigned = []
        for room in distractor_rooms:
            random_distractor = np.random.choice(dist_categories)
            while random_distractor in distractor_assigned:
                random_distractor = np.random.choice(dist_categories)

            G.nodes[room]['base_type'] = "indoor"
            G.nodes[room]['type'] = random_distractor
            G.nodes[room]['ambiguous'] = False

        return G

    def assign_images(self, G,
                      json_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "images.json")):
        """
        Assign Images from ADE20k dataset to a graph whose nodes have already been assigned a specific room type

        Args:
            G: networkx type graph containing node info - {type, base_type, target}
            json_path: Path to a jsonn file containing mappping of room_types to various images

        Return:
            G: Graph with updated nodes with randomly assigned image of a specific room_type
        """

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        images_assigned = []
        for node in G.nodes():
            room_type = G.nodes[node]['type']
            room_type = room_type.split(" ")[0] # For ambiguous cases - remove assigned number
            random_image = np.random.choice(json_data[room_type])
            while random_image in images_assigned:
                random_image = np.random.choice(json_data[room_type])

            G.nodes[node]['image'] = random_image

        # TODO: ADE20k categories are well-defined. Can make MapWorld with real-world constraints
        # Maybe add constraints like "h/hallway" should be in between nodes of "indoor/targets" etc..
        # Or "s/street" should be strictly between "g/garage" and a category from "sports_and_leisure" or "transportation" etc..

        return G

    def print_mapping(self, G):
        """
        Print a mapping of node: room_type - image_url for all nodes in the graph
        """
        for this_node in G.nodes():
            print('{}: {} - {:>50}'.format(this_node,
                                           G.nodes[this_node]['type'],
                                           G.nodes[this_node]['image']))


    def plot_agent_graph(self, G, agent_pos, target_pos):
        """
        BUG - Fails when no outdoor room is present in the graph...

        Plot a graph showing locations of agent position and target room

        Args:
            G: networkx type graph
            agent_pos = Current positon of Agent - [x,y]/node
            target_pos = Position of Target room - [x,y]/node
        """
        node_labels = {node: G.nodes[node]['type'] for node in G.nodes()}
        pos = {n: n for n in G.nodes()}

        nx.draw_networkx_nodes(G, pos, node_shape='s', node_color='lightblue')
        nx.draw_networkx_edges(G, pos)

        label_pos = {k: (v[0], v[1] + 0.2) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels=node_labels, font_size=10)

        # Draw custom agent and target nodes over main graph
        nx.draw_networkx_nodes(G, pos, nodelist=[agent_pos], node_color='blue', node_shape='o', node_size=300)
        nx.draw_networkx_nodes(G, pos, nodelist=[target_pos], node_color='black', node_shape='s', node_size=600)

        custom_labels = {
            agent_pos: "agent",
            target_pos: "target"
        }
        custom_label_pos = {k: (v[0], v[1] - 0.2) for k, v in pos.items() if k in custom_labels}
        nx.draw_networkx_labels(G, custom_label_pos, labels=custom_labels, font_size=9, font_color='black')

        plt.axis('off')
        plt.show()

    @staticmethod
    def select_random_room(available_rooms: list, occupied: Tuple):
        """
        Pick a random room from a list of (available rooms - occupied)
        Args:
            available_rooms: List of available nodes that can be assigned a room
            occupied: A node that already has been assigned a room

        Returns:
            A random room chosen
        """

        if occupied in available_rooms:
            available_rooms.remove(occupied)
        return available_rooms[np.random.choice(len(available_rooms))]
    

    def set_positions(self, ambiguous_rooms: list, indoor_rooms: list, outdoor_rooms: list, region: str, occupied: Tuple = None) -> Dict:
        """
        Set agent position/target position.
        Based on the list of rooms provided and required area for the node

        Args:
            ambiguous_rooms: List of rooms assigned as ambiguous
            indoor_rooms: List of rooms assigned as indoor
            outdoor_rooms: List of rooms assigned as outdoor
            region: Area of room that needs to be assigned - indoor/outdoor/random/ambiguous
            occupied: Pass an already set position occupied as agent/target.

        Return:
            position/node: (x, y)
        """

        # For Random nodes, assign in any of the available nodes in the graph irrespective of target, area, category,...
        if region == "random":
            available_rooms = ambiguous_rooms + indoor_rooms + outdoor_rooms
            # Check if the node is already been assigned as a location for target node/agent position
            return self.select_random_room(available_rooms, occupied)

        if region == "indoor":
            if occupied in indoor_rooms:
                indoor_rooms.remove(occupied)

            if indoor_rooms:
                return indoor_rooms[np.random.choice(len(indoor_rooms))]
            else:
                print("No Indoor rooms avaialable! Using a random available room.")
                available_rooms = outdoor_rooms + ambiguous_rooms
                return self.select_random_room(available_rooms, occupied)

        if region == "outdoor":
            if occupied in outdoor_rooms:
                outdoor_rooms.remove(occupied)

            if outdoor_rooms:
                return outdoor_rooms[np.random.choice(len(outdoor_rooms))]
            else:
                print("No Outdoor rooms avaialable! Using a random available room.")
                available_rooms = indoor_rooms + ambiguous_rooms
                return self.select_random_room(available_rooms, occupied)
            
        if region == "ambiguous":
            if occupied in ambiguous_rooms:
                ambiguous_rooms.remove(occupied)
            
            if ambiguous_rooms:
                return ambiguous_rooms[np.random.choice(len(ambiguous_rooms))]
            else:
                print("No Ambiguous rooms avaialable! Using random available room.")
                available_rooms = outdoor_rooms + indoor_rooms
                return self.select_random_room(available_rooms, occupied)


    def metadata(self, G, start_type: str = "outdoor", end_type: str = "outdoor"):
        """
        Generate metadata for the Graph incl. start/end points

        Args:
            G: networkx type graph
            start_type = "outdoor"/"indoor"/"random"/"ambiguous" - Defining agent start position
            end_type = "outdoor"/"indoor"/"random"/"ambiguous" - Defining target room position
        """

        # Metadata values
        graph_id = ""
        named_nodes = []
        unnamed_nodes = []
        named_edges = []
        unnamed_edges = []
        node_to_category = {}
        category_to_node = {}
        node_to_image = {}
        category_to_image = {}

        # Additional values
        ambiguous_rooms = []
        indoor_rooms = []
        outdoor_rooms = []

        # Nodes Metadata
        for node in G.nodes():
            # Clean Node name
            node_name = G.nodes[node]['type']
            node_name = " ".join(node_name.split("__"))
            node_name = " ".join(node_name.split("_"))
            node_name = node_name.capitalize()

            # Create a graph id based on node position and node category
            graph_id += str(list(node)[0]) + str(list(node)[1]) + node_name[0].lower()

            named_nodes.append(node_name)
            unnamed_nodes.append(node)

            node_to_category[str(node)] = node_name
            category_to_node[node_name] = node
            node_to_image[str(node)] = G.nodes[node]['image']
            category_to_image[node_name] = G.nodes[node]['image']

            # Additional info
            if G.nodes[node]['ambiguous']:
                ambiguous_rooms.append(node)

            if G.nodes[node]['base_type'] == "indoor":
                indoor_rooms.append(node)

            if G.nodes[node]['base_type'] == "outdoor":
                outdoor_rooms.append(node)

        # Edge Metadata
        for edge in G.edges():
            named_edge = []
            for e in edge:
                named_edge.append(node_to_category[str(e)])
            named_edges.append(tuple(named_edge))
            unnamed_edges.append(edge)

        # Set Random start and Target positions
        # Can be overridden by the environment if required
        target_pos = self.set_positions(ambiguous_rooms, indoor_rooms, outdoor_rooms, end_type)
        start_pos = self.set_positions(ambiguous_rooms, indoor_rooms, ambiguous_rooms, start_type, target_pos)
        
        graph_metadata = {
            "graph_id": graph_id,
            "m": self.m,
            "n": self.n,
            "named_nodes": named_nodes,
            "unnamed_nodes": unnamed_nodes,
            "named_edges": named_edges,
            "unnamed_edges": unnamed_edges,
            "node_to_category": node_to_category,
            "category_to_node": category_to_node,
            "node_to_image": node_to_image,
            "category_to_image": category_to_image,
            "start_node": start_pos,
            "target_node": target_pos,
        }

        return graph_metadata


if __name__ == '__main__':
    ademap = ADEMap(4, 4, 10)
    G = ademap.create_cyclic_graph()
    G = ademap.assign_types(G, ambiguity=[3,3], use_outdoor_categories=False)
    G = ademap.assign_images(G)
    metadata = ademap.metadata(G)

    print(metadata)

