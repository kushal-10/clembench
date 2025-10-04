## MapWorld Engine

MapWorld is a game engine designed for evaluating multimodal cLLMs (Chat Optimized Large Language Models  | clems) in a room-to-room navigational setting. 

This environment is based on [ADE20K dataset](https://ade20k.csail.mit.edu/) (for realistic room images and categories) and [Sempix Mapworld](https://github.com/clp-research/sempix/tree/master/03_Tasks/MapWorld) for creation of different graph topologies and assigning images/categories to each node. We use a custom [Gymnasium](https://gymnasium.farama.org/) based environment to control the navigation of the agent(s)

## MapWorld Features
```python


M, N, N_ROOMS = A grid of MxN consisting of N_ROOMS.
Node - Defined as any grid point/cell in a given MxN grid.
Room - A node where a room has already been formed. 
Graph - A grid configuration of N_ROOMS of a particular TYPE (layout) with exactly 1 connected component 
        i.e. every room can be visited from every other room in a given Graph
        Here TYPE can be one of - {random, random cycle, tree, path, cycle, generalized star, ladder}
Map - A Graph with assigned room types and images from ADE 20K dataset.

All graph types can be formed by creating their nx counterparts, but the assignment to a Grid like structure
becomes more complex, so create each graph type here
```

### Quick Start

```python
from engine.ade_maps import ADEMap
from engine.environment import MapWorldEnv

n, m = 3, 3 # Grid Size
rooms = 4 # Number of Rooms

# Create an acyclic graphs
ademap = ADEMap(m, n, rooms) 
graph_a = ademap.create_acyclic_graph()

# Assign room categories from resources/categories.json and assign ambiguity
graph_a = ademap.assign_types(graph_a, ambiguity=[2], use_outdoor_categories=False)
# Here ambiguity refers to how many similar rooms should be assigned in the map
# Ambiguity of [2] refers to - We need 2 rooms of the same type
# Ambiguity of [2,3] refers - We need 2 rooms of Type1 and 3 rooms of Type2
# Example1 - with 4 rooms and ambiguity of [2], we might get Kitchen, Shower, Bedroom 1 and Bedroom 2

# Assign random images from resources/images.json with a random start and random target position
graph_a = ademap.assign_images(graph_a)
metadata = ademap.metadata(graph_a, "random", "random")

# Instantiate environment
env = MapWorldEnv(render_mode="human", size=5, map_metadata=metadata)

# Run an example episode with random moves
env.reset()
env.render()
print(env._agent_location)
for i in range(10):
    env.render()
    random_action = env.action_space.sample()
    env.step(random_action)
env.close()
```
