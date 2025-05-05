import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class BaseMap:

    def __init__(self, m: int = 3, n: int = 3, n_nodes: int = 9):
        """
        Set up a base 2-D graph, assign cycles in the graph if required

        Args:
            m: Number of rows in the graph.
            n: Number of columns in the graph
            n_nodes: Required number of nodes. Should be less than n*m

        Raises:
            ValueError: If any value is unset
            AssertionError: If `n_nodes` > `n*m`
        """
        if not m or not n or not n_nodes:
            raise ValueError(
                f"One of the values passed - m : {m}, n: {n}, n_nodes: {n_nodes} is not set. (n,m,n_nodes) >= 1")
        assert n_nodes <= m * n, "Number of nodes cannot exceed grid size"

        self.m = m
        self.n = n
        self.n_nodes = n_nodes

    @staticmethod
    def get_valid_neighbors(current_pos: np.array = [0, 0], visited: list = [], m: int = 3, n: int = 3):
        """
        Get a list of valid nodes, to which a step can be taken from the current node
        """

        possible_moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        valid_neighbors = []
        for move in possible_moves:
            next_pos = [current_pos[0] + move[0], current_pos[1] + move[1]]

            if 0 <= next_pos[0] < m and 0 <= next_pos[1] < n and tuple(next_pos) not in visited:
                valid_neighbors.append(next_pos)

        return valid_neighbors

    def create_acyclic_graph(self, current_node=None):
        """
        Create an Acyclic graph

        Returns:
            G: A networkx acyclic graph with (n_nodes-1) edges
        """
        G = nx.Graph()
        visited = set()
        tracker = []

        # Pick a random node from the grid
        if not current_node:
            current_node = (np.random.randint(self.m), np.random.randint(self.n))

        tracker.append(current_node)
        visited.add(tuple(current_node))
        G.add_node(tuple(current_node))

        while len(visited) < self.n_nodes:
            current_node = tracker[-1]
            neighbors = self.get_valid_neighbors(current_node, visited, self.m, self.n)
            if neighbors:
                # Traverse in random (valid) directions until no possible neighbor nodes from current node are found
                random_index = np.random.choice(len(neighbors))
                next_node = neighbors[random_index]
                visited.add(tuple(next_node))
                G.add_node(tuple(next_node))
                G.add_edge(tuple(current_node), tuple(next_node))
                tracker.append(next_node)
            else:
                # Then track back to the last node from which a move can be made to a node that has not been visited
                tracker.pop()

        return G

    def create_cyclic_graph(self, n_loops: int = 1):
        """
        Create a cyclic graph with a specified number of loops.

        Args:
            n_loops: Number of loops (cycles) required in the graph.

        Returns:
            loop_graph: A networkx graph containing n_loops cycles.

        Raises:
            ValueError: If parameters are invalid or a valid configuration cannot be found.
        """
        if n_loops < 1:
            raise ValueError(f"Please set n_loops to at least 1 to form a cycle. Passed value: {n_loops}")

        if self.m == 1 or self.n == 1:
            raise ValueError(
                "1-D graph is being passed. Please use a 2-D grid with m, n >= 2 to create a cyclic graph."
            )

        if self.n_nodes < n_loops + 3:
            raise ValueError(
                f"At least {n_loops + 3} nodes are required to form {n_loops} loops."
                f" nodes provided: {self.n_nodes}. Increase n_nodes or reduce n_loops."
            )

        max_attempts = 10
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            possible_starts = set()

            # Pick a unique random starting node
            current_node = [np.random.randint(self.m), np.random.randint(self.n)]
            while tuple(current_node) in possible_starts:
                current_node = [np.random.randint(self.m), np.random.randint(self.n)]
            possible_starts.add(tuple(current_node))

            # Create acyclic graph from the starting node
            G = self.create_acyclic_graph(current_node)

            # Collect all possible additional edges that could be added to form loops
            nodes = G.nodes()
            existing_edges = set(G.edges())
            possible_edges = []

            for node in nodes:
                neighbors = self.get_valid_neighbors(node, [], self.m, self.n)
                for neighbor in neighbors:
                    if tuple(neighbor) in nodes:
                        edge = tuple(sorted((tuple(node), tuple(neighbor))))
                        if edge not in existing_edges and edge not in possible_edges:
                            possible_edges.append(edge)

            # Try to add edges to form desired number of loops
            np.random.shuffle(possible_edges)
            loop_graph = G.copy()
            cycles = nx.cycle_basis(loop_graph)

            while len(cycles) < n_loops and possible_edges:
                edge = possible_edges.pop()
                loop_graph.add_edge(*edge)
                cycles = nx.cycle_basis(loop_graph)

            if len(cycles) >= n_loops:
                return loop_graph

            print(f"[Attempt {attempt}] Could only form {len(cycles)} cycles. Retrying...")

        # If after max attempts, no valid config is found...
        raise ValueError(
            f"Could not find a valid configuration to form {n_loops} loops after {max_attempts} attempts."
            f" Try reducing n_loops or increasing grid size (m x n)."
        )

    def plot_graph(self, G):
        nx.draw_networkx(G, pos={n: n for n in G.nodes()})
        plt.show()

    def __repr__(self) -> str:
        return f"<BaseMap({self.m}, {self.n}, {self.n_nodes})>"


if __name__ == '__main__':
    map = BaseMap(2, 8, 10)
    G = map.create_acyclic_graph()
    map.plot_graph(G)
