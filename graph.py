# worked by Fioralba Frasheri

import heapq


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        # add a node to the graph
        self.nodes[node] = []

    def add_edge(self, from_node, to_node, weight):
        # check for self-loop
        if from_node == to_node:
            return

        # add an edge from 'from_node' to 'to_node' with the given 'weight'
        self.nodes[from_node].append((to_node, weight))

    def dijkstra(self, start_node):
        # Dijkstra's algorithm for finding shortest paths
        # initialize distances to infinity
        dist = {node: float('inf') for node in self.nodes}
        prev = {v: None for v in self.nodes}  # initialize previous nodes

        dist[start_node] = 0  # set the distance from start_node to itself as 0

        H = [(0, start_node)]  # create a priority queue

        while H:
            current_distance, current_node = heapq.heappop(H)

            # skip nodes that have already been visited
            if current_distance > dist[current_node]:
                continue

            for v, weight in self.nodes[current_node]:
                # calculate the potential new distance
                distance = dist[current_node] + weight

                # update the distance and previous node if a shorter path is found
                if distance < dist[v]:
                    dist[v] = distance
                    prev[v] = current_node
                    heapq.heappush(H, (dist[v], v))

        return dist, prev


def get_shortest_path(prev_nodes, node):
    # retrieve the shortest path from the previous nodes
    path = []
    while node is not None:
        path.insert(0, node)
        node = prev_nodes[node]
    return path


def topological_sort(graph):
    # compute the in-degree of each vertex, including self-loops
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v, _ in graph[u]:
            in_degree[v] += 1
        # handle self-loop
        if u in graph[u]:
            in_degree[u] += 1

    # initialize queue with nodes that have no incoming edges
    queue = [u for u in graph if in_degree[u] == 0]

    # process nodes in topological order
    sorted_vertices = []
    while queue:
        u = queue.pop(0)
        sorted_vertices.append(u)

        for v, _ in graph[u]:
            # decrement in-degree of v
            in_degree[v] -= 1
            # if in-degree of v becomes zero, add v to queue
            if in_degree[v] == 0:
                queue.append(v)
        # handle self-loop
        if u in graph[u]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                queue.append(u)

    return sorted_vertices


def longest_path_dag(graph, start):
    # negate the weights of all edges
    negated_graph = {}
    for u in graph:
        negated_graph[u] = {}
        for v, weight in graph[u]:
            negated_graph[u][v] = -weight

    # initialize distances from source to all vertices as negative infinity
    dist = {v: float('-inf') for v in graph}
    # set distance to source as 0
    dist[start] = 0

    # topological sort of the input graph
    sorted_vertices = topological_sort(graph)

    # process vertices in topological order
    for u in sorted_vertices:
        # for each predecessor v of u
        for v in negated_graph[u]:
            # if the distance to v plus the weight of the edge between v and u is greater than the distance to u,
            # update the distance to u
            if dist[v] + negated_graph[u][v] > dist[u]:
                dist[u] = dist[v] + negated_graph[u][v]

    # negate the final distances to obtain the longest path lengths
    for v in dist:
        dist[v] = -dist[v]

    return dist


# example usage
g = Graph()

# assuming the graph is provided in the specified format
with open('graph_input.txt', 'r') as file:
    n, e = map(int, file.readline().split())
    nodes = file.readline().split()
    start_node = int(file.readline())

    for _ in range(e):
        from_node, to_node, weight = map(int, file.readline().split())
        g.add_node(from_node)
        g.add_node(to_node)
        g.add_edge(from_node, to_node, weight)

distances, prev_nodes = g.dijkstra(start_node)

print(distances)

for node in g.nodes:
    shortest_path = get_shortest_path(prev_nodes, node)
    print(
        f"shortest path from node {start_node} to node {node}: {shortest_path}")

# calculate the longest path using the longest_path_dag function
# longest_paths = longest_path_dag(g.nodes, start_node)

# for node in g.nodes:
#     longest_path = longest_paths[node]
#     print(f"longest path from node {start_node} to node {node}: {longest_path}")
