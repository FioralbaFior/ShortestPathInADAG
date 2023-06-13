import networkx as nx
from networkx import draw, spring_layout, get_edge_attributes, draw_networkx_edges, draw_networkx_edge_labels
import matplotlib.pyplot as plt
import heapq


class DirectedGraph:
    def __init__(self, numOfVertices):
        self.numOfVertices = numOfVertices
        self.graph = nx.DiGraph()

    def get_vertices(self):
        return list(self.graph.nodes())

    def add_edge(self, u, v, weight):
        self.graph.add_edge(u, v, weight=weight)

    def visualize_graph(self, shortest_path_dijkstra=None, shortest_path_bellman_ford=None):
        pos = nx.circular_layout(self.graph)  # Use circular layout
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw(self.graph, pos, with_labels=True,
                node_color='lightblue', node_size=500)

        if shortest_path_dijkstra:
            edges = [(shortest_path_dijkstra[i], shortest_path_dijkstra[i+1]) for i in range(len(shortest_path_dijkstra)-1)
                     if self.graph.has_edge(shortest_path_dijkstra[i], shortest_path_dijkstra[i+1])]
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=edges, edge_color='red', width=0.5)

        if shortest_path_bellman_ford:
            edges = [(shortest_path_bellman_ford[i], shortest_path_bellman_ford[i+1]) for i in range(len(shortest_path_bellman_ford)-1)
                     if self.graph.has_edge(shortest_path_bellman_ford[i], shortest_path_bellman_ford[i+1])]
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=edges, edge_color='yellow', width=0.5)

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

        plt.show()

    def is_acyclic(self):
        is_acyclic = nx.is_directed_acyclic_graph(graph.graph)
        return is_acyclic

    def has_negative_edges(self):
        for u, v, edge_data in self.graph.edges(data=True):
            if 'weight' in edge_data and edge_data['weight'] < 0:
                return True
        return False

    def dijkstra(self, start_vertex):
        dist = {v: float('inf') for v in self.graph}
        prev = {v: None for v in self.graph}
        dist[start_vertex] = 0
        visited = [start_vertex]

        H = [(dist[start_vertex], start_vertex)]

        heapq.heapify(H)

        while H:
            _, u = heapq.heappop(H)
            visited.append(u)

            for v, weight in self.graph[u].items():
                if dist[v] > dist[u] + weight['weight']:
                    dist[v] = dist[u] + weight['weight']
                    prev[v] = u
                    heapq.heappush(H, (dist[v], v))
                    visited.append(u)

        return dist, prev, visited

    def bellman_ford(self, start_vertex):
        dist = {v: float('inf') for v in self.graph}
        prev = {v: None for v in self.graph}
        dist[start_vertex] = 0

        for _ in range(self.numOfVertices - 1):
            for u, v, data in self.graph.edges(data=True, default=1):
                weight = data['weight']
                if dist[u] != float("Inf") and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    prev[v] = u

        for u, v, data in self.graph.edges(data=True, default=1):
            weight = data['weight']
            if dist[u] != float("Inf") and dist[u] + weight < dist[v]:
                print("Graph contains negative weight cycle")

        shortest_path = []

        for dist in self.graph:
            shortest_path.append(dist)

        return dist, prev, shortest_path


# Example usage
graph = DirectedGraph(5)

start_vertex = 0

# Add edges to the graph


graph.add_edge(0, 1, 4)
graph.add_edge(0, 2, 2)
graph.add_edge(1, 2, 3)
graph.add_edge(2, 1, 1)
graph.add_edge(1, 3, 2)
graph.add_edge(1, 4, 3)
graph.add_edge(2, 3, 4)
graph.add_edge(2, 4, 5)
graph.add_edge(3, 4, 1)

# Calculate the shortest paths
distance, previous, shortest_path_dijkstra = graph.dijkstra(start_vertex)
distance1, previous1, shortest_path_bellman_ford = graph.bellman_ford(
    start_vertex)


if graph.has_negative_edges():
    graph.visualize_graph(
        shortest_path_bellman_ford=shortest_path_bellman_ford, shortest_path_dijkstra=None)
    print("distances: ", distance, "previous: ", previous)
else:
    graph.visualize_graph(shortest_path_bellman_ford=None,
                          shortest_path_dijkstra=shortest_path_dijkstra)
    print("distances: ", distance1, "previous: ", previous1)
