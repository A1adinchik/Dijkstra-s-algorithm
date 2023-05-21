import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def dijkstra(graph, start, end):
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph[current_node].keys()
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph[current_node][next_node] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

def find_fastest_route(start, end, file_name):
    graph = defaultdict(dict)
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name)
    elif file_name.endswith('.xlsx'):
        df = pd.read_excel(file_name)
    else:
        raise ValueError('Invalid file format. Use .csv or .xlsx')

    for index, row in df.iterrows():
        station1, station2, time = row['Station1'], row['Station2'], row['Time']
        graph[station1][station2] = time
        graph[station2][station1] = time

    shortest_path = dijkstra(graph, start, end)
    if isinstance(shortest_path, str):
        return shortest_path
    path_length = sum(graph[shortest_path[i]][shortest_path[i + 1]] for i in range(len(shortest_path) - 1))
    return shortest_path, path_length, graph

def draw_graph(graph, shortest_path):
    G = nx.Graph()
    for node in graph:
        for neighbor, time in graph[node].items():
            G.add_edge(node, neighbor, time=time)

    pos = nx.spring_layout(G)  # type: ignore # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700) # type: ignore

    # edges
    nx.draw_networkx_edges(G, pos) # type: ignore

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif') # type: ignore

    # shortest path in red
    path_edges = list(zip(shortest_path, shortest_path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='r', node_size=700) # type: ignore
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=3) # type: ignore

    plt.show()

start = "Station1"
end = "Station5"
file_name = "test.xlsx"
shortest_path, path_length, graph = find_fastest_route(start, end, file_name)
print(f"The shortest path from {start} to {end} is: {' -> '.join(shortest_path)} with length {path_length}")

draw_graph(graph, shortest_path)