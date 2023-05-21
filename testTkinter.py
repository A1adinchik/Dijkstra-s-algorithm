import tkinter as tk
from tkinter import filedialog
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class DijkstraProgram:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SuperGPS")

        self.title_label = tk.Label(self.root, text="SuperGPS")
        self.title_label.pack()

        self.start_label = tk.Label(self.root, text="Start Point:")
        self.start_label.pack()
        self.start_entry = tk.Entry(self.root)
        self.start_entry.pack()

        self.end_label = tk.Label(self.root, text="End Point:")
        self.end_label.pack()
        self.end_entry = tk.Entry(self.root)
        self.end_entry.pack()

        self.file_label = tk.Label(self.root, text="File:")
        self.file_label.pack()
        self.file_entry = tk.Entry(self.root)
        self.file_entry.pack()

        self.file_button = tk.Button(self.root, text="Select File", command=self.select_file)
        self.file_button.pack()

        self.run_button = tk.Button(self.root, text="Find Fastest Route", command=self.run_program)
        self.run_button.pack()

    def dijkstra(self, graph, start, end):
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

    def build_graph_from_file(self, file_name):
        # Here you should decide whether the file is csv or excel based on the file extension or content
        # Here's a simple implementation based on extension
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_name)
        else:
            print('Unsupported file format')
            return None
        
        graph = {}
        for i, row in df.iterrows():
            station1, station2, time = row
            if station1 in graph:
                graph[station1][station2] = time
            else:
                graph[station1] = {station2: time}
            
            if station2 in graph:
                graph[station2][station1] = time
            else:
                graph[station2] = {station1: time}
        return graph

    def find_fastest_route(self, start_point, end_point, file_name):
        graph = self.build_graph_from_file(file_name)
        shortest_path = self.dijkstra(graph, start_point, end_point)
        return shortest_path, len(shortest_path)

    def draw_graph(self, graph, shortest_path):
        G = nx.Graph()

        # Add edges to the graph with weights
        for node in graph:
            for edge, weight in graph[node].items():
                G.add_edge(node, edge, weight=weight)

        # Generate positions for the nodes
        pos = nx.spring_layout(G) # type: ignore
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos) # type: ignore

        # Draw node labels
        nx.draw_networkx_labels(G, pos) # type: ignore

        # Draw edges
        nx.draw_networkx_edges(G, pos) # type: ignore

        # Draw edge labels with weights
        edge_labels = nx.get_edge_attributes(G, 'weight') # type: ignore
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels) # type: ignore

        # Highlight the shortest path
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2) # type: ignore

        # Show the plot
        plt.show()

    def select_file(self):
        filename = filedialog.askopenfilename()
        self.file_entry.delete(0, 'end') # clear the entry field
        self.file_entry.insert(0, filename) # insert the selected file name

    def run_program(self):
        start = self.start_entry.get()
        end = self.end_entry.get()
        file = self.file_entry.get()
        shortest_path, path_length = self.find_fastest_route(start, end, file)
        print(f"The shortest path from {start} to {end} is: {' -> '.join(shortest_path)} with length {path_length}")
        self.draw_graph(self.build_graph_from_file(file), shortest_path)

    def start(self):
        self.root.mainloop()

program = DijkstraProgram()
program.start()