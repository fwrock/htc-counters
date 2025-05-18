import json
import math
import heapq
import os
import glob
import argparse
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Generic, List, Optional, Set, Tuple,
                    TypeVar)

# --- Type Variables (Reutilizadas da conversão anterior) ---
V = TypeVar('V')
ID = TypeVar('ID')
W = TypeVar('W')
L = TypeVar('L')

# --- Data Classes (Reutilizadas e adaptadas da conversão anterior) ---

@dataclass(frozen=True)
class EdgeInfo(Generic[W, L]):
    weight: W
    label: L

@dataclass(frozen=True)
class Edge(Generic[V, W, L]):
    source: V
    target: V
    weight: W
    label: L

@dataclass
class NodeGraph:
    id: str
    resource_id: str
    class_type: str
    latitude: float
    longitude: float

    def __str__(self) -> str:
        return f"Node({self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NodeGraph):
            return NotImplemented
        return self.id == other.id

    def euclidean_distance(self, other: 'NodeGraph') -> float:
        lat_diff = self.latitude - other.latitude
        lon_diff = self.longitude - other.longitude
        return math.sqrt(lat_diff**2 + lon_diff**2)

@dataclass
class EdgeGraph:
    id: str
    resource_id: str
    class_type: str
    length: float

    def __str__(self) -> str:
        return f"EdgeLabel({self.id}, len={self.length:.2f})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EdgeGraph):
            return NotImplemented
        return self.id == other.id

@dataclass
class LoadedGraphData(Generic[V, ID, W, L]):
    graph: 'Graph[V, W, L]'
    nodes_by_id: Dict[ID, V]
    edge_labels_by_id: Dict[ID, L]

# --- Main Graph Class (Reutilizada da conversão anterior) ---
class Graph(Generic[V, W, L]):
    def __init__(self, adjacency_list: Optional[Dict[V, Dict[V, EdgeInfo[W, L]]]] = None):
        self._adjacency_list: Dict[V, Dict[V, EdgeInfo[W, L]]] = \
            adjacency_list if adjacency_list is not None else {}

    @property
    def vertices(self) -> Set[V]:
        return set(self._adjacency_list.keys())

    def add_vertex(self, vertex: V) -> 'Graph[V, W, L]':
        if vertex in self._adjacency_list:
            return self
        new_adj = self._adjacency_list.copy()
        new_adj[vertex] = {}
        return Graph(new_adj)

    def add_edge(self, source: V, target: V, weight: W, label: L) -> 'Graph[V, W, L]':
        current_graph_state = self
        if not current_graph_state.contains(source):
            current_graph_state = current_graph_state.add_vertex(source)
        if not current_graph_state.contains(target):
            current_graph_state = current_graph_state.add_vertex(target)
        
        new_adj = {k: v.copy() for k, v in current_graph_state._adjacency_list.items()}
        edge_info = EdgeInfo(weight, label)
        if source not in new_adj:
             new_adj[source] = {}
        new_adj[source][target] = edge_info
        return Graph(new_adj)

    def add_undirected_edge(self, v1: V, v2: V, weight: W, label: L) -> 'Graph[V, W, L]':
        return self.add_edge(v1, v2, weight, label).add_edge(v2, v1, weight, label)

    def neighbors(self, vertex: V) -> Dict[V, EdgeInfo[W, L]]:
        return self._adjacency_list.get(vertex, {})

    def edge_info(self, source: V, target: V) -> Optional[EdgeInfo[W, L]]:
        return self._adjacency_list.get(source, {}).get(target)

    def contains(self, vertex: V) -> bool:
        return vertex in self._adjacency_list

    def _reconstruct_edge_target_tuple_path(
        self,
        came_from: Dict[V, V],
        start_node: V,
        end_node: V
    ) -> Optional[List[Tuple[Edge[V, W, L], V]]]:
        if start_node == end_node:
            return []
        
        path: List[Tuple[Edge[V, W, L], V]] = []
        curr = end_node
        
        while curr in came_from:
            prev = came_from[curr]
            info = self.edge_info(prev, curr)
            if info is None:
                return None 
            
            edge = Edge(prev, curr, info.weight, info.label)
            path.append((edge, curr))
            
            if prev == start_node:
                return list(reversed(path))
            
            curr = prev
        return None

    def a_star_edge_targets_optimized(
        self,
        start_node: V,
        goal_node: V,
        heuristic: Callable[[V, V], float],
    ) -> Optional[Tuple[float, List[Tuple[Edge[V, W, L], V]]]]:
        if not self.contains(start_node) or not self.contains(goal_node):
            return None
        if start_node == goal_node:
            return (0.0, [])

        g_score: Dict[V, float] = defaultdict(lambda: float('inf'))
        f_score: Dict[V, float] = defaultdict(lambda: float('inf')) 
        came_from: Dict[V, V] = {}
        open_set: List[Tuple[float, int, V]] = [] 
        open_set_nodes: Set[V] = set()

        g_score[start_node] = 0.0
        f_score[start_node] = heuristic(start_node, goal_node)
        
        entry_count = 0 
        heapq.heappush(open_set, (f_score[start_node], entry_count, start_node))
        open_set_nodes.add(start_node)
        entry_count +=1

        while open_set:
            current_f_score_in_queue, _, current = heapq.heappop(open_set)
            open_set_nodes.remove(current)

            if current_f_score_in_queue > f_score[current]: 
                continue
            
            if current == goal_node:
                path_tuples = self._reconstruct_edge_target_tuple_path(came_from, start_node, goal_node)
                if path_tuples is not None:
                    return (g_score[goal_node], path_tuples)
                return None 

            for neighbor, edge_info_obj in self.neighbors(current).items():
                try:
                    weight_val = float(edge_info_obj.weight)
                except (ValueError, TypeError):
                    continue 

                tentative_g_score = g_score[current] + weight_val

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    new_f_score_for_neighbor = tentative_g_score + heuristic(neighbor, goal_node)
                    f_score[neighbor] = new_f_score_for_neighbor
                    
                    if neighbor not in open_set_nodes:
                        heapq.heappush(open_set, (new_f_score_for_neighbor, entry_count, neighbor))
                        open_set_nodes.add(neighbor)
                        entry_count += 1
        return None 

    @classmethod
    def empty(cls) -> 'Graph[V, W, L]':
        return cls()

    @classmethod
    def load_from_json(
        cls,
        json_stream_or_path: Any,
        node_id_extractor: Callable[[V], ID],
        edge_label_id_extractor: Callable[[L], ID],
        default_weight_for_unweighted: W,
        node_parser: Callable[[Dict[str, Any]], V],
        edge_label_parser: Callable[[Dict[str, Any]], L]
    ) -> LoadedGraphData[V, ID, W, L]:
        
        raw_json_graph: Dict[str, Any]
        if isinstance(json_stream_or_path, str):
            with open(json_stream_or_path, 'r', encoding='utf-8') as f:
                raw_json_graph = json.load(f)
        else: 
            raw_json_graph = json.load(json_stream_or_path)
        
        parsed_nodes: List[V] = []
        for i, n_data in enumerate(raw_json_graph.get('nodes', [])):
            try:
                parsed_nodes.append(node_parser(n_data))
            except Exception as e:
                raise ValueError(f"Error parsing node at index {i}: {n_data}. Error: {e}")

        nodes_by_id: Dict[ID, V] = {}
        seen_node_ids: Set[ID] = set()
        for node_obj in parsed_nodes:
            node_id = node_id_extractor(node_obj)
            if node_id in seen_node_ids:
                raise ValueError(f"Duplicate node ID in JSON: {node_id}")
            seen_node_ids.add(node_id)
            nodes_by_id[node_id] = node_obj

        graph: Graph[V, W, L] = Graph.empty()
        edge_labels_by_id: Dict[ID, L] = {}
        seen_edge_label_ids: Set[ID] = set()
        is_directed = raw_json_graph.get('directed', False)

        for i, raw_json_edge in enumerate(raw_json_graph.get('edges', [])):
            try:
                source_node_id: ID = raw_json_edge['source_id']
                target_node_id: ID = raw_json_edge['target_id']
            except KeyError as e:
                raise ValueError(f"Edge at index {i} is missing 'source_id' or 'target_id': {raw_json_edge}. Error: {e}")

            source_node = nodes_by_id.get(source_node_id)
            target_node = nodes_by_id.get(target_node_id)

            if source_node is None:
                raise LookupError(f"Source node with ID '{source_node_id}' (from edge at index {i}) not found.")
            if target_node is None:
                raise LookupError(f"Target node with ID '{target_node_id}' (from edge at index {i}) not found.")

            weight_value = raw_json_edge.get('weight')
            if weight_value is None:
                weight = default_weight_for_unweighted
            else:
                weight = weight_value

            try:
                edge_label_data = raw_json_edge['label']
                edge_label_obj: L = edge_label_parser(edge_label_data)
            except KeyError:
                 raise ValueError(f"Edge at index {i} is missing 'label': {raw_json_edge}")
            except Exception as e:
                raise ValueError(f"Error parsing edge label for edge at index {i}: {raw_json_edge.get('label')}. Error: {e}")

            current_edge_label_id = edge_label_id_extractor(edge_label_obj)
            if current_edge_label_id not in seen_edge_label_ids:
                edge_labels_by_id[current_edge_label_id] = edge_label_obj
                seen_edge_label_ids.add(current_edge_label_id)
            else:
                existing_label = edge_labels_by_id.get(current_edge_label_id)
                if existing_label != edge_label_obj :
                     print(f"WARNING: Duplicate edge label ID '{current_edge_label_id}' "
                           "with different objects. Using first found.")
            
            if is_directed:
                graph = graph.add_edge(source_node, target_node, weight, edge_label_obj)
            else:
                graph = graph.add_undirected_edge(source_node, target_node, weight, edge_label_obj)
                
        return LoadedGraphData(graph, nodes_by_id, edge_labels_by_id)

# --- Funções Auxiliares para o Script Principal ---

def node_id_extractor_func(node: NodeGraph) -> str:
    return node.id

def edge_label_id_extractor_func(label: EdgeGraph) -> str:
    return label.id

def node_parser_func(data: Dict[str, Any]) -> NodeGraph:
    try:
        return NodeGraph(
            id=data['id'],
            resource_id=data['resourceId'],
            class_type=data['classType'],
            latitude=float(data['latitude']),
            longitude=float(data['longitude'])
        )
    except KeyError as e:
        raise ValueError(f"Missing key {e} in node data: {data}")
    except ValueError as e:
        raise ValueError(f"Type conversion error for node data {data}: {e}")


def edge_label_parser_func(data: Dict[str, Any]) -> EdgeGraph:
    try:
        return EdgeGraph(
            id=data['id'],
            resource_id=data['resourceId'],
            class_type=data['classType'],
            length=float(data['length'])
        )
    except KeyError as e:
        raise ValueError(f"Missing key {e} in edge label data: {data}")
    except ValueError as e:
        raise ValueError(f"Type conversion error for edge label data {data}: {e}")

def parse_car_trips_from_file(file_path: str) -> List[Tuple[str, str]]:
    trips: List[Tuple[str, str]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            car_data_list = json.load(f)
            for car_entry in car_data_list:
                try:
                    origin_id = car_entry['data']['content']['origin']
                    destination_id = car_entry['data']['content']['destination']
                    trips.append((origin_id, destination_id))
                except KeyError as e:
                    print(f"Warning: Skipping entry in {file_path} due to missing key: {e}. Entry: {car_entry.get('id', 'Unknown ID')}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Skipping file.")
    except Exception as e:
        print(f"Warning: An unexpected error occurred while processing {file_path}: {e}. Skipping file.")
    return trips

def heuristic_func(node_a: NodeGraph, node_b: NodeGraph) -> float:
    return node_a.euclidean_distance(node_b)


# --- Script Principal ---
def main():
    parser = argparse.ArgumentParser(description="Calculates total A* path items for car trips.")
    parser.add_argument("--cars-dir", required=True, help="Directory containing car_*.json files.")
    parser.add_argument("--map-file", required=True, help="Path to the graph JSON map file.")
    parser.add_argument("--default-weight", type=float, default=1.0, help="Default weight for graph edges if not specified in JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print details for each trip.")

    args = parser.parse_args()

    print("Loading map graph...")
    try:
        loaded_map_data: LoadedGraphData[NodeGraph, str, float, EdgeGraph] = Graph.load_from_json(
            json_stream_or_path=args.map_file,
            node_id_extractor=node_id_extractor_func,
            edge_label_id_extractor=edge_label_id_extractor_func,
            default_weight_for_unweighted=args.default_weight,
            node_parser=node_parser_func,
            edge_label_parser=edge_label_parser_func
        )
    except Exception as e:
        print(f"Error loading map graph from {args.map_file}: {e}")
        return

    map_graph = loaded_map_data.graph
    nodes_map = loaded_map_data.nodes_by_id
    print(f"Map graph loaded. {len(nodes_map)} nodes, {len(map_graph.edges)} edges.")

    total_sum_of_edge_counts = 0 # MUDANÇA AQUI: para somar a quantidade de arestas
    total_sum_of_costs = 0.0 # Opcional: para ainda rastrear o custo total se desejado
    paths_found_count = 0
    paths_not_found_count = 0
    trips_processed_count = 0

    car_file_pattern = os.path.join(args.cars_dir, "cars_*.json")
    car_files = glob.glob(car_file_pattern)

    if not car_files:
        print(f"No car files found matching pattern: {car_file_pattern}")
        return

    print(f"Found {len(car_files)} car files. Processing trips...")

    for car_file_path in car_files:
        if args.verbose:
            print(f"Processing car file: {car_file_path}")
        trips = parse_car_trips_from_file(car_file_path)
        for origin_id, destination_id in trips:
            trips_processed_count += 1
            if args.verbose:
                print(f"  Calculating path for: {origin_id} -> {destination_id}")

            origin_node = nodes_map.get(origin_id)
            destination_node = nodes_map.get(destination_id)

            if origin_node is None:
                if args.verbose:
                    print(f"    Warning: Origin node ID '{origin_id}' not found in map. Skipping trip.")
                paths_not_found_count += 1
                continue
            if destination_node is None:
                if args.verbose:
                    print(f"    Warning: Destination node ID '{destination_id}' not found in map. Skipping trip.")
                paths_not_found_count += 1
                continue
            
            if origin_node == destination_node:
                if args.verbose:
                    print(f"    Origin and destination are the same ({origin_id}). Path length: 0, Cost: 0.")
                paths_found_count += 1
                # total_sum_of_edge_counts += 0 (já é 0)
                # total_sum_of_costs += 0.0 (já é 0)
                continue

            astar_result = map_graph.a_star_edge_targets_optimized(
                origin_node,
                destination_node,
                heuristic_func
            )

            if astar_result:
                cost, path_tuples = astar_result
                number_of_edges_in_path = len(path_tuples) # MUDANÇA AQUI

                total_sum_of_edge_counts += number_of_edges_in_path # MUDANÇA AQUI
                total_sum_of_costs += cost # Ainda somando o custo, para informação adicional

                paths_found_count += 1
                if args.verbose:
                    print(f"    Path found. Number of edges: {number_of_edges_in_path}, Cost: {cost:.2f}")
            else:
                if args.verbose:
                    print(f"    Warning: A* path not found for {origin_id} -> {destination_id}.")
                paths_not_found_count += 1
    
    print("\n--- Summary ---")
    print(f"Total trips processed: {trips_processed_count}")
    print(f"Paths successfully found: {paths_found_count}")
    print(f"Paths not found (or nodes missing): {paths_not_found_count}")
    print(f"Total sum of A* path edge counts: {total_sum_of_edge_counts}") # MUDANÇA AQUI
    print(f"Total sum of A* path costs (for reference): {total_sum_of_costs:.2f}")


if __name__ == "__main__":
    main()