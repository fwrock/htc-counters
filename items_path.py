import json
import math
import heapq
import os
import glob
import argparse
# collections.deque não é mais usado diretamente aqui, mas pode ser útil em outras partes de uma lib de grafo
# from collections import deque
from collections import defaultdict
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generic, List, Optional, Set, Tuple,
                    TypeVar)

# --- Type Variables ---
V = TypeVar('V')
ID = TypeVar('ID')
W = TypeVar('W')
L = TypeVar('L')

# --- Data Classes (Estruturas para os dados do grafo) ---
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
        if not isinstance(other, NodeGraph): return NotImplemented
        return self.id == other.id
    def euclidean_distance(self, other: 'NodeGraph') -> float:
        return math.sqrt((self.latitude - other.latitude)**2 + (self.longitude - other.longitude)**2)

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
        if not isinstance(other, EdgeGraph): return NotImplemented
        return self.id == other.id

@dataclass
class LoadedGraphData(Generic[V, ID, W, L]):
    graph: 'Graph[V, W, L]'
    nodes_by_id: Dict[ID, V]
    edge_labels_by_id: Dict[ID, L]


class Graph(Generic[V, W, L]):
    def __init__(self):
        self._adjacency_list: Dict[V, Dict[V, EdgeInfo[W, L]]] = {}

    @property
    def vertices(self) -> Set[V]:
        return set(self._adjacency_list.keys())

    def add_vertex(self, vertex: V) -> None:
        if vertex not in self._adjacency_list:
            self._adjacency_list[vertex] = {}

    def add_edge(self, source: V, target: V, weight: W, label: L) -> None:
        self.add_vertex(source)
        self.add_vertex(target)
        self._adjacency_list[source][target] = EdgeInfo(weight, label)

    def add_undirected_edge(self, v1: V, v2: V, weight: W, label: L) -> None:
        self.add_edge(v1, v2, weight, label)
        self.add_edge(v2, v1, weight, label)

    def neighbors(self, vertex: V) -> Dict[V, EdgeInfo[W, L]]:
        return self._adjacency_list.get(vertex, {})

    def edge_info(self, source: V, target: V) -> Optional[EdgeInfo[W, L]]:
        return self._adjacency_list.get(source, {}).get(target)

    def contains(self, vertex: V) -> bool:
        return vertex in self._adjacency_list
    
    def _reconstruct_edge_target_tuple_path(
        self, came_from: Dict[V, V], start_node: V, end_node: V
    ) -> Optional[List[Tuple[Edge[V, W, L], V]]]:
        if start_node == end_node: return []
        path: List[Tuple[Edge[V, W, L], V]] = []
        curr = end_node
        while curr in came_from:
            prev = came_from[curr]
            info = self.edge_info(prev, curr)
            if info is None: return None 
            path.append((Edge(prev, curr, info.weight, info.label), curr))
            if prev == start_node: return list(reversed(path))
            curr = prev
        return None

    def a_star_edge_targets_optimized(
        self, start_node: V, goal_node: V, heuristic: Callable[[V, V], float]
    ) -> Optional[Tuple[float, List[Tuple[Edge[V, W, L], V]]]]:
        if not self.contains(start_node) or not self.contains(goal_node): return None
        if start_node == goal_node: return (0.0, [])

        g_score: Dict[V, float] = defaultdict(lambda: float('inf'))
        f_score: Dict[V, float] = defaultdict(lambda: float('inf')) 
        came_from: Dict[V, V] = {}
        entry_count = 0
        open_set_heap: List[Tuple[float, int, V]] = [] 

        g_score[start_node] = 0.0
        f_score[start_node] = heuristic(start_node, goal_node)
        heapq.heappush(open_set_heap, (f_score[start_node], entry_count, start_node)); entry_count +=1

        while open_set_heap:
            current_f_in_heap, _, current_node = heapq.heappop(open_set_heap)
            if current_f_in_heap > f_score[current_node]: continue
            if current_node == goal_node:
                path_tuples = self._reconstruct_edge_target_tuple_path(came_from, start_node, goal_node)
                if path_tuples is not None: return (g_score[goal_node], path_tuples)
                return None 
            for neighbor_node, edge_data in self.neighbors(current_node).items():
                try: edge_weight = float(edge_data.weight)
                except (ValueError, TypeError): continue 
                tentative_g_score = g_score[current_node] + edge_weight
                if tentative_g_score < g_score[neighbor_node]:
                    came_from[neighbor_node] = current_node
                    g_score[neighbor_node] = tentative_g_score
                    f_score[neighbor_node] = tentative_g_score + heuristic(neighbor_node, goal_node)
                    heapq.heappush(open_set_heap, (f_score[neighbor_node], entry_count, neighbor_node)); entry_count += 1
        return None 

    @classmethod
    def empty(cls) -> 'Graph[V, W, L]':
        return cls()

    @classmethod
    def load_from_json(
        cls, json_stream_or_path: Any, node_id_extractor: Callable[[V], ID],
        edge_label_id_extractor: Callable[[L], ID], default_weight_for_unweighted: W,
        node_parser: Callable[[Dict[str, Any]], V], edge_label_parser: Callable[[Dict[str, Any]], L],
        verbose: bool = False, log_interval: int = 50000
    ) -> LoadedGraphData[V, ID, W, L]:
        
        if verbose: print("  [GraphLoad] Iniciando leitura do arquivo JSON do mapa...")
        raw_json_graph: Dict[str, Any]
        if isinstance(json_stream_or_path, str):
            with open(json_stream_or_path, 'r', encoding='utf-8') as f:
                raw_json_graph = json.load(f)
        else: 
            raw_json_graph = json.load(json_stream_or_path)
        if verbose: print(f"  [GraphLoad] Arquivo JSON do mapa lido ({len(raw_json_graph.get('nodes',[]))} nós declarados, {len(raw_json_graph.get('edges',[]))} arestas declaradas).")

        nodes_by_id: Dict[ID, V] = {}
        seen_node_ids: Set[ID] = set()
        raw_nodes_list = raw_json_graph.get('nodes', [])
        if verbose: print(f"  [GraphLoad] Iniciando parsing de {len(raw_nodes_list)} nós declarados...")
        for i, node_data_dict in enumerate(raw_nodes_list):
            try:
                node_obj = node_parser(node_data_dict)
                node_id = node_id_extractor(node_obj)
                if node_id in seen_node_ids:
                    print(f"  [GraphLoad] AVISO: ID de nó duplicado '{node_id}' na declaração de nós. Usando primeira ocorrência.")
                    continue 
                seen_node_ids.add(node_id)
                nodes_by_id[node_id] = node_obj
            except Exception as e:
                raise ValueError(f"Erro ao processar nó JSON na posição {i}: {node_data_dict}. Detalhe: {e}")
            if verbose and (i + 1) % log_interval == 0:
                print(f"    [GraphLoad] {i+1}/{len(raw_nodes_list)} nós declarados parseados...")
        if verbose: print(f"  [GraphLoad] Parsing de nós declarados concluído. {len(nodes_by_id)} nós únicos carregados no mapa ID->Nó.")

        graph = cls.empty()
        if verbose: print(f"  [GraphLoad] Adicionando {len(nodes_by_id)} vértices ao grafo...")
        # Adicionar vértices pode ser feito em lote ou um por um.
        # Se add_vertex for muito simples (como é agora), o loop é ok.
        for i, node_obj in enumerate(nodes_by_id.values()):
            graph.add_vertex(node_obj)
            if verbose and (i + 1) % log_interval == 0 and len(nodes_by_id) > log_interval : # Evitar log se for pequeno
                 print(f"    [GraphLoad] {i+1}/{len(nodes_by_id)} vértices adicionados...")
        if verbose: print(f"  [GraphLoad] Vértices adicionados ao grafo: {len(graph.vertices)}.")

        edge_labels_by_id: Dict[ID, L] = {}
        seen_edge_label_ids: Set[ID] = set()
        is_directed = raw_json_graph.get('directed', False)
        raw_edges_list = raw_json_graph.get('edges', [])
        if verbose: print(f"  [GraphLoad] Iniciando adição de {len(raw_edges_list)} arestas ao grafo (Direcionado: {is_directed})...")
        
        actual_edges_added_count = 0
        for i, edge_data_dict in enumerate(raw_edges_list):
            source_id, target_id = None, None # Inicializa para o bloco finally
            try:
                source_id = edge_data_dict['source_id']
                target_id = edge_data_dict['target_id']
                source_node = nodes_by_id.get(source_id)
                target_node = nodes_by_id.get(target_id)

                if source_node is None:
                    if verbose: print(f"  [GraphLoad] AVISO: Nó de origem '{source_id}' (aresta {i}) não encontrado. Pulando aresta.")
                    continue
                if target_node is None:
                    if verbose: print(f"  [GraphLoad] AVISO: Nó de destino '{target_id}' (aresta {i}) não encontrado. Pulando aresta.")
                    continue

                weight = edge_data_dict.get('weight', default_weight_for_unweighted)
                edge_label_obj = edge_label_parser(edge_data_dict['label'])
                label_id = edge_label_id_extractor(edge_label_obj)
                if label_id not in seen_edge_label_ids:
                    edge_labels_by_id[label_id] = edge_label_obj
                    seen_edge_label_ids.add(label_id)

                if is_directed:
                    graph.add_edge(source_node, target_node, weight, edge_label_obj)
                else:
                    graph.add_undirected_edge(source_node, target_node, weight, edge_label_obj)
                actual_edges_added_count +=1

            except KeyError as e:
                if verbose: print(f"  [GraphLoad] AVISO: Aresta {i} com dados incompletos (chave: {e}). Pulando. Dados: {edge_data_dict}")
                continue
            except Exception as e:
                if verbose: print(f"  [GraphLoad] AVISO: Erro ao processar aresta {i} ({source_id}->{target_id}). Pulando. Detalhe: {e}")
                continue
            finally:
                if verbose and (i + 1) % log_interval == 0:
                    print(f"    [GraphLoad] {i+1}/{len(raw_edges_list)} arestas declaradas processadas...")
        
        if verbose: print(f"  [GraphLoad] Adição de arestas concluída. {actual_edges_added_count} arestas efetivamente adicionadas.")
        return LoadedGraphData(graph, nodes_by_id, edge_labels_by_id)

# --- Funções Auxiliares (mantidas como antes) ---
def node_id_extractor_func(node: NodeGraph) -> str: return node.id
def edge_label_id_extractor_func(label: EdgeGraph) -> str: return label.id

def node_parser_func(data: Dict[str, Any]) -> NodeGraph:
    try:
        return NodeGraph(id=data['id'], resource_id=data['resourceId'], class_type=data['classType'],
                         latitude=float(data['latitude']), longitude=float(data['longitude']))
    except KeyError as e: raise ValueError(f"Chave ausente {e} nos dados do nó: {data}")
    except (TypeError, ValueError) as e: raise ValueError(f"Erro de tipo/valor {data}: {e}")

def edge_label_parser_func(data: Dict[str, Any]) -> EdgeGraph:
    try:
        return EdgeGraph(id=data['id'], resource_id=data['resourceId'], class_type=data['classType'],
                         length=float(data['length']))
    except KeyError as e: raise ValueError(f"Chave ausente {e} nos dados do label: {data}")
    except (TypeError, ValueError) as e: raise ValueError(f"Erro de tipo/valor {data}: {e}")

def parse_car_trips_from_file(file_path: str, verbose: bool = False) -> List[Tuple[str, str]]:
    trips: List[Tuple[str, str]] = []
    if verbose: print(f"  [CarParse] Lendo arquivo de viagens: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            car_data_list = json.load(f)
            for entry_idx, car_entry in enumerate(car_data_list):
                try:
                    origin_id = car_entry['data']['content']['origin']
                    destination_id = car_entry['data']['content']['destination']
                    trips.append((origin_id, destination_id))
                except KeyError as e:
                    if verbose: print(f"    [CarParse] AVISO: Pulando entrada {entry_idx} em {os.path.basename(file_path)} (chave: {e}). ID: {car_entry.get('id', 'N/A')}")
    except json.JSONDecodeError:
        print(f"  [CarParse] AVISO: Não foi possível decodificar JSON de {file_path}. Pulando arquivo.")
    except Exception as e:
        print(f"  [CarParse] AVISO: Erro inesperado ao processar {file_path}: {e}. Pulando arquivo.")
    if verbose: print(f"  [CarParse] {len(trips)} viagens encontradas em {os.path.basename(file_path)}.")
    return trips

def heuristic_func(node_a: NodeGraph, node_b: NodeGraph) -> float:
    return node_a.euclidean_distance(node_b)

# --- Script Principal ---
def main():
    parser = argparse.ArgumentParser(description="Calcula a soma das contagens de arestas dos caminhos A* para viagens de carro.")
    parser.add_argument("--cars-dir", required=True, help="Diretório contendo arquivos cars_*.json.")
    parser.add_argument("--map-file", required=True, help="Caminho para o arquivo JSON do mapa do grafo.")
    parser.add_argument("--default-weight", type=float, default=1.0, help="Peso padrão para arestas do grafo.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Imprime logs detalhados durante o processamento.")
    parser.add_argument("--log-interval", type=int, default=50000, help="Intervalo para logs de progresso durante carregamento de nós/arestas.")

    args = parser.parse_args()

    print(f"Iniciando script com modo verbose: {args.verbose}, intervalo de log: {args.log_interval}")
    print(f"Carregando mapa do grafo: {args.map_file}")
    try:
        loaded_map_data = Graph.load_from_json(
            args.map_file, node_id_extractor_func, edge_label_id_extractor_func,
            args.default_weight, node_parser_func, edge_label_parser_func,
            verbose=args.verbose, log_interval=args.log_interval
        )
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar o mapa do grafo de {args.map_file}: {e}")
        import traceback; traceback.print_exc()
        return

    map_graph = loaded_map_data.graph
    nodes_map = loaded_map_data.nodes_by_id
    # Contagem de arestas mais precisa para grafos não direcionados (onde cada aresta é adicionada duas vezes na representação)
    # No entanto, sum(len(adj) for adj in map_graph._adjacency_list.values()) conta as arestas direcionadas internas.
    # Se o grafo for não direcionado e as arestas adicionadas com add_undirected_edge, este valor será 2*|E|.
    # Para uma contagem de arestas "lógicas", precisaria de mais info ou dividir por 2 se não direcionado.
    # Por ora, a contagem de "slots" de arestas na lista de adjacência.
    num_internal_edges = sum(len(adj) for adj in map_graph._adjacency_list.values())
    print(f"Mapa do grafo carregado. Nós: {len(nodes_map)}, Entradas na lista de adjacência (arestas internas): {num_internal_edges}")


    total_sum_of_edge_counts = 0
    total_sum_of_costs_reference = 0.0
    paths_found_count = 0
    paths_not_found_count = 0
    global_trips_processed_count = 0 # Contador global de viagens

    car_file_pattern = os.path.join(args.cars_dir, "cars_*.json")
    car_files = glob.glob(car_file_pattern)

    if not car_files:
        print(f"Nenhum arquivo de carro encontrado com o padrão: {car_file_pattern}")
        return

    print(f"Encontrados {len(car_files)} arquivos de carro. Iniciando processamento de viagens...")

    for file_idx, car_file_path in enumerate(car_files):
        trips_in_current_file = parse_car_trips_from_file(car_file_path, verbose=args.verbose)
        
        if not trips_in_current_file and args.verbose:
            print(f"  Nenhuma viagem válida encontrada em {os.path.basename(car_file_path)}.")
            # Continua para o próximo arquivo mesmo se este estiver vazio ou com erros
        
        for trip_idx, (origin_id, destination_id) in enumerate(trips_in_current_file):
            global_trips_processed_count += 1
            if args.verbose:
                print(f"  [A*Trip] Arquivo {file_idx+1}/{len(car_files)}, Viagem {trip_idx+1}/{len(trips_in_current_file)} (Global: {global_trips_processed_count}): {origin_id} -> {destination_id}")

            origin_node = nodes_map.get(origin_id)
            destination_node = nodes_map.get(destination_id)

            if origin_node is None:
                if args.verbose: print(f"    [A*Trip] AVISO: Nó de origem '{origin_id}' não encontrado no mapa.")
                paths_not_found_count += 1
                continue
            if destination_node is None:
                if args.verbose: print(f"    [A*Trip] AVISO: Nó de destino '{destination_id}' não encontrado no mapa.")
                paths_not_found_count += 1
                continue
            
            astar_result = map_graph.a_star_edge_targets_optimized(
                origin_node, destination_node, heuristic_func
            )

            if astar_result:
                cost, path_tuples = astar_result
                number_of_edges_in_path = len(path_tuples)
                total_sum_of_edge_counts += number_of_edges_in_path
                total_sum_of_costs_reference += cost
                paths_found_count += 1
                if args.verbose:
                    print(f"    [A*Trip] Caminho encontrado. Nº de Arestas: {number_of_edges_in_path}, Custo: {cost:.2f}")
            else:
                if args.verbose:
                    print(f"    [A*Trip] AVISO: Caminho A* não encontrado para {origin_id} -> {destination_id}.")
                paths_not_found_count += 1
        
        # Log de progresso após cada arquivo de carro
        print(f"Processado arquivo {os.path.basename(car_file_path)} ({len(trips_in_current_file)} viagens). "
              f"Total de viagens processadas até agora: {global_trips_processed_count}. "
              f"Total de arestas acumuladas: {total_sum_of_edge_counts}")
    
    print("\n--- Resumo Final ---")
    print(f"Total de arquivos de carro processados: {len(car_files)}")
    print(f"Total de viagens individuais processadas: {global_trips_processed_count}")
    print(f"Caminhos encontrados com sucesso: {paths_found_count}")
    print(f"Caminhos não encontrados (ou nós ausentes): {paths_not_found_count}")
    print(f"SOMATÓRIO DA QUANTIDADE DE ARESTAS EM TODOS OS CAMINHOS ENCONTRADOS: {total_sum_of_edge_counts}")
    if args.verbose or total_sum_of_costs_reference > 0 : # Mostrar se verbose ou se houver custos
        print(f"Somatório dos custos A* de todos os caminhos (referência): {total_sum_of_costs_reference:.2f}")

if __name__ == "__main__":
    main()