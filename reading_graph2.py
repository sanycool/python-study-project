import networkx as nx
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import random
def create_graph_from_csv(csv_file="friends_network_detailed.csv"):
    """
    Создает направленный граф из CSV файла
    """
    # Создаем направленный граф
    G = nx.DiGraph()
    
    # Словарь для хранения информации о пользователях
    users_info = {}
    
    # Читаем CSV файл
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            cur_id = row['ID']
            cur_name = row['Name']
            handshake_id = row['handshake_ID']
            handshake_name = row['handshake_name']
            connection_type = row['Connection Type']
            
            # Добавляем информацию о основном друге
            users_info[cur_id] = cur_name
            G.add_node(cur_id, name=cur_name, label=cur_name)
            G.add_edge(handshake_id, cur_id)
    
    return G, users_info

def random_subgraph(G, N=10000, seed=69, neccesary_nodes=['273517530', '232453448', '184300739']):
    all_nodes = list(G.nodes())
    
    if len(all_nodes) <= N:
        return G

    random.seed(seed)
    #Выбираем случайные N узлов, и берем наши, чтобы наверняка
    sampled_nodes = random.sample(all_nodes, N) + neccesary_nodes
    #G.subgraph() выберет ребра, где оба конца из sampled_nodes
    sub = G.subgraph(sampled_nodes)
    return sub
        
    
if __name__ == "__main__":
    WE = [(273517530, 'EGOR K'), (232453448, 'ALEX S'), (184300739, "EGOR S")]
    G, users_info = create_graph_from_csv("friends_big_network_detailed.csv")
    # nx.Graph в принт выдаст кол-во узлов и ребер
    print(G)
        
    
    # Показываем топ-10 самых связанных пользователей
    degrees = dict(G.degree())
    top_10 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nТоп-10 самых связанных пользователей:")
    for node, degree in top_10:
        name = users_info.get(node, node)
        print(f"  {name}: {degree} связей")

    # центральности
    centralities = [
        (nx.degree_centrality, 'степени (degree)'),       
        (nx.closeness_centrality, 'близости (closeness)'),
        (nx.eigenvector_centrality, 'собств. вектору (eigenvector)'),
        (nx.betweenness_centrality, 'посредничеству (betweenness)'),
        #вычисление всех кратчайших путей для betweenness идет слишком долго, посчитаем отдельно
        
    ]
    for centr_func, centr_name in centralities:
        print('Считаем центральность по '+centr_name+'...')
        centrality = centr_func(G)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nСамые центральные узлы по {centr_name}:")
        for node, centr in top_central:
            name = users_info.get(node, node)
            print(f"  {name}: {centr:.4f}")

        print(f"\nНаша центральность по {centr_name}:")
        for i, name in WE:
            centr = centrality[str(i)]
            print(f"  {name}: {centr:.4f}")
            
    '''я хочу выбрать из 43000 узлов полного графа только 1000,
        сохранить ребра между ними как в полном графе,
        посчитать центральность "наших" узлов в этом подграфе'''
    # Считаем центральность по посредничеству отдельно
    # Ничего не вышло, время выполнения есть O(nodes * edges)
    # обрезание графа сильно портит метрику
    '''sub = random_subgraph(G)
    print(sub)
    centrality = nx.betweenness_centrality(sub)
    #neccesary_nodes = ['273517530', '232453448', '184300739']            
    #N = 1000
    #sampled_nodes = random.sample(list(all_nodes), N) + neccesary_nodes
    #centrality = nx.betweenness_centrality_subset(uG, sampled_nodes, sampled_nodes)
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:8]
    print("\nСамые центральные узлы по посредничеству:")
    for node, centr in top_central:
        name = users_info.get(node, node)
        print(f"  {name}: {centr:.4f}")

    print(f"\nНаша центральность по посредничеству:")
    for i, name in WE:
        centr = centrality[str(i)]
        print(f"  {name}: {centr:.4f}")'''
