import streamlit as st
import pulp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.manifold import MDS

st.title("工場内物流最適化問題（配送経路問題）")

# 距離行列の定義
distance_matrix = np.array([
    [0, 10, 15, 20, 10, 25],
    [10, 0, 35, 25, 15, 20],
    [15, 35, 0, 30, 20, 25],
    [20, 25, 30, 0, 15, 10],
    [10, 15, 20, 15, 0, 10],
    [25, 20, 25, 10, 10, 0]
])

n = len(distance_matrix)

# 距離のディクショナリを作成
distances = {}
for i in range(1, n+1):
    for j in range(1, n+1):
        if i != j:
            distances[(i, j)] = distance_matrix[i-1][j-1]

# MDSを用いて2次元埋め込み（距離行列にできるだけ沿う配置）
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
pos_mds = mds.fit_transform(distance_matrix)

# ノード番号（1始まり）と2次元座標の辞書作成
pos_dict = {i + 1: pos_mds[i] for i in range(n)}

# 入力グラフの表示
st.subheader("入力ネットワーク")

# 完全グラフ（全てのエッジを持つグラフ）の作成
fig_input, ax_input = plt.subplots(figsize=(10, 8))
G_input = nx.Graph()
nodes = list(range(1, n + 1))
G_input.add_nodes_from(nodes)

# 各ノード間のエッジを追加（重みとして距離を設定）
for i in range(n):
    for j in range(i + 1, n):
        G_input.add_edge(i + 1, j + 1, weight=distance_matrix[i][j])

# グラフ描画
nx.draw(G_input, pos=pos_dict, with_labels=True, node_color='skyblue', 
        node_size=700, font_weight='bold')
edge_labels = nx.get_edge_attributes(G_input, 'weight')
nx.draw_networkx_edge_labels(G_input, pos=pos_dict, edge_labels=edge_labels)
ax_input.set_title("距離行列に基づく完全グラフ")
plt.axis('off')
plt.tight_layout()

st.pyplot(fig_input)

# PuLPで問題を解く
def solve_tsp():
    # 問題の定義
    prob = pulp.LpProblem("AGV_TSP", pulp.LpMinimize)
    
    # 決定変数
    x = {}
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
    
    u = {}
    for i in range(1, n+1):
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n, cat=pulp.LpInteger)
    
    # 目的関数
    prob += pulp.lpSum(distances[(i, j)] * x[(i, j)] for i in range(1, n+1) for j in range(1, n+1) if i != j)
    
    # 制約条件
    # 各ステーションには1回だけ入る
    for j in range(1, n+1):
        prob += pulp.lpSum(x[(i, j)] for i in range(1, n+1) if i != j) == 1
    
    # 各ステーションからは1回だけ出る
    for i in range(1, n+1):
        prob += pulp.lpSum(x[(i, j)] for j in range(1, n+1) if i != j) == 1
    
    # Miller-Tucker-Zemlin制約（部分巡回路の除去）
    for i in range(2, n+1):
        for j in range(2, n+1):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 結果の抽出
    route = []
    current = 1  # スタートは1
    route.append(current)
    
    for _ in range(n-1):
        for j in range(1, n+1):
            if j != current and pulp.value(x[(current, j)]) == 1:
                route.append(j)
                current = j
                break
    
    route.append(1)  # 最後に1に戻る
    
    total_distance = sum(distances[(route[i-1], route[i])] for i in range(1, len(route)))
    
    return route, total_distance

# 問題を解く
st.subheader("問題を解く")
route, total_distance = solve_tsp()

# 結果を表示
st.subheader("最適解")
st.write(f"最適巡回路: {' → '.join(map(str, route))}")
st.write(f"総移動距離: {total_distance}")

# グラフ描画
st.subheader("最適巡回路のグラフ表示")

fig, ax = plt.subplots(figsize=(10, 8))
G = nx.DiGraph()

# ノードを追加
for i in range(1, n+1):
    G.add_node(i)

# エッジを追加
for i in range(len(route)-1):
    G.add_edge(route[i], route[i+1])

# ノードを描画
nx.draw_networkx_nodes(G, pos_dict, node_size=700, node_color='lightblue')

# エッジを描画
edge_labels = {(route[i], route[i+1]): f"{i+1}" for i in range(len(route)-1)}
nx.draw_networkx_edges(G, pos_dict, width=1.5, alpha=0.7, 
                      edge_color='blue', connectionstyle='arc3,rad=0.1',
                      arrowsize=15)

# ラベルを描画
nx.draw_networkx_labels(G, pos_dict)

# エッジラベルを描画
nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=edge_labels, 
                            font_color='red', font_weight='bold')

plt.axis('off')
plt.tight_layout()

st.pyplot(fig)
