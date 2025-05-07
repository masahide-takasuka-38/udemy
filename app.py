import streamlit as st
import pulp
import pandas as pd
import numpy as np
import altair as alt

# アプリのタイトル
st.title("在庫管理最適化問題")

# 固定パラメータ
h = 2.0      # 在庫保管コスト
p = 5.0      # 欠品コスト
K = 50       # 段取りコスト
initial_inventory = 10  # 初期在庫
M = 10000    # 十分大きな数

# 月と需要の定義
months = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
demands = [15, 10, 20, 10, 25, 20, 15, 15, 10, 20, 25, 15]

# 最適化問題を解く
def solve_inventory_problem():
    T = len(demands)  # 期間数
    
    # 問題の定義
    prob = pulp.LpProblem("InventoryManagement", pulp.LpMinimize)
    
    # 決定変数
    x = {t: pulp.LpVariable(f"x_{t}", lowBound=0) for t in range(T)}
    I_plus = {t: pulp.LpVariable(f"I_plus_{t}", lowBound=0) for t in range(T)}
    I_minus = {t: pulp.LpVariable(f"I_minus_{t}", lowBound=0) for t in range(T)}
    y = {t: pulp.LpVariable(f"y_{t}", cat=pulp.LpBinary) for t in range(T)}
    
    # 初期条件
    I_plus_prev = initial_inventory
    I_minus_prev = 0
    
    # 目的関数
    prob += pulp.lpSum(h * I_plus[t] + p * I_minus[t] + K * y[t] for t in range(T))
    
    # 制約条件
    for t in range(T):
        # 在庫バランス制約
        prob += I_plus[t] - I_minus[t] == I_plus_prev - I_minus_prev + x[t] - demands[t]
        I_plus_prev, I_minus_prev = I_plus[t], I_minus[t]
        
        # 段取り判定用の制約
        prob += x[t] <= M * y[t]
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 結果を抽出
    results = {
        "order_quantities": [pulp.value(x[t]) for t in range(T)],
        "positive_inventory": [pulp.value(I_plus[t]) for t in range(T)],
        "negative_inventory": [pulp.value(I_minus[t]) for t in range(T)],
        "setup_indicators": [pulp.value(y[t]) for t in range(T)],
        "objective_value": pulp.value(prob.objective)
    }
    
    # 派生した結果を計算
    results["net_inventory"] = [results["positive_inventory"][t] - results["negative_inventory"][t] for t in range(T)]
    results["holding_costs"] = [h * results["positive_inventory"][t] for t in range(T)]
    results["shortage_costs"] = [p * results["negative_inventory"][t] for t in range(T)]
    results["setup_costs"] = [K * results["setup_indicators"][t] for t in range(T)]
    results["total_costs"] = [results["holding_costs"][t] + results["shortage_costs"][t] + results["setup_costs"][t] for t in range(T)]
    
    return results

# 最適化を実行
results = solve_inventory_problem()

# 結果を表示
st.header("最適化結果")
st.subheader(f"総コスト: {results['objective_value']:.2f}")

# 結果の表を作成
result_df = pd.DataFrame({
    "月": months,
    "需要": demands,
    "発注量": [round(q, 1) for q in results["order_quantities"]],
    "発注の有無": [int(y) for y in results["setup_indicators"]],
    "正味在庫": [round(net, 1) for net in results["net_inventory"]],
    "合計コスト": [round(tc, 1) for tc in results["total_costs"]]
})

st.table(result_df)

# グラフで可視化
st.subheader("月ごとの在庫・需要・発注の推移")

# データフレームを準備
chart_data = pd.DataFrame({
    '月': months,
    '需要': demands,
    '発注量': results["order_quantities"],
    '正味在庫': results["net_inventory"]
})

# 長形式に変換
chart_data_long = pd.melt(chart_data, id_vars=['月'], value_vars=['需要', '発注量', '正味在庫'],
                      var_name='項目', value_name='値')

# Altairでグラフ作成
chart = alt.Chart(chart_data_long).mark_line(point=True).encode(
    x=alt.X('月', sort=None),
    y='値',
    color='項目',
    tooltip=['月', '項目', '値']
).properties(
    width=700,
    height=400
)

st.altair_chart(chart, use_container_width=True)
