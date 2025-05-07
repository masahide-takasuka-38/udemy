import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# アプリのタイトル
st.title("生産計画最適化問題")

# デフォルトパラメータ値
default_params = {
    "製品": ["製品A", "製品B", "製品C"],
    "利益率 (円/個)": [500, 600, 630],
    "機械X加工時間 (分/個)": [10, 8, 12],
    "機械Y加工時間 (分/個)": [6, 12, 8],
    "機械Z加工時間 (分/個)": [8, 9, 10]
}

default_machines = {
    "機械": ["機械X", "機械Y", "機械Z"],
    "稼働可能時間 (分)": [2400, 2000, 1800]
}

# パラメータ入力
st.header("パラメータ設定")

# 製品パラメータをデータエディタで編集可能にする
st.subheader("製品パラメータ")
df_params = pd.DataFrame(default_params)

edited_df_params = st.data_editor(
    df_params,
    num_rows="fixed",
    column_config={
        "製品": st.column_config.TextColumn("製品"),
        "利益率 (円/個)": st.column_config.NumberColumn(
            "利益率 (円/個)",
            min_value=0,
            format="%d円"
        ),
        "機械X加工時間 (分/個)": st.column_config.NumberColumn(
            "機械X加工時間 (分/個)",
            min_value=0,
            format="%d分"
        ),
        "機械Y加工時間 (分/個)": st.column_config.NumberColumn(
            "機械Y加工時間 (分/個)",
            min_value=0,
            format="%d分"
        ),
        "機械Z加工時間 (分/個)": st.column_config.NumberColumn(
            "機械Z加工時間 (分/個)",
            min_value=0,
            format="%d分"
        ),
    },
    use_container_width=True
)

# 機械パラメータをデータエディタで編集可能にする
st.subheader("機械稼働可能時間")
df_machines = pd.DataFrame(default_machines)

edited_df_machines = st.data_editor(
    df_machines,
    num_rows="fixed",
    column_config={
        "機械": st.column_config.TextColumn("機械"),
        "稼働可能時間 (分)": st.column_config.NumberColumn(
            "稼働可能時間 (分)",
            min_value=0,
            format="%d分"
        ),
    },
    use_container_width=True
)

# 編集されたパラメータを取得
p_A = edited_df_params.loc[0, "利益率 (円/個)"]
p_B = edited_df_params.loc[1, "利益率 (円/個)"]
p_C = edited_df_params.loc[2, "利益率 (円/個)"]

t_X_A = edited_df_params.loc[0, "機械X加工時間 (分/個)"]
t_X_B = edited_df_params.loc[1, "機械X加工時間 (分/個)"]
t_X_C = edited_df_params.loc[2, "機械X加工時間 (分/個)"]

t_Y_A = edited_df_params.loc[0, "機械Y加工時間 (分/個)"]
t_Y_B = edited_df_params.loc[1, "機械Y加工時間 (分/個)"]
t_Y_C = edited_df_params.loc[2, "機械Y加工時間 (分/個)"]

t_Z_A = edited_df_params.loc[0, "機械Z加工時間 (分/個)"]
t_Z_B = edited_df_params.loc[1, "機械Z加工時間 (分/個)"]
t_Z_C = edited_df_params.loc[2, "機械Z加工時間 (分/個)"]

t_X = edited_df_machines.loc[0, "稼働可能時間 (分)"]
t_Y = edited_df_machines.loc[1, "稼働可能時間 (分)"]
t_Z = edited_df_machines.loc[2, "稼働可能時間 (分)"]

# PuLPで最適化問題を解く
def solve_production_planning():
    # 問題の定義
    prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)
    
    # 決定変数の定義
    x_A = pulp.LpVariable("x_A", lowBound=0, cat='Integer')
    x_B = pulp.LpVariable("x_B", lowBound=0, cat='Integer')
    x_C = pulp.LpVariable("x_C", lowBound=0, cat='Integer')
    
    # 目的関数の定義
    prob += p_A * x_A + p_B * x_B + p_C * x_C, "利益"
    
    # 制約条件の定義
    prob += t_X_A * x_A + t_X_B * x_B + t_X_C * x_C <= t_X, "機械X制約"
    prob += t_Y_A * x_A + t_Y_B * x_B + t_Y_C * x_C <= t_Y, "機械Y制約"
    prob += t_Z_A * x_A + t_Z_B * x_B + t_Z_C * x_C <= t_Z, "機械Z制約"
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 結果を返す
    if pulp.LpStatus[prob.status] == 'Optimal':
        return {
            "status": "最適解が見つかりました",
            "x_A": pulp.value(x_A),
            "x_B": pulp.value(x_B),
            "x_C": pulp.value(x_C),
            "profit": pulp.value(prob.objective),
            "machine_X_usage": t_X_A * pulp.value(x_A) + t_X_B * pulp.value(x_B) + t_X_C * pulp.value(x_C),
            "machine_Y_usage": t_Y_A * pulp.value(x_A) + t_Y_B * pulp.value(x_B) + t_Y_C * pulp.value(x_C),
            "machine_Z_usage": t_Z_A * pulp.value(x_A) + t_Z_B * pulp.value(x_B) + t_Z_C * pulp.value(x_C)
        }
    else:
        return {"status": "最適解が見つかりませんでした"}

# 最適化を実行
if st.button("最適化を実行"):
    with st.spinner("計算中..."):
        result = solve_production_planning()
    
    if result["status"] == "最適解が見つかりました":
        # 結果の表示
        st.subheader("最適解")
        st.success(f"総利益: {result['profit']:,.0f}円")
        
        # 生産量の表示
        production_data = {
            "製品": ["製品A", "製品B", "製品C"],
            "生産量 (個)": [int(result["x_A"]), int(result["x_B"]), int(result["x_C"])],
            "利益 (円)": [
                int(result["x_A"] * p_A),
                int(result["x_B"] * p_B),
                int(result["x_C"] * p_C)
            ]
        }
        
        df_production = pd.DataFrame(production_data)
        st.table(df_production)
        
        # 機械使用時間の表示
        machine_usage_data = {
            "機械": ["機械X", "機械Y", "機械Z"],
            "使用時間 (分)": [
                int(result["machine_X_usage"]),
                int(result["machine_Y_usage"]),
                int(result["machine_Z_usage"])
            ],
            "稼働可能時間 (分)": [t_X, t_Y, t_Z],
            "使用率 (%)": [
                round(result["machine_X_usage"] / t_X * 100, 1),
                round(result["machine_Y_usage"] / t_Y * 100, 1),
                round(result["machine_Z_usage"] / t_Z * 100, 1)
            ]
        }
        
        df_machine_usage = pd.DataFrame(machine_usage_data)
        st.table(df_machine_usage)
        
        # 機械使用率のグラフ描画
        st.subheader("機械使用率グラフ")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        machines = ["Machine X", "Machine Y", "Machine Z"]
        usage = [result["machine_X_usage"], result["machine_Y_usage"], result["machine_Z_usage"]]
        capacity = [t_X, t_Y, t_Z]
        
        x = np.arange(len(machines))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, usage, width, label='usage time')
        rects2 = ax.bar(x + width/2, capacity, width, label='uptime', alpha=0.7)
        
        ax.set_ylabel('Time (min)')
        ax.set_title('Hours of use and uptime per machine')
        ax.set_xticks(x)
        ax.set_xticklabels(machines)
        ax.legend()
        
        # バーの上に値を表示
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        st.pyplot(fig)
        
        # 製品別生産量と利益のグラフ
        st.subheader("製品別生産量と利益")
        fig2, ax1 = plt.subplots(figsize=(10, 6))
        
        products = ["Product A", "Product B", "Product C"]
        quantities = [result["x_A"], result["x_B"], result["x_C"]]
        profits = [
            result["x_A"] * p_A,
            result["x_B"] * p_B,
            result["x_C"] * p_C
        ]
        
        x = np.arange(len(products))
        width = 0.35
        
        rects1 = ax1.bar(x, quantities, width, label='Production quantity (pcs)', color='skyblue')
        
        ax1.set_ylabel('Production quantity (pcs)')
        ax1.set_title('Production and Profit by Product')
        ax1.set_xticks(x)
        ax1.set_xticklabels(products)
        
        # 利益の右軸を作成
        ax2 = ax1.twinx()
        ax2.set_ylabel('Profit (yen)')
        line = ax2.plot(x, profits, 'ro-', label='Profit (yen)')
        
        # 凡例の結合
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # バーの上に値を表示
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        
        # 利益点の値を表示
        for i, profit in enumerate(profits):
            ax2.annotate(f'{profit:,.0f}(yen)',
                        xy=(i, profit),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        st.pyplot(fig2)
        
    else:
        st.error(result["status"])
