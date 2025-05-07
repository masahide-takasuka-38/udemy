import streamlit as st
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# アプリのタイトル
st.title("材料切断最適化問題（カッティングストック問題）")

# デフォルトの切断パターンデータ
default_patterns = {
    "パターン": ["パターン1", "パターン2", "パターン3", "パターン4", "パターン5", "パターン6"],
    "1.2m": [5, 0, 0, 0, 0, 1],
    "1.5m": [0, 4, 1, 0, 2, 0],
    "2.1m": [0, 0, 2, 0, 0, 1],
    "2.8m": [0, 0, 0, 2, 1, 1],
    "mill ends (m)": [0.0, 0.0, 0.3, 0.4, 0.2, 0.0]
}

# デフォルトの需要データ
default_demands = {
    "製品サイズ": ["1.2m", "1.5m", "2.1m", "2.8m"],
    "需要 (本)": [50, 80, 30, 40]
}

# パラメータを編集可能にする
st.subheader("パラメータ設定")

# 切断パターンの編集
st.write("切断パターン（各パターンで何本の各製品が作れるかを指定）:")
edited_patterns = st.data_editor(
    pd.DataFrame(default_patterns),
    num_rows="fixed",
    column_config={
        "パターン": st.column_config.TextColumn("パターン", disabled=True),
        "1.2m": st.column_config.NumberColumn("1.2m", min_value=0, step=1, format="%d本"),
        "1.5m": st.column_config.NumberColumn("1.5m", min_value=0, step=1, format="%d本"),
        "2.1m": st.column_config.NumberColumn("2.1m", min_value=0, step=1, format="%d本"),
        "2.8m": st.column_config.NumberColumn("2.8m", min_value=0, step=1, format="%d本"),
        "mill ends (m)": st.column_config.NumberColumn("端材 (m)", min_value=0.0, format="%.1fm")
    },
    use_container_width=True
)

# 需要の編集
st.write("製品需要:")
edited_demands = st.data_editor(
    pd.DataFrame(default_demands),
    num_rows="fixed",
    column_config={
        "製品サイズ": st.column_config.TextColumn("製品サイズ", disabled=True),
        "需要 (本)": st.column_config.NumberColumn("需要 (本)", min_value=0, step=1, format="%d本")
    },
    use_container_width=True
)

# 材料長さの編集
material_length = st.number_input("材料の長さ (m)", min_value=1.0, value=6.0, step=0.1)

# 値を確実に整数に変換する関数
def safe_int(value):
    try:
        return int(float(value))
    except:
        return 0

# 需要値を取得 (安全に整数に変換)
demand_1_2 = safe_int(edited_demands.loc[0, "需要 (本)"])
demand_1_5 = safe_int(edited_demands.loc[1, "需要 (本)"])
demand_2_1 = safe_int(edited_demands.loc[2, "需要 (本)"])
demand_2_8 = safe_int(edited_demands.loc[3, "需要 (本)"])

# PuLPモデルを定義して解く関数
def solve_cutting_stock():
    # 問題の定義
    model = pulp.LpProblem("CuttingStockProblem", pulp.LpMinimize)
    
    # 決定変数: 各パターンの使用回数
    y = {j: pulp.LpVariable(f"y_{j}", lowBound=0, cat='Integer') for j in range(1, 7)}
    
    # 目的関数: 材料使用本数の最小化
    model += pulp.lpSum(y[j] for j in range(1, 7)), "材料使用本数"
    
    # 制約条件 (安全に整数に変換して)
    pattern_1_2m = [safe_int(edited_patterns.loc[j-1, "1.2m"]) for j in range(1, 7)]
    pattern_1_5m = [safe_int(edited_patterns.loc[j-1, "1.5m"]) for j in range(1, 7)]
    pattern_2_1m = [safe_int(edited_patterns.loc[j-1, "2.1m"]) for j in range(1, 7)]
    pattern_2_8m = [safe_int(edited_patterns.loc[j-1, "2.8m"]) for j in range(1, 7)]
    
    model += pulp.lpSum(pattern_1_2m[j-1] * y[j] for j in range(1, 7)) >= demand_1_2, "製品1.2mの需要"
    model += pulp.lpSum(pattern_1_5m[j-1] * y[j] for j in range(1, 7)) >= demand_1_5, "製品1.5mの需要"
    model += pulp.lpSum(pattern_2_1m[j-1] * y[j] for j in range(1, 7)) >= demand_2_1, "製品2.1mの需要"
    model += pulp.lpSum(pattern_2_8m[j-1] * y[j] for j in range(1, 7)) >= demand_2_8, "製品2.8mの需要"
    
    # 解く
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 結果を抽出
    if pulp.LpStatus[model.status] == 'Optimal':
        pattern_waste = [float(edited_patterns.loc[j-1, "mill ends (m)"]) for j in range(1, 7)]
        
        result = {
            "status": "最適解が見つかりました",
            "patterns": {j: pulp.value(y[j]) for j in range(1, 7)},
            "total_materials": pulp.value(model.objective),
            "produced": {
                "1.2m": sum(pattern_1_2m[j-1] * pulp.value(y[j]) for j in range(1, 7)),
                "1.5m": sum(pattern_1_5m[j-1] * pulp.value(y[j]) for j in range(1, 7)),
                "2.1m": sum(pattern_2_1m[j-1] * pulp.value(y[j]) for j in range(1, 7)),
                "2.8m": sum(pattern_2_8m[j-1] * pulp.value(y[j]) for j in range(1, 7))
            },
            "total_waste": sum(pattern_waste[j-1] * pulp.value(y[j]) for j in range(1, 7))
        }
        return result
    else:
        return {"status": "最適解が見つかりませんでした"}

# 最適化を実行
if st.button("最適化を実行"):
    with st.spinner("計算中..."):
        result = solve_cutting_stock()
    
    if result["status"] == "最適解が見つかりました":
        st.success(f"最適解が見つかりました！使用材料: {int(result['total_materials'])}本")
        
        # パターンの使用回数表示
        pattern_usage = {
            "パターン": [f"パターン{j}" for j in range(1, 7)],
            "使用回数": [int(result["patterns"][j]) for j in range(1, 7)]
        }
        pattern_usage_df = pd.DataFrame(pattern_usage)
        st.subheader("パターン使用回数")
        st.table(pattern_usage_df)
        
        # 製品生産本数と需要の比較
        production_vs_demand = {
            "製品サイズ": ["1.2m", "1.5m", "2.1m", "2.8m"],
            "生産本数": [
                int(result["produced"]["1.2m"]),
                int(result["produced"]["1.5m"]),
                int(result["produced"]["2.1m"]),
                int(result["produced"]["2.8m"])
            ],
            "需要本数": [
                demand_1_2,
                demand_1_5,
                demand_2_1,
                demand_2_8
            ],
            "余剰": [
                int(result["produced"]["1.2m"]) - demand_1_2,
                int(result["produced"]["1.5m"]) - demand_1_5,
                int(result["produced"]["2.1m"]) - demand_2_1,
                int(result["produced"]["2.8m"]) - demand_2_8
            ]
        }
        production_df = pd.DataFrame(production_vs_demand)
        st.subheader("製品生産本数と需要の比較")
        st.table(production_df)
        
        # 廃棄物の量
        st.subheader("廃棄物")
        total_waste = result["total_waste"]
        total_materials = max(1, result["total_materials"])  # ゼロ除算防止
        st.info(f"総端材: {total_waste:.2f}m (平均: {total_waste/total_materials:.2f}m/本)")
        
        # グラフで最適解を可視化
        st.subheader("最適解の可視化")
        
        # 使用されるパターンを抽出
        used_patterns = [j for j in range(1, 7) if result["patterns"][j] > 0]
        
        # 各パターンの詳細を表示
        colors = {
            "1.2m": "#FF9999",  # 赤系
            "1.5m": "#66B2FF",  # 青系
            "2.1m": "#99FF99",  # 緑系
            "2.8m": "#FFCC99",  # オレンジ系
            "mill ends": "#CCCCCC"   # グレー
        }
        
        # パターンの詳細データを作成
        pattern_details = []
        for j in used_patterns:
            pattern = []
            current_pos = 0
            pattern_idx = j - 1  # pythonのインデックスに合わせる
            
            # 各製品サイズを順番に配置
            product_sizes = {"1.2m": 1.2, "1.5m": 1.5, "2.1m": 2.1, "2.8m": 2.8}
            
            for size_name, size_value in product_sizes.items():
                count = safe_int(edited_patterns.loc[pattern_idx, size_name])
                for _ in range(count):
                    pattern.append({
                        "size": size_value, 
                        "start": current_pos, 
                        "type": size_name
                    })
                    current_pos += size_value
            
            # mill endsがあれば追加
            waste = float(edited_patterns.loc[pattern_idx, "mill ends (m)"])
            if waste > 0:
                pattern.append({
                    "size": waste,
                    "start": current_pos,
                    "type": "mill ends"
                })
            
            pattern_details.append({
                "pattern_id": j,
                "count": int(result["patterns"][j]),
                "items": pattern
            })
        
        # 最適解を可視化するグラフの作成
        if used_patterns:  # 使用されるパターンが存在する場合のみグラフ作成
            fig, ax = plt.subplots(figsize=(10, 8))
            
            bar_height = 0.6
            y_position = 0
            
            # 各パターンごとに棒グラフを描画
            for pattern in pattern_details:
                pattern_id = pattern["pattern_id"]
                count = pattern["count"]
                items = pattern["items"]
                
                # パターン番号と使用回数を左側に表示
                ax.text(-0.5, y_position + bar_height/2, f"Patterns{pattern_id} ({int(count)}p)",
                        va='center', ha='right', fontweight='bold')
                
                # 各アイテムを描画
                for item in items:
                    rect = patches.Rectangle(
                        (item["start"], y_position),
                        item["size"],
                        bar_height,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=colors[item["type"]]
                    )
                    ax.add_patch(rect)
                    
                    # アイテムサイズをラベルとして表示
                    if item["type"] != "mill ends" or item["size"] >= 0.2:  # mill endsが小さすぎる場合はラベル非表示
                        ax.text(
                            item["start"] + item["size"]/2,
                            y_position + bar_height/2,
                            f"{item['type']}",
                            ha='center',
                            va='center',
                            fontsize=9
                        )
                
                y_position += 1.2  # 次のパターンのための位置調整
            
            # グラフの設定
            ax.set_xlim(-0.5, material_length + 0.5)
            ax.set_ylim(0, max(1, y_position))
            ax.set_xlabel('Length (m)')
            ax.set_yticks([])
            ax.set_title('Optimal cutting pattern')
            
            # グリッド線
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # 凡例
            handles = [patches.Patch(facecolor=colors[key], edgecolor='black', label=key) for key in colors]
            ax.legend(handles=handles, loc='upper right')
            
            # 材料全体の長さを示す
            for i in range(int(y_position // 1.2)):
                ax.axvline(x=material_length, ymin=i*1.2/max(1, y_position), ymax=(i*1.2+bar_height)/max(1, y_position),
                          color='black', linestyle='-', linewidth=2)
                ax.text(material_length + 0.05, i*1.2 + bar_height/2, f"{material_length}m", va='center')
            
            st.pyplot(fig)
        else:
            st.warning("使用するパターンがありません")
        
        # 製品生産量と需要の比較グラフ
        st.subheader("製品サイズ別の生産量と需要")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(production_vs_demand["製品サイズ"]))
        width = 0.35
        
        # リストが空でないことを確認
        if len(production_vs_demand["生産本数"]) > 0 and len(production_vs_demand["需要本数"]) > 0:
            rects1 = ax2.bar(x - width/2, production_vs_demand["生産本数"], width, label='Number of units produced')
            rects2 = ax2.bar(x + width/2, production_vs_demand["需要本数"], width, label='quantity demanded')
            
            ax2.set_xlabel('Product size')
            ax2.set_ylabel('number')
            ax2.set_title('Production and demand by product size')
            ax2.set_xticks(x)
            ax2.set_xticklabels(production_vs_demand["製品サイズ"])
            ax2.legend()
            
            # バーの上に値を表示
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax2.annotate(f'{int(height)}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            st.pyplot(fig2)
        else:
            st.warning("グラフを表示するためのデータがありません")
        
    else:
        st.error(result["status"])
