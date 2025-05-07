import streamlit as st
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# アプリのタイトル
st.title("生産スケジューリング最適化問題")

# デフォルトのジョブデータ
default_jobs_data = {
    'Job': ['Job 1', 'Job 2', 'Job 3', 'Job 4', 'Job 5'],
    'p_j': [4, 5, 4, 6, 3],
    'd_j': [8, 5, 19, 17, 25],
    'w_j': [2, 4, 1, 2, 5]
}

# 入力データ編集用のフォーム
st.subheader("ジョブデータ入力")
st.write("以下の表でジョブデータを編集できます：")

# データフレームを作成し、編集可能にする
jobs_data_edit = st.data_editor(
    pd.DataFrame(default_jobs_data),
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "Job": st.column_config.TextColumn("Job Name"),
        "p_j": st.column_config.NumberColumn("Processing Time", min_value=1),
        "d_j": st.column_config.NumberColumn("Due Date", min_value=1),
        "w_j": st.column_config.NumberColumn("Weight", min_value=1)
    }
)

# 編集されたデータを取得
jobs_data = {
    'Job': jobs_data_edit['Job'].tolist(),
    'p_j': jobs_data_edit['p_j'].tolist(),
    'd_j': jobs_data_edit['d_j'].tolist(),
    'w_j': jobs_data_edit['w_j'].tolist()
}

# PuLPで問題を解く
def solve_job_scheduling():
    # ジョブ数
    n = len(jobs_data['Job'])
    
    # ジョブのインデックス
    jobs = range(n)
    
    # パラメータ取得
    p = jobs_data['p_j']
    d = jobs_data['d_j']
    w = jobs_data['w_j']
    
    # 十分大きな数M
    M = sum(p) + max(d)
    
    # 問題定義
    prob = pulp.LpProblem("JobScheduling", pulp.LpMinimize)
    
    # 決定変数
    C = {j: pulp.LpVariable(f"C_{j}", lowBound=0) for j in jobs}
    T = {j: pulp.LpVariable(f"T_{j}", lowBound=0) for j in jobs}
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary) 
         for i in jobs for j in jobs if i != j}
    
    # 目的関数
    prob += pulp.lpSum(w[j] * T[j] for j in jobs)
    
    # 制約条件
    # 納期遅れ定義
    for j in jobs:
        prob += T[j] >= C[j] - d[j]
    
    # ジョブ順序と完了時間の関係
    for i in jobs:
        for j in jobs:
            if i != j:
                prob += C[i] + p[j] <= C[j] + M * (1 - x[(i, j)])
    
    # ジョブ順序の一貫性
    for i in jobs:
        for j in jobs:
            if i < j:  # 対称性を避けるため片方だけ定義
                prob += x[(i, j)] + x[(j, i)] == 1
    
    # 完了時間は処理時間以上
    for j in jobs:
        prob += C[j] >= p[j]
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 結果を抽出
    if pulp.LpStatus[prob.status] == 'Optimal':
        completion_times = [pulp.value(C[j]) for j in jobs]
        tardiness = [pulp.value(T[j]) for j in jobs]
        
        # ジョブの順序を決定
        job_order = []
        for j in jobs:
            start_time = pulp.value(C[j]) - p[j]
            job_order.append((j, start_time, pulp.value(C[j])))
        
        # 開始時間でソート
        job_order.sort(key=lambda x: x[1])
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(prob.objective),
            'completion_times': completion_times,
            'tardiness': tardiness,
            'job_order': job_order,
            'processing_times': p,
            'due_dates': d,
            'weights': w
        }
    else:
        return {'status': 'Not Solved'}

# 問題を解くボタン
if st.button("スケジューリング問題を解く"):
    # 結果を表示
    with st.spinner("問題を解いています..."):
        results = solve_job_scheduling()

    if results['status'] == 'Optimal':
        st.success(f"最適解が見つかりました！総加重納期遅れ: {results['objective_value']:.2f}")
        
        # 結果のテーブル
        result_df = pd.DataFrame({
            'Job': jobs_data['Job'],
            'Processing Time (p_j)': results['processing_times'],
            'Due Date (d_j)': results['due_dates'],
            'Weight (w_j)': results['weights'],
            'Completion Time (C_j)': [round(c, 2) for c in results['completion_times']],
            'Tardiness (T_j)': [round(t, 2) for t in results['tardiness']],
            'Weighted Tardiness (w_j * T_j)': [round(results['weights'][j] * results['tardiness'][j], 2) for j in range(len(results['weights']))]
        })
        
        st.subheader("詳細結果")
        st.table(result_df)
        
        # ジョブの順序
        job_sequence = [jobs_data['Job'][job[0]] for job in results['job_order']]
        st.subheader("最適なジョブ順序")
        st.write(" → ".join(job_sequence))
        
        # ガントチャート描画
        st.subheader("スケジュールのガントチャート")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # ジョブごとの色を設定
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF', '#C2C2F0', '#FFD700', '#90EE90']
        
        # ジョブの処理時間バーを描画
        y_pos = 0
        for i, job_info in enumerate(results['job_order']):
            j, start_time, completion_time = job_info
            job_name = jobs_data['Job'][j]
            
            # ジョブバー
            rect = patches.Rectangle(
                (start_time, y_pos),
                results['processing_times'][j],
                0.6,
                facecolor=colors[j % len(colors)],
                edgecolor='black',
                label=job_name
            )
            ax.add_patch(rect)
            
            # ジョブ名とデータを中央に表示
            ax.text(
                start_time + results['processing_times'][j]/2, 
                y_pos + 0.3, 
                f"{job_name}\np={results['processing_times'][j]}, d={results['due_dates'][j]}, w={results['weights'][j]}", 
                ha='center', 
                va='center',
                fontsize=9
            )
            
            # 納期を縦線で表示
            ax.axvline(x=results['due_dates'][j], color='red', linestyle='--', alpha=0.5)
            ax.text(results['due_dates'][j], -0.2, f"d_{j+1}", ha='center', fontsize=8)
            
            y_pos += 1
        
        # グラフの設定
        ax.set_ylim(-0.5, len(results['job_order']))
        ax.set_xlim(0, max([job[2] for job in results['job_order']]) + 1)
        ax.set_xlabel('Time')
        ax.set_yticks([])
        ax.set_title('Job Schedule')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 凡例
        handles = [patches.Patch(color=colors[j % len(colors)], label=jobs_data['Job'][j]) for j in range(len(jobs_data['Job']))]
        ax.legend(handles=handles, loc='upper right')
        
        st.pyplot(fig)
        
        # 加重納期遅れの内訳
        st.subheader("加重納期遅れの内訳")
        
        weighted_tardiness = [results['weights'][j] * results['tardiness'][j] for j in range(len(results['weights']))]
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.bar(jobs_data['Job'], weighted_tardiness, color=colors[:len(jobs_data['Job'])])
        
        ax2.set_xlabel('Job')
        ax2.set_ylabel('Weighted Tardiness (w_j * T_j)')
        ax2.set_title('Weighted Tardiness by Job')
        
        # バーの上に値を表示
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.1f}',
                        ha='center', va='bottom')
        
        st.pyplot(fig2)

    else:
        st.error("問題を解くことができませんでした。")
