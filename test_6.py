import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go

# ----------------------------------------------------
# 0) CSV Load & Preprocessing
# ----------------------------------------------------
@st.cache_data
def load_data():
    """
    Reads 'score.csv' and preprocesses:
      - Keep columns: Week, Team, KPI, Actual, Final
      - Remove non-numeric characters from Actual
      - Convert Actual, Final to float
      - Drop rows where Actual or Final is NaN
      - Extract numeric part of Week (W1 -> 1) as Week_num
    """
    # CSV 파일 경로를 필요에 맞게 조정하세요.
    csv_path = "score.csv"
    df = pd.read_csv(csv_path)

    # Actual 컬럼에서 숫자, '.', '-' 외의 문자를 제거
    df['Actual'] = (
        df['Actual'].astype(str)
        .str.replace(r'[^0-9.\-]', '', regex=True)
    )
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    df['Final'] = pd.to_numeric(df['Final'], errors='coerce')

    # Actual 또는 Final 값이 NaN인 행 제거
    df.dropna(subset=['Actual', 'Final'], inplace=True)

    # Week에서 숫자만 추출 (예: W1 -> 1)
    df['Week_num'] = df['Week'].str.extract(r'(\d+)').astype(int)

    return df

# ----------------------------------------------------
# 1) 데이터 로드
# ----------------------------------------------------
df = load_data()

# ----------------------------------------------------
# 2) 변수 설정: KPI, Team, Week 등
# ----------------------------------------------------
all_kpis = sorted(df['KPI'].dropna().unique())
all_teams = sorted(df['Team'].dropna().unique())
unique_weeks = sorted(df['Week'].unique(), key=lambda w: int(w[1:]))
week_options = unique_weeks + ["Total"]

# ----------------------------------------------------
# 3) 사이드바 필터 (공통)
# ----------------------------------------------------
st.sidebar.title("Common Filters")
selected_kpi = st.sidebar.selectbox("Select a KPI (Target)", options=all_kpis)
selected_week_or_total = st.sidebar.selectbox("Select Week (or 'Total')", options=week_options)
selected_team = st.sidebar.selectbox("Select a Team", options=all_teams)

# ----------------------------------------------------
# 4) 피벗테이블 생성: Final (sum) & Actual (mean) for selected_kpi
# ----------------------------------------------------
# Final 데이터 집계
df_kpi_final = df[df['KPI'] == selected_kpi].copy()
grouped_final = df_kpi_final.groupby(['Week', 'Team'], as_index=False)['Final'].sum()
pivot_final = grouped_final.pivot(index='Team', columns='Week', values='Final').fillna(0)
pivot_final['Total'] = pivot_final.sum(axis=1)

# Actual 데이터 집계
df_kpi_act = df[df['KPI'] == selected_kpi].copy()
grouped_act = df_kpi_act.groupby(['Week', 'Team'], as_index=False)['Actual'].mean()
pivot_act = grouped_act.pivot(index='Team', columns='Week', values='Actual').fillna(0)
pivot_act['Average'] = pivot_act.mean(axis=1)

# ----------------------------------------------------
# Row1: "Target : ... best score : ... No. 1 team : ..."
# ----------------------------------------------------
st.subheader("Row1: Max Score Info")
if pivot_final.empty:
    st.write("No Final data.")
else:
    max_total = pivot_final['Total'].max()
    max_team = pivot_final['Total'].idxmax()
    st.markdown(f"""**Target** : {selected_kpi}  
**Best score** : {max_total:.2f}  
**No. 1 team** : {max_team}""")

# ----------------------------------------------------
# Row2: Final Score 차트
# ----------------------------------------------------
st.subheader("Row2: Final Score Chart")
if pivot_final.empty or (selected_week_or_total not in pivot_final.columns):
    st.write("No data for this chart.")
else:
    bar_df_final = pivot_final[[selected_week_or_total]].reset_index()
    bar_df_final.columns = ['Team', 'Final_Value']
    chart_final = (
        alt.Chart(bar_df_final)
        .mark_bar()
        .encode(
            x=alt.X('Team:N', sort=None),
            y=alt.Y('Final_Value:Q'),
            tooltip=['Team', 'Final_Value']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart_final, use_container_width=False)

# ----------------------------------------------------
# Row3: Final Table
# ----------------------------------------------------
st.subheader("Row3: Final Table")
if pivot_final.empty:
    st.write("No Final data.")
else:
    st.dataframe(pivot_final.style.format(precision=2))

# ----------------------------------------------------
# Row4: Actual Chart
# ----------------------------------------------------
st.subheader("Row4: Actual Chart")
actual_col = selected_week_or_total
if actual_col == "Total":
    actual_col = "Average"
if pivot_act.empty or (actual_col not in pivot_act.columns):
    st.write("No data for this chart.")
else:
    bar_df_act = pivot_act[[actual_col]].reset_index()
    bar_df_act.columns = ['Team', 'Actual_Value']
    chart_act = (
        alt.Chart(bar_df_act)
        .mark_bar(color='orange')
        .encode(
            x=alt.X('Team:N', sort=None),
            y=alt.Y('Actual_Value:Q'),
            tooltip=['Team', 'Actual_Value']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart_act, use_container_width=False)

# ----------------------------------------------------
# Row5: Actual Table
# ----------------------------------------------------
st.subheader("Row5: Actual Table")
if pivot_act.empty:
    st.write("No Actual data.")
else:
    st.dataframe(pivot_act.style.format(precision=2))

# ----------------------------------------------------
# Row6: Rank Chart (선택한 팀 기준)
# ----------------------------------------------------
st.subheader("Row6: Rank Chart")
df_rank = df.copy()
df_rank['Rank'] = df_rank.groupby(['Week', 'KPI'])['Final'].rank(method='dense', ascending=False)
df_team_rank = df_rank[df_rank['Team'] == selected_team].copy()
df_team_rank = df_team_rank.sort_values(['KPI', 'Week_num'])

if df_team_rank.empty:
    st.write("No Rank data for the selected team.")
else:
    line_chart = (
        alt.Chart(df_team_rank)
        .mark_line(point=True)
        .encode(
            x=alt.X('Week_num:Q', axis=alt.Axis(title='Weeknum')),
            y=alt.Y('Rank:Q', sort='descending', axis=alt.Axis(title='Rank (1=best)')),
            color=alt.Color('KPI:N', legend=alt.Legend(title='KPI')),
            tooltip=['Week', 'KPI', 'Final', 'Rank']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(line_chart, use_container_width=False)

# ----------------------------------------------------
# Row7: Rank Table
# ----------------------------------------------------
st.subheader("Row7: Rank Table")
if df_team_rank.empty:
    st.write("No Rank data.")
else:
    rank_pivot = df_team_rank.pivot(index='KPI', columns='Week', values='Rank')
    rank_pivot['Total'] = rank_pivot.sum(axis=1)
    st.dataframe(rank_pivot.fillna("").style.format(precision=2))

# ----------------------------------------------------
# Row8: Radar Chart (선택한 팀 vs. 전체 최고)
# ----------------------------------------------------
st.subheader("Row8: Radar Chart")
if df_team_rank.empty:
    st.write("No data for Radar Chart.")
else:
    df_final_only = df.copy()
    team_mean = (df_final_only[df_final_only['Team'] == selected_team]
                 .groupby('KPI')['Final'].mean()
                 .rename('Team_Mean')
                 .reset_index())
    max_final = (df_final_only.groupby('KPI')['Final'].max()
                 .rename('Max_Final')
                 .reset_index())
    radar_df = pd.merge(team_mean, max_final, on='KPI', how='outer').fillna(0)
    if radar_df.empty:
        st.write("No radar data.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_df['Max_Final'],
            theta=radar_df['KPI'],
            fill='toself',
            name='Max'
        ))
        fig.add_trace(go.Scatterpolar(
            r=radar_df['Team_Mean'],
            theta=radar_df['KPI'],
            fill='toself',
            name=f"{selected_team} Mean"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, radar_df['Max_Final'].max() * 1.1]
                )
            ),
            showlegend=True,
            width=700,
            height=400
        )
        st.plotly_chart(fig, use_container_width=False)

# ----------------------------------------------------
# Row9: 선택한 팀의 Actual Table
# ----------------------------------------------------
st.subheader("Row9: The Team's Actual Table")
df_team_act = df[df['Team'] == selected_team].copy()
if df_team_act.empty:
    st.write("No Actual data for this team.")
else:
    team_act_pivot = df_team_act.pivot(index='KPI', columns='Week', values='Actual')
    st.dataframe(team_act_pivot.fillna("").style.format(precision=2))

# ----------------------------------------------------
# Row10: Overall Summation Chart (모든 KPI)
# ----------------------------------------------------
st.subheader("Row10: Overall Summation Chart")
df_total_all_kpi = df.groupby(['Week', 'Team'], as_index=False)['Final'].sum()

if df_total_all_kpi.empty:
    st.write("No overall data.")
else:
    chart_overall = (
        alt.Chart(df_total_all_kpi)
        .mark_bar()
        .encode(
            x=alt.X('Team:N', sort='-y'),
            y=alt.Y('Final:Q', stack='zero'),
            color='Week:N',
            tooltip=['Team', 'Week', 'Final']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart_overall, use_container_width=False)

# ----------------------------------------------------
# Row11: Overall Summation Table & Ranking
# ----------------------------------------------------
st.subheader("Row11: Overall Summation Table")
if df_total_all_kpi.empty:
    st.write("No overall data.")
else:
    pivot_total_all = df_total_all_kpi.pivot(index='Team', columns='Week', values='Final').fillna(0)
    pivot_total_all['GrandTotal'] = pivot_total_all.sum(axis=1)

    st.write("Overall Summation Table (all KPI, all weeks)")
    st.dataframe(pivot_total_all.style.format(precision=2))

    # GrandTotal 기준 내림차순 정렬 후 랭킹 부여
    ranking_df = pivot_total_all[['GrandTotal']].copy()
    ranking_df = ranking_df.sort_values('GrandTotal', ascending=False)
    ranking_df['Rank'] = ranking_df['GrandTotal'].rank(method='dense', ascending=False).astype(int)
    st.dataframe(ranking_df.style.format(precision=2))

st.write("---")
st.write("End of layout version 1 by HWK QIP.")

# ----------------------------------------------------
# Row12: KPI별 주차별 점수 및 HWK Total (팀별 점수와 전체 평균)
# ----------------------------------------------------
st.subheader("Row12: KPI별 주차별 점수 & HWK Total (팀별 점수와 전체 평균)")

# x축에 사용할 순서 (unique_weeks + HWK Total)
period_order = unique_weeks + ["HWK Total"]

# 모든 KPI(7개 항목)에 대해 반복 처리
for kpi in all_kpis:
    st.markdown(f"### KPI: {kpi}")
    
    # KPI별 데이터 필터링
    df_kpi = df[df['KPI'] == kpi].copy()
    if df_kpi.empty:
        st.write("해당 KPI에 대한 데이터가 없습니다.")
        continue

    # 팀별, 주차별 Final 점수 합계를 구함
    pivot = df_kpi.groupby(['Team', 'Week'])['Final'].sum().unstack(fill_value=0)
    
    # 모든 주차(unique_weeks)가 컬럼에 포함되도록 재정렬 (없으면 0으로 채움)
    pivot = pivot.reindex(unique_weeks, axis=1, fill_value=0)
    
    # 각 팀별로 "HWK Total" 열 추가 (모든 주차의 합계)
    pivot["HWK Total"] = pivot.sum(axis=1)
    
    # 팀별 피벗 테이블에 모든 팀 평균 행("Average") 추가
    pivot_with_avg = pivot.copy()
    pivot_with_avg.loc["Average"] = pivot_with_avg.mean(axis=0)
    
    # [테이블] 결과 출력 (소수점 2자리)
    st.dataframe(pivot_with_avg.style.format(precision=2))
    
    # [차트] 팀별 점수를 시각화하기 위해 피벗 테이블(평균행 제외)을 long format으로 변환
    pivot_reset = pivot.reset_index()  # 'Team'이 컬럼이 되도록 변환
    df_melted = pd.melt(
        pivot_reset,
        id_vars="Team",
        value_vars=period_order,
        var_name="Period",
        value_name="Score"
    )
    
    # 각 Period(주차, HWK Total)의 전체 평균 계산
    avg_series = pivot.mean(axis=0)
    average_df = avg_series.reset_index()
    average_df.columns = ["Period", "Average_Score"]
    
    # Altair 차트 생성: 팀별 막대 차트 + 전체 평균 선 및 점 표시
    base = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Period:N', sort=period_order, title="주차 / HWK Total"),
        y=alt.Y('Score:Q', title="Final 점수"),
        color=alt.Color('Team:N', title="팀"),
        tooltip=['Team', 'Period', 'Score']
    )
    
    average_line = alt.Chart(average_df).mark_line(color='black', strokeWidth=3).encode(
        x=alt.X('Period:N', sort=period_order),
        y=alt.Y('Average_Score:Q', title="전체 평균")
    )
    
    average_points = alt.Chart(average_df).mark_point(color='black', size=100).encode(
        x=alt.X('Period:N', sort=period_order),
        y=alt.Y('Average_Score:Q')
    )
    
    chart = (base + average_line + average_points).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=False)

