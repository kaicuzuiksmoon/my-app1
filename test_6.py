import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import re

# ====================================================
# 언어 선택 (오른쪽 상단)
# ====================================================
col1, col2 = st.columns([3, 1])
with col2:
    language = st.selectbox("Language / 언어", options=["한글", "English"])

if language == "한글":
    t = {
        "common_filters": "공통 필터",
        "select_kpi": "KPI 선택 (목표)",
        "select_week": "주차 선택 (또는 'Total')",
        "select_team": "팀 선택",
        "max_score_info": "최대 점수 정보",
        "target": "목표",
        "best_score": "최고 점수",
        "no1_team": "1위 팀",
        "final_score_chart": "최종 점수 차트",
        "final_table": "최종 테이블",
        "actual_chart": "실적 차트",
        "actual_table": "실적 테이블",
        "rank_chart": "순위 차트",
        "rank_table": "순위 테이블",
        "radar_chart": "레이더 차트",
        "team_actual_table": "선택 팀의 실적 테이블",
        "overall_chart": "전체 KPI 최종 합계 차트",
        "overall_table": "전체 KPI 최종 합계 테이블 및 순위",
        "row12_final": "KPI별 주차별 최종 점수 및 HWK Total (팀별 점수와 전체 평균)",
        "row13_actual": "KPI별 주차별 실적 (Actual) 및 HWK Total (팀별 실적과 전체 평균)",
        "factory": "공장명",
        "team": "팀",
        "hwk_total": "HWK Total (평균)",
        "period": "주차 / HWK Total",
        "score": "점수",
        "overall_average": "전체 평균"
    }
else:
    t = {
        "common_filters": "Common Filters",
        "select_kpi": "Select a KPI (Target)",
        "select_week": "Select Week (or 'Total')",
        "select_team": "Select a Team",
        "max_score_info": "Max Score Info",
        "target": "Target",
        "best_score": "Best Score",
        "no1_team": "No. 1 Team",
        "final_score_chart": "Final Score Chart",
        "final_table": "Final Table",
        "actual_chart": "Actual Chart",
        "actual_table": "Actual Table",
        "rank_chart": "Rank Chart",
        "rank_table": "Rank Table",
        "radar_chart": "Radar Chart",
        "team_actual_table": "Team's Actual Table",
        "overall_chart": "Overall Summation Chart",
        "overall_table": "Overall Summation Table & Ranking",
        "row12_final": "KPI-wise Weekly Final Score & HWK Total (Team scores & Overall Average)",
        "row13_actual": "KPI-wise Weekly Actual & HWK Total (Team scores & Overall Average)",
        "factory": "Factory",
        "team": "Team",
        "hwk_total": "HWK Total (Average)",
        "period": "Week / HWK Total",
        "score": "Score",
        "overall_average": "Overall Average"
    }

# ====================================================
# 0) CSV 로드 및 전처리
# ====================================================
@st.cache_data
def load_data():
    """
    'score.csv' 파일을 읽어와 전처리:
      - 사용 열: Week, Team, KPI, Actual, Final
      - Actual에서 숫자, '.', '-' 이외 문자 제거
      - Actual, Final을 float으로 변환
      - Actual 또는 Final이 NaN인 행 제거
      - Week에서 숫자만 추출 (예: W1 → 1)하여 Week_num 생성
    """
    csv_path = "score.csv"  # 코드 파일과 동일한 위치에 있어야 함
    df = pd.read_csv(csv_path)

    df['Actual'] = df['Actual'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    df['Final'] = pd.to_numeric(df['Final'], errors='coerce')

    df.dropna(subset=['Actual', 'Final'], inplace=True)
    df['Week_num'] = df['Week'].str.extract(r'(\d+)').astype(int)

    return df

# ====================================================
# 1) 데이터 로드
# ====================================================
df = load_data()

# ====================================================
# 2) 변수 설정: KPI, Team, Week 등
# ====================================================
all_kpis = sorted(df['KPI'].dropna().unique())
all_teams = sorted(df['Team'].dropna().unique())
unique_weeks = sorted(df['Week'].unique(), key=lambda w: int(w[1:]))
week_options = unique_weeks + ["Total"]

# ====================================================
# 3) 사이드바 필터 (공통)
# ====================================================
st.sidebar.title(t["common_filters"])
selected_kpi = st.sidebar.selectbox(t["select_kpi"], options=all_kpis)
selected_week_or_total = st.sidebar.selectbox(t["select_week"], options=week_options)
selected_team = st.sidebar.selectbox(t["select_team"], options=all_teams)

# ====================================================
# 4) 피벗테이블 생성: Final (합계) & Actual (평균) for selected_kpi
# ====================================================
# Final 데이터 집계
df_kpi_final = df[df['KPI'] == selected_kpi].copy()
grouped_final = df_kpi_final.groupby(['Week', 'Team'], as_index=False)['Final'].sum()
pivot_final = grouped_final.pivot(index='Team', columns='Week', values='Final').fillna(0)
pivot_final['Total'] = pivot_final.sum(axis=1)

# Actual 데이터 집계 (평균 사용)
df_kpi_act = df[df['KPI'] == selected_kpi].copy()
grouped_act = df_kpi_act.groupby(['Week', 'Team'], as_index=False)['Actual'].mean()
pivot_act = grouped_act.pivot(index='Team', columns='Week', values='Actual').fillna(0)
pivot_act['Average'] = pivot_act.mean(axis=1)

# ====================================================
# Row1: 최대 점수 정보
# ====================================================
st.subheader(t["max_score_info"])
if pivot_final.empty:
    st.write("No Final data." if language == "English" else "최종 데이터 없음.")
else:
    max_total = pivot_final['Total'].max()
    max_team = pivot_final['Total'].idxmax()
    st.markdown(f"**{t['target']}** : {selected_kpi}  \n**{t['best_score']}** : {max_total:.2f}  \n**{t['no1_team']}** : {max_team}")

# ====================================================
# Row2: Final Score 차트
# ====================================================
st.subheader(t["final_score_chart"])
if pivot_final.empty or (selected_week_or_total not in pivot_final.columns):
    st.write("No data for this chart." if language == "English" else "차트 데이터 없음.")
else:
    bar_df_final = pivot_final[[selected_week_or_total]].reset_index()
    bar_df_final.columns = [t["team"], "Final_Value"]
    chart_final = alt.Chart(bar_df_final).mark_bar().encode(
        x=alt.X(f'{t["team"]}:N', sort=None),
        y=alt.Y('Final_Value:Q'),
        tooltip=[t["team"], 'Final_Value']
    ).properties(width=700, height=400)
    st.altair_chart(chart_final, use_container_width=False)

# ====================================================
# Row3: Final Table
# ====================================================
st.subheader(t["final_table"])
if pivot_final.empty:
    st.write("No Final data." if language == "English" else "최종 데이터 없음.")
else:
    st.dataframe(pivot_final.style.format(precision=2))

# ====================================================
# Row4: Actual Chart (선택 KPI, 선택 주차)
# ====================================================
st.subheader(t["actual_chart"])
actual_col = selected_week_or_total
if actual_col == "Total":
    actual_col = "Average"
if pivot_act.empty or (actual_col not in pivot_act.columns):
    st.write("No data for this chart." if language == "English" else "차트 데이터 없음.")
else:
    bar_df_act = pivot_act[[actual_col]].reset_index()
    bar_df_act.columns = [t["team"], "Actual_Value"]
    chart_act = alt.Chart(bar_df_act).mark_bar(color='orange').encode(
        x=alt.X(f'{t["team"]}:N', sort=None),
        y=alt.Y('Actual_Value:Q'),
        tooltip=[t["team"], 'Actual_Value']
    ).properties(width=700, height=400)
    st.altair_chart(chart_act, use_container_width=False)

# ====================================================
# Row5: Actual Table (선택 KPI)
# ====================================================
st.subheader(t["actual_table"])
if pivot_act.empty:
    st.write("No Actual data." if language == "English" else "실적 데이터 없음.")
else:
    st.dataframe(pivot_act.style.format(precision=2))

# ====================================================
# Row6: Rank Chart (선택한 팀 기준)
# ====================================================
st.subheader(t["rank_chart"])
df_rank = df.copy()
df_rank['Rank'] = df_rank.groupby(['Week', 'KPI'])['Final'].rank(method='dense', ascending=False)
df_team_rank = df_rank[df_rank['Team'] == selected_team].copy()
df_team_rank = df_team_rank.sort_values(['KPI', 'Week_num'])
if df_team_rank.empty:
    st.write("No Rank data for the selected team." if language == "English" else "선택한 팀의 순위 데이터 없음.")
else:
    line_chart = alt.Chart(df_team_rank).mark_line(point=True).encode(
        x=alt.X('Week_num:Q', axis=alt.Axis(title='Weeknum')),
        y=alt.Y('Rank:Q', sort='descending', axis=alt.Axis(title='Rank (1=best)')),
        color=alt.Color('KPI:N', legend=alt.Legend(title='KPI')),
        tooltip=['Week', 'KPI', 'Final', 'Rank']
    ).properties(width=700, height=400)
    st.altair_chart(line_chart, use_container_width=False)

# ====================================================
# Row7: Rank Table
# ====================================================
st.subheader(t["rank_table"])
if df_team_rank.empty:
    st.write("No Rank data." if language == "English" else "순위 데이터 없음.")
else:
    rank_pivot = df_team_rank.pivot(index='KPI', columns='Week', values='Rank')
    rank_pivot['Total'] = rank_pivot.sum(axis=1)
    st.dataframe(rank_pivot.fillna("").style.format(precision=2))

# ====================================================
# Row8: Radar Chart (선택한 팀 vs. 전체 최고)
# ====================================================
st.subheader(t["radar_chart"])
if df_team_rank.empty:
    st.write("No data for Radar Chart." if language == "English" else "레이더 차트 데이터 없음.")
else:
    df_final_only = df.copy()
    team_mean = df_final_only[df_final_only['Team'] == selected_team].groupby('KPI')['Final'].mean().rename('Team_Mean').reset_index()
    max_final = df_final_only.groupby('KPI')['Final'].max().rename('Max_Final').reset_index()
    radar_df = pd.merge(team_mean, max_final, on='KPI', how='outer').fillna(0)
    if radar_df.empty:
        st.write("No radar data." if language == "English" else "레이더 데이터 없음.")
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

# ====================================================
# Row9: 선택한 팀의 Actual Table
# ====================================================
st.subheader(t["team_actual_table"])
df_team_act = df[df['Team'] == selected_team].copy()
if df_team_act.empty:
    st.write("No Actual data for this team." if language == "English" else "선택한 팀의 실적 데이터 없음.")
else:
    team_act_pivot = df_team_act.pivot(index='KPI', columns='Week', values='Actual')
    st.dataframe(team_act_pivot.fillna("").style.format(precision=2))

# ====================================================
# Row10: Overall Summation Chart (모든 KPI)
# ====================================================
st.subheader(t["overall_chart"])
df_total_all_kpi = df.groupby(['Week', 'Team'], as_index=False)['Final'].sum()
if df_total_all_kpi.empty:
    st.write("No overall data." if language == "English" else "전체 데이터 없음.")
else:
    chart_overall = alt.Chart(df_total_all_kpi).mark_bar().encode(
        x=alt.X(f'{t["team"]}:N', sort='-y'),
        y=alt.Y('Final:Q', stack='zero'),
        color='Week:N',
        tooltip=[t["team"], 'Week', 'Final']
    ).properties(width=700, height=400)
    st.altair_chart(chart_overall, use_container_width=False)

# ====================================================
# Row11: Overall Summation Table & Ranking
# ====================================================
st.subheader(t["overall_table"])
if df_total_all_kpi.empty:
    st.write("No overall data." if language == "English" else "전체 데이터 없음.")
else:
    pivot_total_all = df_total_all_kpi.pivot(index='Team', columns='Week', values='Final').fillna(0)
    pivot_total_all['GrandTotal'] = pivot_total_all.sum(axis=1)
    st.write("Overall Summation Table (all KPI, all weeks)" if language == "English" else "전체 KPI (모든 주차) 최종 합계 테이블")
    st.dataframe(pivot_total_all.style.format(precision=2))
    ranking_df = pivot_total_all[['GrandTotal']].copy()
    ranking_df = ranking_df.sort_values('GrandTotal', ascending=False)
    ranking_df['Rank'] = ranking_df['GrandTotal'].rank(method='dense', ascending=False).astype(int)
    st.dataframe(ranking_df.style.format(precision=2))

st.write("---")
st.write("End of layout version 1 by HWK QIP." if language == "English" else "HWK QIP 버전 1 종료.")

# ====================================================
# Row12: KPI별 주차별 최종 점수 (Final) & HWK Total (팀별 점수와 전체 평균)
# ====================================================
st.subheader(t["row12_final"])
period_order = unique_weeks + ["HWK Total"]
for kpi in all_kpis:
    st.markdown(f"### KPI: {kpi}")
    df_kpi = df[df['KPI'] == kpi].copy()
    if df_kpi.empty:
        st.write("해당 KPI에 대한 데이터가 없습니다." if language == "한글" else "No data for this KPI.")
        continue
    pivot = df_kpi.groupby(['Team', 'Week'])['Final'].sum().unstack(fill_value=0)
    pivot = pivot.reindex(unique_weeks, axis=1, fill_value=0)
    pivot["HWK Total"] = pivot.sum(axis=1)
    pivot_with_avg = pivot.copy()
    pivot_with_avg.loc["Average"] = pivot_with_avg.mean(axis=0)
    st.dataframe(pivot_with_avg.style.format(precision=2))
    pivot_reset = pivot.reset_index()
    df_melted = pd.melt(pivot_reset, id_vars="Team", value_vars=period_order, var_name=t["period"], value_name=t["score"])
    avg_series = pivot.mean(axis=0)
    average_df = avg_series.reset_index()
    average_df.columns = [t["period"], "Average_Score"]
    base = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X(f'{t["period"]}:N', sort=period_order, title=t["period"]),
        y=alt.Y(f'{t["score"]}:Q', title=t["score"]),
        color=alt.Color('Team:N', title=t["team"]),
        tooltip=[t["team"], f'{t["period"]}', f'{t["score"]}']
    )
    average_line = alt.Chart(average_df).mark_line(color='black', strokeWidth=3).encode(
        x=alt.X(f'{t["period"]}:N', sort=period_order),
        y=alt.Y("Average_Score:Q", title=t["overall_average"])
    )
    average_points = alt.Chart(average_df).mark_point(color='black', size=100).encode(
        x=alt.X(f'{t["period"]}:N', sort=period_order),
        y=alt.Y("Average_Score:Q")
    )
    chart = (base + average_line + average_points).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=False)

# ====================================================
# Row13: KPI별 주차별 실적 (Actual) & HWK Total (팀별 실적과 전체 평균)
# ====================================================
st.subheader(t["row13_actual"])
for kpi in all_kpis:
    st.markdown(f"### KPI: {kpi}")
    df_kpi = df[df['KPI'] == kpi].copy()
    if df_kpi.empty:
        st.write("해당 KPI에 대한 데이터가 없습니다." if language == "한글" else "No data for this KPI.")
        continue
    # 팀별, 주차별 Actual 합계 집계
    pivot_act2 = df_kpi.groupby(['Team', 'Week'])['Actual'].sum().unstack(fill_value=0)
    pivot_act2 = pivot_act2.reindex(unique_weeks, axis=1, fill_value=0)
    # ★ HWK Total은 합계(sum)가 아니라 평균(mean)으로 계산
    pivot_act2["HWK Total"] = pivot_act2.mean(axis=1)
    pivot_act2 = pivot_act2.reset_index()
    # 공장명(Factory) 칼럼 추가: 팀명에 따라 매핑
    def get_factory(team):
        m = re.search(r'\d+', team)
        if m:
            num = int(m.group())
            if num in [1, 2]:
                return "Building A"
            elif num == 3:
                return "Building B"
            elif num in [4, 5]:
                return "Building C"
            elif num in [6, 7]:
                return "Building D"
        return ""
    pivot_act2["Factory"] = pivot_act2["Team"].apply(get_factory)
    # 칼럼 순서 재정렬: Factory, Team, 그리고 나머지
    cols = pivot_act2.columns.tolist()
    if "Factory" in cols:
        cols.remove("Factory")
    new_cols = ["Factory", "Team"] + [col for col in cols if col not in ["Factory", "Team"]]
    pivot_act2 = pivot_act2[new_cols]
    # 평균 행 추가 (숫자형 칼럼에 대해)
    numeric_cols = [col for col in pivot_act2.columns if col not in ["Factory", "Team"]]
    avg_row = {col: pivot_act2[col].mean() for col in numeric_cols}
    avg_row["Factory"] = ""
    avg_row["Team"] = "Average" if language == "English" else "평균"
    # (pandas 1.4 이전 버전에서는 .append() 사용, 이후 버전에서는 pd.concat() 권장)
    pivot_act2_with_avg = pivot_act2.append(avg_row, ignore_index=True)
    st.dataframe(pivot_act2_with_avg.style.format(precision=2))
    
    # --- 차트 생성 ---
    # 차트 데이터에서는 "평균" 행은 제외
    chart_data = pivot_act2[pivot_act2["Team"] != ("Average" if language == "English" else "평균")].copy()
    period_columns = unique_weeks + ["HWK Total"]
    chart_data_melted = pd.melt(chart_data, id_vars=["Team"], value_vars=period_columns, var_name=t["period"], value_name=t["score"])
    avg_series_act = chart_data[period_columns].mean()
    average_df_act = avg_series_act.reset_index()
    average_df_act.columns = [t["period"], "Average_Score"]
    base_act = alt.Chart(chart_data_melted).mark_bar().encode(
        x=alt.X(f'{t["period"]}:N', sort=period_order, title=t["period"]),
        y=alt.Y(f'{t["score"]}:Q', title=t["score"]),
        color=alt.Color('Team:N', title=t["team"]),
        tooltip=[t["team"], f'{t["period"]}', f'{t["score"]}']
    )
    average_line_act = alt.Chart(average_df_act).mark_line(color='black', strokeWidth=3).encode(
        x=alt.X(f'{t["period"]}:N', sort=period_order),
        y=alt.Y("Average_Score:Q", title=t["overall_average"])
    )
    average_points_act = alt.Chart(average_df_act).mark_point(color='black', size=100).encode(
        x=alt.X(f'{t["period"]}:N', sort=period_order),
        y=alt.Y("Average_Score:Q")
    )
    chart_act2 = (base_act + average_line_act + average_points_act).properties(width=700, height=400)
    st.altair_chart(chart_act2, use_container_width=False)
