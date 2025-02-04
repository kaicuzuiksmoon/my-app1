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

# ----------------------------------------------------
# 다국어 텍스트 사전
# ----------------------------------------------------
if language == "한글":
    t = {
        "common_filters": "공통 필터",
        "select_display": "점수/실적 선택",
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
        "select_display": "Select Display Type (Score/Actual)",
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
# 1) 데이터 로드 및 기본 변수 설정
# ====================================================
df = load_data()

all_kpis = sorted(df['KPI'].dropna().unique())
all_teams = sorted(df['Team'].dropna().unique())
unique_weeks = sorted(df['Week'].unique(), key=lambda w: int(w[1:]))
week_options = unique_weeks + ["Total"]

# 팀 선택 옵션에 "Total" (한글이면 "전체") 추가
if language == "한글":
    team_options = all_teams + ["전체"]
else:
    team_options = all_teams + ["Total"]

# ====================================================
# 2) 사이드바 필터 (공통)
# ====================================================
st.sidebar.title(t["common_filters"])

# "점수/실적 선택" 추가
if language == "한글":
    display_options = ["점수", "실적"]
else:
    display_options = ["Score", "Actual"]
display_type = st.sidebar.selectbox(t["select_display"], options=display_options)

selected_kpi = st.sidebar.selectbox(t["select_kpi"], options=all_kpis)
selected_week_or_total = st.sidebar.selectbox(t["select_week"], options=week_options)
selected_team = st.sidebar.selectbox(t["select_team"], options=team_options)

# ====================================================
# 3) Final 및 Actual 데이터 피벗 (KPI별 선택)
# ====================================================
# Final 데이터 집계 (최종 점수)
df_kpi_final = df[df['KPI'] == selected_kpi].copy()
grouped_final = df_kpi_final.groupby(['Week', 'Team'], as_index=False)['Final'].sum()
pivot_final = grouped_final.pivot(index='Team', columns='Week', values='Final').fillna(0)
pivot_final['Total'] = pivot_final.sum(axis=1)

# Actual 데이터 집계 (실적 – 평균 사용)
df_kpi_act = df[df['KPI'] == selected_kpi].copy()
grouped_act = df_kpi_act.groupby(['Week', 'Team'], as_index=False)['Actual'].mean()
pivot_act = grouped_act.pivot(index='Team', columns='Week', values='Actual').fillna(0)
pivot_act['Average'] = pivot_act.mean(axis=1)

# ====================================================
# 4) 선택한 표시 항목에 따라 필요한 차트/테이블만 출력
# ====================================================
if display_type in ["점수", "Score"]:
    # --------------------------
    # [점수 관련 화면]
    # --------------------------
    # Row1: 최대 점수 정보 (Final)
    st.subheader(t["max_score_info"])
    if pivot_final.empty:
        st.write("No Final data." if language=="English" else "최종 데이터 없음.")
    else:
        max_total = pivot_final['Total'].max()
        max_team = pivot_final['Total'].idxmax()
        st.markdown(f"**{t['target']}**: {selected_kpi}  \n**{t['best_score']}**: {max_total:.2f}  \n**{t['no1_team']}**: {max_team}")

    # Row2: 최종 점수 차트
    st.subheader(t["final_score_chart"])
    if pivot_final.empty or (selected_week_or_total not in pivot_final.columns):
        st.write("No data for this chart." if language=="English" else "차트 데이터 없음.")
    else:
        bar_df_final = pivot_final[[selected_week_or_total]].reset_index()
        bar_df_final.columns = [t["team"], "Final_Value"]
        chart_final = alt.Chart(bar_df_final).mark_bar().encode(
            x=alt.X(f'{t["team"]}:N', sort=None),
            y=alt.Y('Final_Value:Q'),
            tooltip=[t["team"], 'Final_Value']
        ).properties(width=700, height=400)
        st.altair_chart(chart_final, use_container_width=False)

    # Row3: 최종 테이블
    st.subheader(t["final_table"])
    if pivot_final.empty:
        st.write("No Final data." if language=="English" else "최종 데이터 없음.")
    else:
        st.dataframe(pivot_final.style.format(precision=2))

    # Row12: KPI별 주차별 최종 점수 및 HWK Total (팀별 점수와 전체 평균)
    st.subheader(t["row12_final"])
    # period_order에서 오른쪽 끝 컬럼은 사전에서 정의한 HWK Total (Average)로 사용
    period_order = unique_weeks + [t["hwk_total"]]
    for kpi in all_kpis:
        st.markdown(f"### KPI: {kpi}")
        df_kpi = df[df['KPI'] == kpi].copy()
        if df_kpi.empty:
            st.write("No data for this KPI." if language=="English" else "해당 KPI에 대한 데이터가 없습니다.")
            continue
        # 기존에는 합계(sum)를 사용했으나, 여기서는 평균(mean)으로 계산
        pivot = df_kpi.groupby(['Team', 'Week'])['Final'].sum().unstack(fill_value=0)
        pivot = pivot.reindex(unique_weeks, axis=1, fill_value=0)
        pivot[t["hwk_total"]] = pivot.mean(axis=1)  # ← 합계 대신 평균으로 계산
        pivot_with_avg = pivot.copy()
        pivot_with_avg.loc["Average"] = pivot_with_avg.mean(axis=0)
        st.dataframe(pivot_with_avg.style.format(precision=2))
        
        pivot_reset = pivot.reset_index()
        df_melted = pd.melt(pivot_reset, id_vars="Team", value_vars=period_order,
                            var_name=t["period"], value_name=t["score"])
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

else:
    # --------------------------
    # [실적 관련 화면]
    # --------------------------
    # Row4: 실적 차트
    st.subheader(t["actual_chart"])
    actual_col = selected_week_or_total
    if actual_col == "Total":
        actual_col = "Average"
    if pivot_act.empty or (actual_col not in pivot_act.columns):
        st.write("No data for this chart." if language=="English" else "차트 데이터 없음.")
    else:
        bar_df_act = pivot_act[[actual_col]].reset_index()
        bar_df_act.columns = [t["team"], "Actual_Value"]
        chart_act = alt.Chart(bar_df_act).mark_bar(color='orange').encode(
            x=alt.X(f'{t["team"]}:N', sort=None),
            y=alt.Y('Actual_Value:Q'),
            tooltip=[t["team"], 'Actual_Value']
        ).properties(width=700, height=400)
        st.altair_chart(chart_act, use_container_width=False)

    # Row5: 실적 테이블
    st.subheader(t["actual_table"])
    if pivot_act.empty:
        st.write("No Actual data." if language=="English" else "실적 데이터 없음.")
    else:
        st.dataframe(pivot_act.style.format(precision=2))

    # Row13: KPI별 주차별 실적 (Actual) & HWK Total (팀별 실적과 전체 평균)
    st.subheader(t["row13_actual"])
    for kpi in all_kpis:
        st.markdown(f"### KPI: {kpi}")
        df_kpi = df[df['KPI'] == kpi].copy()
        if df_kpi.empty:
            st.write("No data for this KPI." if language=="English" else "해당 KPI에 대한 데이터가 없습니다.")
            continue
        # 팀별, 주차별 Actual 합계 집계
        pivot_act2 = df_kpi.groupby(['Team', 'Week'])['Actual'].sum().unstack(fill_value=0)
        pivot_act2 = pivot_act2.reindex(unique_weeks, axis=1, fill_value=0)
        # ★ HWK Total은 합계(sum)가 아니라 평균(mean)으로 계산 (실적 데이터에도 동일하게 적용)
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
        avg_row["Team"] = "Average" if language=="English" else "평균"
        pivot_act2_with_avg = pd.concat([pivot_act2, pd.DataFrame([avg_row])], ignore_index=True)
        st.dataframe(pivot_act2_with_avg.style.format(precision=2))
        
        # --- 차트 생성 ---
        chart_data = pivot_act2[pivot_act2["Team"] != ("Average" if language=="English" else "평균")].copy()
        period_columns = unique_weeks + ["HWK Total"]
        chart_data_melted = pd.melt(chart_data, id_vars=["Team"], value_vars=period_columns,
                                     var_name=t["period"], value_name=t["score"])
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
