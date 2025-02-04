import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import os

# 페이지 설정: 타이틀 및 와이드 레이아웃
st.set_page_config(page_title="KPI Dashboard", layout="wide")

# ------------------------------------------
# 데이터 로드 및 전처리 함수
# ------------------------------------------
@st.cache_data
def load_data():
    # score.csv 파일은 현재 작업 디렉토리에 있어야 합니다.
    df = pd.read_csv("score.csv")
    return df

def convert_to_numeric(x):
    """
    문자열에서 숫자만 추출하여 float로 변환합니다.
    예) "100.86 minutes" -> 100.86, "2.38%" -> 2.38, "$278.87" -> 278.87
    '-' 또는 빈 값은 NaN으로 처리합니다.
    """
    try:
        if isinstance(x, str):
            if x.strip() == '-' or x.strip() == '':
                return np.nan
            # 정규식을 사용하여 숫자, 소수점, 음수 부호만 남깁니다.
            num = re.sub(r'[^\d\.-]', '', x)
            return float(num)
        else:
            return x
    except:
        return np.nan

# 데이터 로드
df = load_data()

# Week 컬럼 전처리: "W1", "W2", ... 에서 숫자만 추출하여 Week_num 컬럼 생성
df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if isinstance(x, str) and re.sub(r'\D', '', x) != '' else np.nan)

# ------------------------------------------
# 사이드바: 인터랙티브 필터 설정
# ------------------------------------------
st.sidebar.header("Filter Options")

# 1. KPI 선택 (예: issue_tracking, AQL_performance 등)
kpi_list = sorted(df["KPI"].unique())
selected_kpi = st.sidebar.selectbox("Select KPI", kpi_list, index=0)

# 2. 팀 선택 (비교용 다중 선택)
team_list = sorted(df["Team"].unique())
selected_teams = st.sidebar.multiselect("Select Teams for Comparison", team_list, default=team_list)

# 3. 주(Week) 범위 선택: Week_num이 숫자형이므로 슬라이더 사용
week_min = int(df["Week_num"].min())
week_max = int(df["Week_num"].max())
selected_week_range = st.sidebar.slider("Select Week Range", min_value=week_min, max_value=week_max, value=(week_min, week_max))

# 4. 팀 상세 조회용: 단일 팀 선택
selected_team_detail = st.sidebar.selectbox("Select Team for Details", team_list)

# ------------------------------------------
# 필터 적용: KPI, 팀, 주 범위에 따라 데이터 서브셋 생성
# ------------------------------------------
# KPI 비교, 트렌드, 랭킹에 사용할 데이터 (long format)
df_filtered = df[
    (df["KPI"] == selected_kpi) &
    (df["Team"].isin(selected_teams)) &
    (df["Week_num"] >= selected_week_range[0]) &
    (df["Week_num"] <= selected_week_range[1])
].copy()

# Actual 값을 numeric으로 변환
df_filtered["Actual_numeric"] = df_filtered["Actual"].apply(convert_to_numeric)

# 트렌드 분석용 데이터도 같은 조건으로 준비 (정렬 처리)
df_trend = df_filtered.sort_values("Week_num")

# ------------------------------------------
# 메인 대시보드 레이아웃
# ------------------------------------------
st.title("KPI Dashboard")

# -----------------------------------------------------
# 1. 특정 KPI별 팀 성과 비교 (바 차트)
# -----------------------------------------------------
st.markdown("### 1. KPI Performance Comparison by Team")
# 팀별 평균 Actual 값을 계산 (선택된 주 범위 내)
grouped = df_filtered.groupby("Team")["Actual_numeric"].mean().reset_index()
fig_bar = px.bar(
    grouped,
    x="Team",
    y="Actual_numeric",
    labels={"Actual_numeric": f"Average {selected_kpi} Value"},
    title=f"Average {selected_kpi} by Team"
)
st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# -----------------------------------------------------
# 2. 주간 성과 트렌드 분석 (라인 차트)
# -----------------------------------------------------
st.markdown("### 2. Weekly Performance Trend Analysis")
fig_line = px.line(
    df_trend,
    x="Week_num",
    y="Actual_numeric",
    color="Team",
    markers=True,
    labels={"Week_num": "Week", "Actual_numeric": f"{selected_kpi} Value"},
    title=f"Weekly Trend of {selected_kpi}"
)
st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

# -----------------------------------------------------
# 3. KPI별 상위/하위 팀 랭킹 (수평 바 차트)
# -----------------------------------------------------
st.markdown("### 3. KPI Top/Bottom Team Rankings")
# 팀별 평균 Actual 값을 다시 계산 (NaN 제외)
ranking = df_filtered.groupby("Team")["Actual_numeric"].mean().reset_index().dropna()
ranking_sorted = ranking.sort_values("Actual_numeric", ascending=False)

# 상위, 하위 팀 개수 결정 (최대 5팀)
top_n = 5 if len(ranking_sorted) >= 5 else len(ranking_sorted)
bottom_n = 5 if len(ranking_sorted) >= 5 else len(ranking_sorted)
top_df = ranking_sorted.head(top_n)
bottom_df = ranking_sorted.tail(bottom_n).sort_values("Actual_numeric", ascending=True)

# 두 컬럼으로 나누어 차트 표시
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Top {top_n} Teams - {selected_kpi}")
    fig_top = px.bar(
        top_df,
        x="Actual_numeric",
        y="Team",
        orientation="h",
        text="Actual_numeric",
        labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
    )
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True, key="top_chart")
with col2:
    st.subheader(f"Bottom {bottom_n} Teams - {selected_kpi}")
    fig_bottom = px.bar(
        bottom_df,
        x="Actual_numeric",
        y="Team",
        orientation="h",
        text="Actual_numeric",
        labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
    )
    fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# -----------------------------------------------------
# 4. 특정 팀의 KPI 상세 조회 기능
# -----------------------------------------------------
st.markdown("### 4. Team-Specific KPI Detailed View")
# 선택된 팀에 대한 데이터 필터링
df_team = df[df["Team"] == selected_team_detail].copy()
df_team["Actual_numeric"] = df_team["Actual"].apply(convert_to_numeric)
df_team["Week_num"] = df_team["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if isinstance(x, str) and re.sub(r'\D', '', x) != '' else np.nan)

# 선택된 팀의 최신 주(Week)를 파악
latest_week = int(df_team["Week_num"].max())
latest_data = df_team[df_team["Week_num"] == latest_week]
# 이전 주 데이터 (delta 계산용)
previous_data = df_team[df_team["Week_num"] == (latest_week - 1)]

st.subheader(f"Details for {selected_team_detail} (Week {latest_week})")
# KPI별 최신 값과 전주 대비 변화(Delta)를 st.metric으로 표시
cols = st.columns(3)  # 한 행에 3개씩 표시
i = 0
for index, row in latest_data.iterrows():
    kpi_name = row["KPI"]
    current_value = row["Actual_numeric"]
    # 동일 KPI의 이전 주 값 찾기
    prev_row = previous_data[previous_data["KPI"] == kpi_name]
    if not prev_row.empty:
        previous_value = prev_row.iloc[0]["Actual_numeric"]
        delta = current_value - previous_value if pd.notnull(current_value) and pd.notnull(previous_value) else None
    else:
        delta = None
    # st.metric에 표시 (소수점 2자리)
    cols[i % 3].metric(
        label=kpi_name,
        value=f"{current_value:.2f}" if pd.notnull(current_value) else "N/A",
        delta=f"{delta:+.2f}" if delta is not None else "N/A"
    )
    i += 1
    if i % 3 == 0:
        cols = st.columns(3)

# 선택된 팀에 대한 전체 상세 데이터를 테이블 형태로 표시
st.markdown("#### Detailed Data for Selected Team")
st.dataframe(df_team.sort_values("Week_num"))

