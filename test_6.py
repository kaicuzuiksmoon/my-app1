import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import os

# --------------------------------------------------
# 1. 페이지 설정 및 다국어 번역용 사전 정의
# --------------------------------------------------
st.set_page_config(page_title="HWK Quality competition Event", layout="wide")

# 번역 사전 (영어/한글)
trans = {
    "title": {
        "en": "HWK Quality competition Event",
        "ko": "HWK 품질 경쟁 이벤트"
    },
    "kpi_comparison": {
        "en": "1. KPI Performance Comparison by Team",
        "ko": "1. 팀별 KPI 성과 비교"
    },
    "weekly_trend": {
        "en": "2. Weekly Performance Trend Analysis",
        "ko": "2. 주간 성과 트렌드 분석"
    },
    "top_bottom_rankings": {
        "en": "3. KPI Top/Bottom Team Rankings",
        "ko": "3. KPI 상위/하위 팀 순위"
    },
    "last_week_details": {
        "en": "Last Week performance Details for {team} (Week {week})",
        "ko": "지난주 성과 상세보기: {team} (Week {week})"
    },
    "total_week_details": {
        "en": "Total Week Performance Detail for {team} (All weeks)",
        "ko": "전체 주차 누적 실적 상세: {team} (All weeks)"
    },
    "detailed_data": {
        "en": "Detailed Data for Selected Team",
        "ko": "선택된 팀의 상세 데이터"
    },
    "select_kpi": {
        "en": "Select KPI",
        "ko": "KPI 선택"
    },
    "select_teams": {
        "en": "Select Teams for Comparison",
        "ko": "비교할 팀 선택 (HWK Total 포함)"
    },
    "select_team_details": {
        "en": "Select Team for Details",
        "ko": "상세 조회할 팀 선택 (HWK Total 포함)"
    },
    "select_week_range": {
        "en": "Select Week Range",
        "ko": "주차 범위 선택"
    },
    "language": {
        "en": "Language",
        "ko": "언어"
    },
    "avg_by_team": {
        "en": "Average {kpi} by Team",
        "ko": "팀별 {kpi} 평균"
    },
    "weekly_trend_title": {
        "en": "Weekly Trend of {kpi}",
        "ko": "{kpi} 주간 추이"
    },
    "top_teams": {
        "en": "Top {n} Teams - {kpi}",
        "ko": "{kpi} 상위 {n} 팀"
    },
    "bottom_teams": {
        "en": "Bottom {n} Teams - {kpi}",
        "ko": "{kpi} 하위 {n} 팀"
    },
    "week_col": {
        "en": "Week {week}",
        "ko": "{week}주차"
    },
    "average": {
        "en": "Average",
        "ko": "평균"
    }
}

# --------------------------------------------------
# 2. 우측 상단 언어 선택 (영어/한글)
# --------------------------------------------------
col_title, col_lang = st.columns([4, 1])
with col_lang:
    lang = st.radio("Language / 언어", options=["en", "ko"], index=0, horizontal=True)

# 제목 출력 (선택한 언어)
st.title(trans["title"][lang])

# --------------------------------------------------
# 3. 유틸리티 함수 정의
# --------------------------------------------------
@st.cache_data
def load_data():
    # score.csv 파일은 이 코드와 동일한 디렉토리에 있어야 합니다.
    df = pd.read_csv("score.csv")
    return df

def convert_to_numeric(x):
    try:
        if isinstance(x, str):
            if x.strip() == '-' or x.strip() == '':
                return np.nan
            num = re.sub(r'[^\d\.-]', '', x)
            return float(num)
        else:
            return x
    except:
        return np.nan

def extract_unit(s):
    """
    문자열 s의 끝부분에 있는 숫자가 아닌 문자를 unit으로 추출
    예: "4.26%" -> "%", "100.86 minutes" -> " minutes"
    """
    if isinstance(s, str):
        m = re.search(r'([^\d\.\-]+)$', s.strip())
        return m.group(1).strip() if m else ""
    else:
        return ""

def format_label(row):
    """
    한 행(row)의 Actual_numeric 값과 Final 값을 소수점 2자리로 포맷하여
    "value{unit} (Final point)" 형식의 문자열로 반환
    """
    unit = extract_unit(row["Actual"]) if pd.notnull(row["Actual"]) else ""
    return f"{row['Actual_numeric']:.2f}{unit} ({row['Final']} point)"

def cumulative_performance(sub_df, kpi):
    """
    누적 실적 계산:
      - KPI가 "shortage_cost" (대소문자 무시)인 경우: Actual_numeric의 합계
      - 그 외: Actual_numeric의 평균
    """
    if kpi.lower() == "shortage_cost":
        return sub_df["Actual_numeric"].sum()
    else:
        return sub_df["Actual_numeric"].mean()

# KPI별 "좋은 값" 기준 (True이면 값이 클수록 좋은 것)
better_when_higher = {
    "5 prs validation": True,
    "6S_audit": True,
    "AQL_performance": False,
    "B-grade": True,
    "attendance": True,
    "issue_tracking": True,
    "shortage_cost": True
}

# --------------------------------------------------
# 4. 데이터 로드 및 전처리
# --------------------------------------------------
df = load_data()
# Week 컬럼에서 숫자만 추출하여 Week_num 컬럼 생성 (예: "W1" -> 1)
df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if isinstance(x, str) and re.sub(r'\D', '', x) != '' else np.nan)
# Actual 값을 numeric으로 변환
df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)

# --------------------------------------------------
# 5. 사이드바 위젯 (필터)
# --------------------------------------------------
st.sidebar.header("Filter Options")
selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=sorted(df["KPI"].unique()))
# 팀 목록에 "HWK Total" 추가 (전체 팀 평균용)
team_list = sorted(df["Team"].unique())
team_list_extended = team_list.copy()
if "HWK Total" not in team_list_extended:
    team_list_extended.append("HWK Total")
selected_teams = st.sidebar.multiselect(trans["select_teams"][lang], options=team_list_extended, default=team_list)
selected_week_range = st.sidebar.slider(
    trans["select_week_range"][lang],
    int(df["Week_num"].min()),
    int(df["Week_num"].max()),
    (int(df["Week_num"].min()), int(df["Week_num"].max()))
)
selected_team_detail = st.sidebar.selectbox(trans["select_team_details"][lang], options=team_list_extended, index=0)

# --------------------------------------------------
# 6. 데이터 필터링 (KPI, 주차 범위 적용)
# --------------------------------------------------
df_filtered = df[(df["KPI"] == selected_kpi) & 
                 (df["Week_num"] >= selected_week_range[0]) & 
                 (df["Week_num"] <= selected_week_range[1])].copy()

# 최신주 결정 (필터된 데이터 중 최대 주차)
if not df_filtered.empty:
    latest_week = int(df_filtered["Week_num"].max())
else:
    latest_week = None

# 최신주 데이터 (df_latest): 팀별 데이터만 포함 (추후 HWK Total 추가)
df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()

# 만약 비교용 팀에 "HWK Total"이 포함되어 있다면, 전체 팀(팀 필터 미적용) 데이터로 평균 행 생성
if "HWK Total" in selected_teams:
    df_overall = df_filtered[df_filtered["Week_num"] == latest_week].copy()
    if not df_overall.empty:
        overall_actual = df_overall["Actual_numeric"].mean()
        overall_final = round(df_overall["Final"].mean())
        overall_unit = extract_unit(df_overall.iloc[0]["Actual"])
        df_total = pd.DataFrame({
            "Team": ["HWK Total"],
            "Actual_numeric": [overall_actual],
            "Final": [overall_final],
            "Actual": [f"{overall_actual:.2f}{overall_unit}"],
            "Week_num": [latest_week],
            "KPI": [selected_kpi]
        })
        df_latest = pd.concat([df_latest, df_total], ignore_index=True)

# 비교 차트에 사용할 데이터: 선택한 팀만 추출
df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
df_comp["Label"] = df_comp.apply(format_label, axis=1)

# --------------------------------------------------
# 7. [1] KPI Performance Comparison by Team (바 차트)
# --------------------------------------------------
st.markdown(trans["kpi_comparison"][lang])
fig_bar = px.bar(
    df_comp,
    x="Team",
    y="Actual_numeric",
    text="Label",
    labels={"Actual_numeric": trans["avg_by_team"][lang].format(kpi=selected_kpi)}
)
fig_bar.update_traces(texttemplate="%{text}", textposition='outside')
st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# --------------------------------------------------
# 8. [2] Weekly Performance Trend Analysis (라인 차트)
# --------------------------------------------------
st.markdown(trans["weekly_trend"][lang])
# 개별 팀(“HWK Total” 제외) 데이터
df_trend_individual = df_filtered[df_filtered["Team"].isin([t for t in selected_teams if t != "HWK Total"])].copy()
fig_line = px.line(
    df_trend_individual,
    x="Week_num",
    y="Actual_numeric",
    color="Team",
    markers=True,
    labels={"Week_num": "Week", "Actual_numeric": f"{selected_kpi} Value"},
    title=trans["weekly_trend_title"][lang].format(kpi=selected_kpi)
)
# x축을 정수만 표시하도록 (tick 간격 1)
fig_line.update_xaxes(tickmode='linear', dtick=1)
# HWK Total이 선택되면, 전체 팀 평균 per week를 검은색 점선으로 추가
if "HWK Total" in selected_teams:
    df_overall_trend = df_filtered.groupby("Week_num").agg({"Actual_numeric": "mean", "Final": "mean"}).reset_index()
    fig_line.add_scatter(
        x=df_overall_trend["Week_num"],
        y=df_overall_trend["Actual_numeric"],
        mode='lines+markers',
        name="HWK Total",
        line=dict(color='black', dash='dash')
    )
st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

# --------------------------------------------------
# 9. [3] KPI Top/Bottom Team Rankings (Top 3 / Bottom 3)
# --------------------------------------------------
st.markdown(trans["top_bottom_rankings"][lang])
# 순위 선정 시, "HWK Total"은 제외
df_rank = df_comp[df_comp["Team"] != "HWK Total"].copy()
# KPI별 좋은 값 기준에 따라 정렬 (AQL_performance는 낮은 값이 좋음)
if better_when_higher.get(selected_kpi, True):
    df_rank_sorted_top = df_rank.sort_values("Actual_numeric", ascending=False)
    df_rank_sorted_bottom = df_rank.sort_values("Actual_numeric", ascending=True)
else:
    df_rank_sorted_top = df_rank.sort_values("Actual_numeric", ascending=True)
    df_rank_sorted_bottom = df_rank.sort_values("Actual_numeric", ascending=False)
top_n = 3 if len(df_rank_sorted_top) >= 3 else len(df_rank_sorted_top)
bottom_n = 3 if len(df_rank_sorted_bottom) >= 3 else len(df_rank_sorted_bottom)
top_df = df_rank_sorted_top.head(top_n).copy()
bottom_df = df_rank_sorted_bottom.head(bottom_n).copy()
top_df["Label"] = top_df.apply(format_label, axis=1)
bottom_df["Label"] = bottom_df.apply(format_label, axis=1)
col1, col2 = st.columns(2)
with col1:
    st.subheader(trans["top_teams"][lang].format(n=top_n, kpi=selected_kpi))
    fig_top = px.bar(
        top_df,
        x="Actual_numeric",
        y="Team",
        orientation="h",
        text="Label",
        labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
    )
    fig_top.update_traces(texttemplate="%{text}", textposition='outside')
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True, key="top_chart")
with col2:
    st.subheader(trans["bottom_teams"][lang].format(n=bottom_n, kpi=selected_kpi))
    fig_bottom = px.bar(
        bottom_df,
        x="Actual_numeric",
        y="Team",
        orientation="h",
        text="Label",
        labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
    )
    # 하위 차트는 빨간색으로 표기
    fig_bottom.update_traces(texttemplate="%{text}", textposition='outside', marker_color='red')
    fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# --------------------------------------------------
# 10. [4] Team-Specific KPI Detailed View
# --------------------------------------------------
st.markdown("")

# --- (A) Last Week performance Details ---
st.markdown(trans["last_week_details"][lang].format(team=selected_team_detail, week=latest_week))
if selected_team_detail != "HWK Total":
    # 개별 팀인 경우: 해당 팀의 selected_kpi 최신주와 전주 데이터
    df_detail_last = df[(df["KPI"] == selected_kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == latest_week)].copy()
    df_detail_prev = df[(df["KPI"] == selected_kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == (latest_week - 1))].copy()
    if not df_detail_last.empty:
        row = df_detail_last.iloc[0]
        current_unit = extract_unit(row["Actual"])
        current_value = row["Actual_numeric"]
        current_final = row["Final"]
        current_label = f"{current_value:.2f}{current_unit} ({current_final} point)"
        if not df_detail_prev.empty:
            prev_row = df_detail_prev.iloc[0]
            delta_actual = current_value - prev_row["Actual_numeric"]
            delta_final = current_final - prev_row["Final"]
            arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
            delta_str = f"{arrow}{delta_actual:+.2f}{current_unit} ({delta_final:+d} point)"
        else:
            delta_str = ""
        st.markdown(f"**{selected_kpi}**: {current_label}")
        if delta_str:
            st.markdown(f"<span style='font-size:90%; color:gray;'>{delta_str}</span>", unsafe_allow_html=True)
else:
    # HWK Total인 경우: 전체 팀의 최신주와 전주의 평균 데이터 (선택 KPI)
    df_overall_last = df[(df["KPI"] == selected_kpi) & (df["Week_num"] == latest_week)].copy()
    df_overall_prev = df[(df["KPI"] == selected_kpi) & (df["Week_num"] == (latest_week - 1))].copy()
    if not df_overall_last.empty:
        current_value = df_overall_last["Actual_numeric"].mean()
        current_final = round(df_overall_last["Final"].mean())
        # unit은 첫 행의 것을 사용
        current_unit = extract_unit(df_overall_last.iloc[0]["Actual"])
        current_label = f"{current_value:.2f}{current_unit} ({current_final} point)"
        if not df_overall_prev.empty:
            prev_value = df_overall_prev["Actual_numeric"].mean()
            prev_final = round(df_overall_prev["Final"].mean())
            delta_actual = current_value - prev_value
            delta_final = current_final - prev_final
            arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
            delta_str = f"{arrow}{delta_actual:+.2f}{current_unit} ({delta_final:+d} point)"
        else:
            delta_str = ""
        st.markdown(f"**{selected_kpi}**: {current_label}")
        if delta_str:
            st.markdown(f"<span style='font-size:90%; color:gray;'>{delta_str}</span>", unsafe_allow_html=True)

# --- (B) Total Week Performance Detail (누적 실적) ---
st.markdown("")
st.markdown(trans["total_week_details"][lang].format(team=selected_team_detail))
if selected_team_detail != "HWK Total":
    df_cum = df[(df["KPI"] == selected_kpi) & (df["Team"] == selected_team_detail) &
                (df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])]
    cum_value = cumulative_performance(df_cum, selected_kpi)
    # 전체 팀 중 해당 KPI의 누적 실적 최고값을 구함 (비교 기준)
    team_cum = df[(df["KPI"] == selected_kpi) & 
                  (df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])
                 ].groupby("Team").apply(lambda x: cumulative_performance(x, selected_kpi)).reset_index(name="cum")
    top_cum = team_cum["cum"].max()
    delta_cum = cum_value - top_cum
    arrow_cum = "▲" if delta_cum > 0 else "▼" if delta_cum < 0 else ""
    delta_cum_str = f"{arrow_cum}{delta_cum:+.2f}" if top_cum != 0 else ""
    st.markdown(f"**{selected_kpi} Cumulative:** {cum_value:.2f} " + (f"({delta_cum_str})" if delta_cum_str else ""))
else:
    # HWK Total: 누적 실적은 전체 팀 평균(누적)만 표시 (delta 없음)
    df_overall_cum = df[(df["KPI"] == selected_kpi) & 
                        (df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])]
    cum_value = cumulative_performance(df_overall_cum, selected_kpi)
    st.markdown(f"**{selected_kpi} Cumulative:** {cum_value:.2f}")

# --------------------------------------------------
# 11. [5] Detailed Data Table (행: 7 KPI, 열: 1주차/2주차/3주차/평균)
# --------------------------------------------------
st.markdown("")
st.markdown(trans["detailed_data"][lang])
# KPI 목록 (예: CSV에 있는 모든 KPI)
kpi_all = sorted(df["KPI"].unique())
weeks_to_show = [1, 2, 3]  # 1주차, 2주차, 3주차
data_table = {}
for kpi in kpi_all:
    row_data = {}
    values = []
    finals = []
    unit = ""
    for w in weeks_to_show:
        if selected_team_detail != "HWK Total":
            sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
        else:
            sub_df = df[(df["KPI"] == kpi) & (df["Week_num"] == w)]
        if not sub_df.empty:
            val = sub_df.iloc[0]["Actual_numeric"]
            final_val = sub_df.iloc[0]["Final"]
            unit = extract_unit(sub_df.iloc[0]["Actual"])
            formatted = f"{val:.2f}{unit} ({final_val} point)"
            row_data[f"Week {w}"] = formatted
            values.append(val)
            finals.append(final_val)
        else:
            row_data[f"Week {w}"] = "N/A"
    if values:
        avg_val = sum(values) / len(values)
        avg_final = sum(finals) / len(finals) if finals else 0
        row_data["Average"] = f"{avg_val:.2f}{unit} ({avg_final:.2f} point)"
    else:
        row_data["Average"] = "N/A"
    data_table[kpi] = row_data

table_df = pd.DataFrame(data_table).T
# 열 이름 변경 (예: "Week 1" -> "1주차", "Average" -> "평균")
new_cols = {}
for col in table_df.columns:
    if "Week" in col:
        week_num = col.split()[1]
        new_cols[col] = trans["week_col"][lang].format(week=week_num)
    elif col == "Average":
        new_cols[col] = trans["average"][lang]
table_df.rename(columns=new_cols, inplace=True)
st.dataframe(table_df)
