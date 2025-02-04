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
    # st.radio의 옵션은 "en" 또는 "ko"로 설정 (기본: 영어)
    lang = st.radio("Language / 언어", options=["en", "ko"], index=0, horizontal=True)

# 제목 출력 (언어에 따라)
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
    한 행(row)의 Actual_numeric 값과 Final 값을 소수점 2자리와 함께
    "value{unit} (Final point)" 형식으로 반환
    """
    unit = extract_unit(row["Actual"]) if pd.notnull(row.get("Actual", "")) else ""
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

def get_delta_color(kpi, delta):
    """
    KPI별로 개선(좋은 결과)는 파란색, 악화(나쁜 결과)는 빨간색으로 반환
    - "prs validation", "6S_audit": 수치가 클수록 품질이 좋음 → delta > 0 이면 개선(blue)
    - "AQL_performance", "B-grade", "attendance", "issue_tracking", "shortage_cost": 수치가 작을수록 품질이 좋음 → delta < 0 이면 개선(blue)
    """
    if delta is None:
        return "black"
    kpi_lower = kpi.lower()
    positive_better = ["prs validation", "6s_audit"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
    if kpi_lower in positive_better:
        return "blue" if delta > 0 else "red" if delta < 0 else "black"
    elif kpi_lower in negative_better:
        return "blue" if delta < 0 else "red" if delta > 0 else "black"
    else:
        return "blue" if delta > 0 else "red" if delta < 0 else "black"

def render_custom_metric(col, label, value, delta, delta_color):
    """
    col: Streamlit 컬럼 객체
    label: KPI명
    value: 현재 성과 문자열 (예: "12.34% (10 point)")
    delta: 변화량 문자열 (예: "▲+1.23%(+2 point)")
    delta_color: 델타 문자열의 색상
    폰트 크기는 약간 키워서 14px로 표시
    """
    html_metric = f"""
    <div style="font-size:14px; margin:5px; padding:5px;">
      <div style="font-weight:bold;">{label}</div>
      <div>{value}</div>
      <div style="color:{delta_color};">{delta}</div>
    </div>
    """
    col.markdown(html_metric, unsafe_allow_html=True)

def format_final_label(row):
    """Final score 전용 라벨: 정수형 점수 뒤에 'point' 표기"""
    return f"{row['Final']:.0f} point"

# --------------------------------------------------
# 4. 데이터 로드 및 전처리
# --------------------------------------------------
df = load_data()

# Week 컬럼에서 숫자만 추출하여 Week_num 컬럼 생성 (예: "W1" -> 1)
df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if isinstance(x, str) and re.sub(r'\D', '', x) != '' else np.nan)
# Actual 값을 numeric으로 변환
df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
# Final 컬럼을 숫자형으로 변환
df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

# --------------------------------------------------
# 5. 사이드바 위젯 (필터)
# --------------------------------------------------
st.sidebar.header("Filter Options")
# KPI 목록에 기존 KPI들과 함께 "Final score" 항목 추가
kpi_options = sorted(list(df["KPI"].unique()))
if "Final score" not in kpi_options:
    kpi_options.append("Final score")
selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)
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
if selected_kpi == "Final score":
    # Final score의 경우, KPI 조건 없이 주차 범위만 적용
    df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & 
                     (df["Week_num"] <= selected_week_range[1])].copy()
else:
    df_filtered = df[(df["KPI"] == selected_kpi) & 
                     (df["Week_num"] >= selected_week_range[0]) & 
                     (df["Week_num"] <= selected_week_range[1])].copy()

# 최신주 결정 (필터된 데이터 중 최대 주차)
if not df_filtered.empty:
    latest_week = int(df_filtered["Week_num"].max())
else:
    latest_week = None

# 최신주 데이터 (df_latest)
if selected_kpi == "Final score":
    # Final score의 경우, 각 팀별 해당 주의 Final score(7개 항목의 합계)를 계산
    df_latest = df_filtered[df_filtered["Week_num"] == latest_week].groupby("Team").agg({"Final": "sum"}).reset_index()
    df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
    # "HWK Total" 선택 시 전체 팀의 누적 합계로 계산 (평균이 아닌 누적 합계)
    if "HWK Total" in selected_teams:
        overall_final = df_latest["Final"].sum()
        df_total = pd.DataFrame({
            "Team": ["HWK Total"],
            "Final": [overall_final]
        })
        df_total["Label"] = df_total.apply(format_final_label, axis=1)
        df_latest = pd.concat([df_latest, df_total], ignore_index=True)
else:
    df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()
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
if selected_kpi == "Final score":
    df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
else:
    df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
    df_comp["Label"] = df_comp.apply(format_label, axis=1)

# --------------------------------------------------
# 7. [1] KPI Performance Comparison by Team (바 차트)
# --------------------------------------------------
st.markdown(trans["kpi_comparison"][lang])
if selected_kpi == "Final score":
    fig_bar = px.bar(
        df_comp,
        x="Team",
        y="Final",
        text="Label",
        labels={"Final": "Final score by Team"}
    )
    fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
else:
    fig_bar = px.bar(
        df_comp,
        x="Team",
        y="Actual_numeric",
        text="Label",
        labels={"Actual_numeric": trans["avg_by_team"][lang].format(kpi=selected_kpi)}
    )
    fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# --------------------------------------------------
# 8. [2] Weekly Performance Trend Analysis (라인 차트)
# --------------------------------------------------
st.markdown(trans["weekly_trend"][lang])
if selected_kpi == "Final score":
    # Final score인 경우, 각 팀/주차별로 Final 값(누적 합계)을 계산
    df_trend_individual = df_filtered.groupby(["Team", "Week_num"]).agg({"Final": "sum"}).reset_index()
    fig_line = px.line(
        df_trend_individual,
        x="Week_num",
        y="Final",
        color="Team",
        markers=True,
        labels={"Week_num": "Week", "Final": "Final score"},
        title="Weekly Trend of Final score"
    )
    fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)
    if "HWK Total" in selected_teams:
        # HWK Total: 전체 팀의 주차별 누적 합계 Final score 계산
        df_overall_trend = df_filtered.groupby("Week_num").agg({"Final": "sum"}).reset_index()
        fig_line.add_scatter(
            x=df_overall_trend["Week_num"],
            y=df_overall_trend["Final"],
            mode='lines+markers',
            name="HWK Total",
            line=dict(color='black', dash='dash')
        )
else:
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
    fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)
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
# 랭킹은 "HWK Total" 제외
df_rank = df_comp[df_comp["Team"] != "HWK Total"].copy()
df_rank = df_rank.sort_values("Actual_numeric" if selected_kpi != "Final score" else "Final", ascending=False)
top_n = 3 if len(df_rank) >= 3 else len(df_rank)
bottom_n = 3 if len(df_rank) >= 3 else len(df_rank)
top_df = df_rank.head(top_n).copy()
bottom_df = df_rank.tail(bottom_n).copy().sort_values("Actual_numeric" if selected_kpi != "Final score" else "Final", ascending=True)
if selected_kpi == "Final score":
    top_df["Label"] = top_df.apply(lambda row: format_final_label(row), axis=1)
    bottom_df["Label"] = bottom_df.apply(lambda row: format_final_label(row), axis=1)
else:
    top_df["Label"] = top_df.apply(format_label, axis=1)
    bottom_df["Label"] = bottom_df.apply(format_label, axis=1)
col1, col2 = st.columns(2)
with col1:
    st.subheader(trans["top_teams"][lang].format(n=top_n, kpi=selected_kpi))
    if selected_kpi == "Final score":
        fig_top = px.bar(
            top_df,
            x="Final",
            y="Team",
            orientation="h",
            text="Label",
            labels={"Final": "Final score", "Team": "Team"}
        )
    else:
        fig_top = px.bar(
            top_df,
            x="Actual_numeric",
            y="Team",
            orientation="h",
            text="Label",
            labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
        )
    fig_top.update_traces(texttemplate="%{text}", textposition='inside')
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True, key="top_chart")
with col2:
    st.subheader(trans["bottom_teams"][lang].format(n=bottom_n, kpi=selected_kpi))
    if selected_kpi == "Final score":
        fig_bottom = px.bar(
            bottom_df,
            x="Final",
            y="Team",
            orientation="h",
            text="Label",
            labels={"Final": "Final score", "Team": "Team"}
        )
    else:
        fig_bottom = px.bar(
            bottom_df,
            x="Actual_numeric",
            y="Team",
            orientation="h",
            text="Label",
            labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
        )
    fig_bottom.update_traces(texttemplate="%{text}", textposition='inside', marker_color='red')
    fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# --------------------------------------------------
# 10. [4] Team-Specific KPI Detailed View (카드형 레이아웃)
# --------------------------------------------------
st.markdown("")

# 분기: 선택한 팀이 "HWK Total" 인지에 따라 처리
if selected_team_detail != "HWK Total":
    # 개별 팀의 모든 KPI 데이터 (KPI 조건 없이)
    df_team = df[df["Team"] == selected_team_detail].copy()
else:
    # HWK Total: 최신주에 대한 전체 팀 데이터를 KPI별 평균으로 계산
    df_team = df[df["Week_num"] == latest_week].copy()
    df_team = df_team.groupby("KPI").agg({
        "Actual_numeric": "mean",
        "Final": "mean",
        "Actual": "first"
    }).reset_index()

# --- (A) Last Week performance Details ---
st.markdown(
    f"<div style='font-size:18px; font-weight:bold;'>{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}</div>",
    unsafe_allow_html=True
)
cols = st.columns(3)
i = 0
for kpi in df_team["KPI"].unique():
    if selected_team_detail != "HWK Total":
        df_last = df_team[df_team["Week_num"] == latest_week]
        df_prev = df_team[df_team["Week_num"] == (latest_week - 1)]
    else:
        df_last = df_team.copy()
        df_prev = df[df["Week_num"] == (latest_week - 1)].groupby("KPI").agg({
            "Actual_numeric": "mean",
            "Final": "mean",
            "Actual": "first"
        }).reset_index()
    if not df_last[df_last["KPI"] == kpi].empty:
        row_last = df_last[df_last["KPI"] == kpi].iloc[0]
    else:
        continue
    current_label = format_label(row_last)
    # delta 계산
    if not df_prev[df_prev["KPI"] == kpi].empty:
        row_prev = df_prev[df_prev["KPI"] == kpi].iloc[0]
        if pd.notna(row_last["Actual_numeric"]) and pd.notna(row_prev["Actual_numeric"]):
            delta_actual = row_last["Actual_numeric"] - row_prev["Actual_numeric"]
        else:
            delta_actual = None
        if pd.notna(row_last["Final"]) and pd.notna(row_prev["Final"]):
            delta_final = int(round(row_last["Final"])) - int(round(row_prev["Final"]))
        else:
            delta_final = None
        if delta_actual is not None and delta_final is not None:
            kpi_lower = kpi.lower()
            positive_better = ["prs validation", "6s_audit"]
            if kpi_lower in positive_better:
                arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
            else:
                arrow = "▲" if delta_actual < 0 else "▼" if delta_actual > 0 else ""
            delta_str = f"{arrow}{delta_actual:+.2f}%({delta_final:+d} point)"
        else:
            delta_str = "N/A"
    else:
        delta_str = "N/A"
        delta_actual = None
    delta_color = get_delta_color(kpi, delta_actual)
    render_custom_metric(cols[i % 3], kpi, current_label, delta_str, delta_color)
    i += 1

# --- (B) Total Week Performance Detail (누적 실적) ---
st.markdown("")
st.markdown(
    f"<div style='font-size:18px; font-weight:bold;'>{trans['total_week_details'][lang].format(team=selected_team_detail)}</div>",
    unsafe_allow_html=True
)
if selected_team_detail != "HWK Total":
    df_cum = df[(df["Team"] == selected_team_detail) & 
                (df["Week_num"] >= selected_week_range[0]) & 
                (df["Week_num"] <= selected_week_range[1])]
else:
    df_cum = df[(df["Week_num"] >= selected_week_range[0]) & 
                (df["Week_num"] <= selected_week_range[1])]
df_cum_group = df_cum.groupby("KPI").apply(lambda x: cumulative_performance(x, x["KPI"].iloc[0])).reset_index(name="cum")
cols_total = st.columns(3)
i = 0
for kpi in df_cum_group["KPI"].unique():
    sub_df = df_cum[df_cum["KPI"] == kpi]
    cum_value = cumulative_performance(sub_df, kpi)
    team_cum = df[(df["KPI"] == kpi) & 
                  (df["Week_num"] >= selected_week_range[0]) & 
                  (df["Week_num"] <= selected_week_range[1])].groupby("Team").apply(lambda x: cumulative_performance(x, kpi)).reset_index(name="cum")
    kpi_lower = kpi.lower()
    positive_better = ["prs validation", "6s_audit"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
    if kpi_lower in positive_better:
        best_value = team_cum["cum"].max() if not team_cum.empty else 0
        delta = cum_value - best_value
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""
    elif kpi_lower in negative_better:
        best_value = team_cum["cum"].min() if not team_cum.empty else 0
        delta = cum_value - best_value
        arrow = "▲" if delta < 0 else "▼" if delta > 0 else ""
    else:
        best_value = team_cum["cum"].max() if not team_cum.empty else 0
        delta = cum_value - best_value
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""
    delta_str = f"{arrow}{abs(delta):+.2f} point" if best_value != 0 else ""
    delta_color = get_delta_color(kpi, delta)
    render_custom_metric(cols_total[i % 3], kpi, f"{cum_value:.2f}", delta_str, delta_color)
    i += 1

# --------------------------------------------------
# 11. [5] Detailed Data Table (행과 열 전환: 행=주차, 열=KPI)
# --------------------------------------------------
st.markdown("")
st.markdown(trans["detailed_data"][lang])
# 주차 수는 현재 3주차로 되어 있으나 추후 12주차까지 늘어날 것을 고려하여 유연하게 처리
# 우선 KPI별로 기존 데이터를 구성한 후 전치
kpi_all = sorted(df["KPI"].unique())
# weeks_to_show 예시 (추후 12주차까지 확장 가능)
weeks_to_show = list(range(1, 4))  
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
            # 괄호 부분을 줄바꿈하여 2줄로 표기
            formatted = f"{val:.2f}{unit}<br>({final_val} point)"
            row_data[f"Week {w}"] = formatted
            values.append(val)
            finals.append(final_val)
        else:
            row_data[f"Week {w}"] = "N/A"
    if values:
        avg_val = sum(values) / len(values)
        avg_final = sum(finals) / len(finals) if finals else 0
        row_data["Average"] = f"{avg_val:.2f}{unit}<br>({avg_final:.2f} point)"
    else:
        row_data["Average"] = "N/A"
    data_table[kpi] = row_data

# 기존 data_table: key=KPI, value=dict(주차별 실적) → 전치하여 행=주차, 열=KPI
table_df = pd.DataFrame(data_table)  # 행: 주차 (Week 1, Week 2, …, Average), 열: KPI
# 재정렬: 인덱스를 주차 순서로 정렬 (만약 일부 주차가 없으면 그대로 둠)
index_order = [f"Week {w}" for w in weeks_to_show] + ["Average"]
table_df = table_df.reindex(index_order)
# 인덱스(주차)를 다국어로 변환
new_index = {}
for idx in table_df.index:
    if idx.startswith("Week"):
        week_num = idx.split()[1]
        new_index[idx] = trans["week_col"][lang].format(week=week_num)
    elif idx == "Average":
        new_index[idx] = trans["average"][lang]
    else:
        new_index[idx] = idx
table_df.rename(index=new_index, inplace=True)
# HTML 태그(<br>)가 포함되어 있으므로 st.dataframe 대신 st.markdown으로 HTML 테이블 렌더링
st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)
