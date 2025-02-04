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
# 2-1. 하단에 작은 폰트 CSS (Last Week / Total Week 섹션 전용)
# --------------------------------------------------
st.markdown(
    """
    <style>
    .small-metric [data-testid="stMetricValue"] {
        font-size: 16px;
    }
    .small-metric [data-testid="stMetricDelta"] {
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True
)

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

# 최신주 데이터 (df_latest): 여기서는 팀별(원래 데이터에 있는) 데이터만 포함
df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()

# 만약 비교용 팀에 "HWK Total"이 포함되어 있다면 전체 팀(팀 필터 미적용) 데이터로 전체 평균 행 생성
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
# x축을 정수만 표시하도록 업데이트
fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)
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
# 랭킹은 "HWK Total" 제외
df_rank = df_comp[df_comp["Team"] != "HWK Total"].copy()
df_rank = df_rank.sort_values("Actual_numeric", ascending=False)
top_n = 3 if len(df_rank) >= 3 else len(df_rank)
bottom_n = 3 if len(df_rank) >= 3 else len(df_rank)
top_df = df_rank.head(top_n).copy()
bottom_df = df_rank.tail(bottom_n).copy().sort_values("Actual_numeric", ascending=True)
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
    fig_bottom.update_traces(texttemplate="%{text}", textposition='outside', marker_color='red')
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
st.markdown(trans["last_week_details"][lang].format(team=selected_team_detail, week=latest_week))
# 영역에 작은 폰트 적용 (small-metric)
st.markdown('<div class="small-metric">', unsafe_allow_html=True)
if selected_team_detail != "HWK Total":
    df_last = df_team[df_team["Week_num"] == latest_week]
    df_prev = df_team[df_team["Week_num"] == (latest_week - 1)]
else:
    # 그룹화된 경우: KPI별 평균을 이미 구했으므로 사용 (Week 정보가 없으므로 그대로 사용)
    df_last = df_team.copy()
    # 지난주 데이터: 그룹화하여 KPI별 평균 계산
    df_prev = df[df["Week_num"] == (latest_week - 1)].groupby("KPI").agg({
        "Actual_numeric": "mean",
        "Final": "mean",
        "Actual": "first"
    }).reset_index()
cols = st.columns(3)
i = 0
for kpi in df_last["KPI"].unique():
    if selected_team_detail != "HWK Total":
        row_last = df_last[df_last["KPI"] == kpi].iloc[0]
        prev_rows = df_prev[df_prev["KPI"] == kpi]
    else:
        row_last = df_last[df_last["KPI"] == kpi].iloc[0]
        prev_rows = df_prev[df_prev["KPI"] == kpi]
    current_label = format_label(row_last)
    if not prev_rows.empty:
        row_prev = prev_rows.iloc[0]
        delta_actual = row_last["Actual_numeric"] - row_prev["Actual_numeric"] if pd.notna(row_last["Actual_numeric"]) and pd.notna(row_prev["Actual_numeric"]) else None
        if pd.notna(row_last["Final"]) and pd.notna(row_prev["Final"]):
            delta_final = int(round(row_last["Final"])) - int(round(row_prev["Final"]))
        else:
            delta_final = None
        arrow = ""
        if delta_actual is not None:
            arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
        if delta_actual is not None and delta_final is not None:
            delta_str = f"{arrow}{delta_actual:+.2f}%({delta_final:+d} point)"
        else:
            delta_str = "N/A"
    else:
        delta_str = "N/A"
    cols[i % 3].metric(label=kpi, value=current_label, delta=delta_str)
    i += 1
st.markdown('</div>', unsafe_allow_html=True)

# --- (B) Total Week Performance Detail (누적 실적) ---
st.markdown("")
st.markdown(trans["total_week_details"][lang].format(team=selected_team_detail))
st.markdown('<div class="small-metric">', unsafe_allow_html=True)
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
    top_cum = team_cum["cum"].max() if not team_cum.empty else 0
    delta_cum = cum_value - top_cum
    arrow_cum = "▲" if delta_cum > 0 else "▼" if delta_cum < 0 else ""
    delta_cum_str = f"{arrow_cum}{delta_cum:+.2f} point" if top_cum != 0 else ""
    cols_total[i % 3].metric(label=kpi, value=f"{cum_value:.2f}", delta=delta_cum_str)
    i += 1
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# 11. [5] Detailed Data Table (행: 7 KPI, 열: 1주차/2주차/3주차/평균)
# --------------------------------------------------
st.markdown("")
st.markdown(trans["detailed_data"][lang])
kpi_all = sorted(df["KPI"].unique())
weeks_to_show = [1, 2, 3]
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
            # shortage_cost의 경우 수치 앞에 $ 추가
            if kpi.lower() == "shortage_cost":
                formatted = f"${val:.2f} ({final_val} point)"
            else:
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
new_cols = {}
for col in table_df.columns:
    if "Week" in col:
        week_num = col.split()[1]
        new_cols[col] = trans["week_col"][lang].format(week=week_num)
    elif col == "Average":
        new_cols[col] = trans["average"][lang]
table_df.rename(columns=new_cols, inplace=True)
st.dataframe(table_df)
