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

# 번역 사전 (영어/한글/베트남어)
trans = {
    "title": {
        "en": "HWK Quality competition Event",
        "ko": "HWK 품질 경쟁 이벤트",
        "vi": "HWK sự kiện thi đua chất lượng"
    },
    "kpi_comparison": {
        "en": "1. KPI Performance Comparison by Team",
        "ko": "1. 팀별 KPI 성과 비교",
        "vi": "So sánh Hiệu suất KPI theo Nhóm"
    },
    "weekly_trend": {
        "en": "2. Weekly Performance Trend Analysis",
        "ko": "2. 주간 성과 트렌드 분석",
        "vi": "2. Phân tích Xu hướng Hiệu suất Hàng Tuần"
    },
    "top_bottom_rankings": {
        "en": "3. KPI Top/Bottom Team Rankings",
        "ko": "3. KPI 상위/하위 팀 순위",
        "vi": "3. Xếp hạng Nhóm KPI Cao/Thấp Nhất"
    },
    "last_week_details": {
        "en": "Last Week performance Details for {team} (Week {week})",
        "ko": "지난주 성과 상세보기: {team} (Week {week})",
        "vi": "Chi tiết Hiệu suất Tuần Trước của {team} (Tuần {week})"
    },
    "total_week_details": {
        "en": "Total Week Performance Detail for {team} (All weeks)",
        "ko": "전체 주차 누적 실적 상세: {team} (All weeks)",
        "vi": "Chi tiết Hiệu suất Tổng Tuần của {team} (Tất cả các tuần)"
    },
    "detailed_data": {
        "en": "Detailed Data for Selected Team",
        "ko": "선택된 팀의 상세 데이터",
        "vi": "Dữ liệu Chi tiết cho Nhóm Đã Chọn"
    },
    "select_kpi": {
        "en": "Select KPI",
        "ko": "KPI 선택",
        "vi": "Chọn KPI"
    },
    "select_teams": {
        "en": "Select Teams for Comparison",
        "ko": "비교할 팀 선택 (HWK Total 포함)",
        "vi": "Chọn Nhóm để So Sánh"
    },
    "select_team_details": {
        "en": "Select Team for Details",
        "ko": "상세 조회할 팀 선택 (HWK Total 포함)",
        "vi": "chọn Nhóm để xem chi tiết"
    },
    "select_week_range": {
        "en": "Select Week Range",
        "ko": "주차 범위 선택",
        "vi": "Chọn Phạm vi Tuần"
    },
    "language": {
        "en": "Language",
        "ko": "언어",
        "vi": "ngôn ngữ"
    },
    "avg_by_team": {
        "en": "Average {kpi} by Team",
        "ko": "팀별 {kpi} 평균",
        "vi": "Trung bình {kpi} theo Nhóm"
    },
    "weekly_trend_title": {
        "en": "Weekly Trend of {kpi}",
        "ko": "{kpi} 주간 추이",
        "vi": "Xu hướng Hàng Tuần của {kpi}"
    },
    "top_teams": {
        "en": "Top {n} Teams - {kpi}",
        "ko": "{kpi} 상위 {n} 팀",
        "vi": "Top {n} Nhóm - {kpi}"
    },
    "bottom_teams": {
        "en": "Bottom {n} Teams - {kpi}",
        "ko": "{kpi} 하위 {n} 팀",
        "vi": "Nhóm {n} Thấp Nhất - {kpi}"
    },
    "week_col": {
        "en": "Week {week}",
        "ko": "{week}주차",
        "vi": "Tuần {week}"
    },
    "average": {
        "en": "Average",
        "ko": "평균",
        "vi": "Trung bình"
    }
}

# --------------------------------------------------
# 2. 우측 상단 언어 선택 (영어/한글/베트남어)
# --------------------------------------------------
col_title, col_lang = st.columns([4, 1])
with col_lang:
    lang = st.radio("Language / 언어 / ngôn ngữ", options=["en", "ko", "vi"], index=0, horizontal=True)

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
    if isinstance(s, str):
        m = re.search(r'([^\d\.\-]+)$', s.strip())
        return m.group(1).strip() if m else ""
    else:
        return ""

def format_label(row):
    unit = extract_unit(row["Actual"]) if pd.notnull(row.get("Actual", "")) else ""
    return f"{row['Actual_numeric']:.2f}{unit} ({row['Final']} point)"

def cumulative_performance(sub_df, kpi):
    if kpi.lower() == "shortage_cost":
        return sub_df["Actual_numeric"].sum()
    else:
        return sub_df["Actual_numeric"].mean()

def get_delta_color(kpi, delta):
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
    html_metric = f"""
    <div style="font-size:14px; margin:5px; padding:5px;">
      <div style="font-weight:bold;">{label}</div>
      <div>{value}</div>
      <div style="color:{delta_color};">{delta}</div>
    </div>
    """
    col.markdown(html_metric, unsafe_allow_html=True)

def format_final_label(row):
    return f"{row['Final']:.0f} point"

# --------------------------------------------------
# 4. 데이터 로드 및 전처리
# --------------------------------------------------
df = load_data()

# "Week" 열의 값에서 숫자만 추출하여 Week_num 열 생성 (예: "w4" → 4)
df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if isinstance(x, str) and re.sub(r'\D', '', x) != '' else np.nan)
df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

# 디버그용: CSV에 있는 주차 확인
st.write("CSV에 있는 주차:", sorted(df["Week_num"].dropna().unique()))

# --------------------------------------------------
# 5. 사이드바 위젯 (필터)
# --------------------------------------------------
st.sidebar.header("Filter Options")
kpi_options = sorted(list(df["KPI"].unique()))
if "Final score" not in kpi_options:
    kpi_options.append("Final score")
selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)
team_list = sorted(df["Team"].unique())
team_list_extended = team_list.copy()
if "HWK Total" not in team_list_extended:
    team_list_extended.append("HWK Total")
selected_teams = st.sidebar.multiselect(trans["select_teams"][lang], options=team_list_extended, default=team_list)

# 슬라이더는 전체 CSV 데이터를 기준으로 주차 범위(정수 단위, step=1)를 지정
selected_week_range = st.sidebar.slider(
    trans["select_week_range"][lang],
    int(df["Week_num"].min()),
    int(df["Week_num"].max()),
    (int(df["Week_num"].min()), int(df["Week_num"].max())),
    step=1
)
selected_team_detail = st.sidebar.selectbox(trans["select_team_details"][lang], options=team_list_extended, index=0)

# --------------------------------------------------
# 6. 데이터 필터링 (KPI, 주차 범위 적용)
# --------------------------------------------------
if selected_kpi == "Final score":
    df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & 
                     (df["Week_num"] <= selected_week_range[1])].copy()
else:
    df_filtered = df[(df["KPI"] == selected_kpi) & 
                     (df["Week_num"] >= selected_week_range[0]) & 
                     (df["Week_num"] <= selected_week_range[1])].copy()

# KPI가 Final score가 아닐 경우, 최신 주차를 기준으로 상세정보 활용
if selected_kpi != "Final score":
    if not df_filtered.empty:
        latest_week = int(df_filtered["Week_num"].max())
    else:
        latest_week = None
else:
    if not df_filtered.empty:
        latest_week = int(df_filtered["Week_num"].max())
    else:
        latest_week = None

# --------------------------------------------------
# 7. [1] KPI Performance Comparison by Team (바 차트)
# --------------------------------------------------
st.markdown(trans["kpi_comparison"][lang])
if selected_kpi == "Final score":
    df_latest = df_filtered.groupby("Team").agg({"Final": "sum"}).reset_index()
    df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
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
            
if selected_kpi == "Final score":
    df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
else:
    df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
    df_comp["Label"] = df_comp.apply(format_label, axis=1)

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
    df_trend_individual = df_filtered.sort_values("Week_num").groupby("Team").apply(
        lambda x: x.assign(CumFinal=x["Final"].cumsum())
    ).reset_index(drop=True)
    fig_line = px.line(
        df_trend_individual,
        x="Week_num",
        y="CumFinal",
        color="Team",
        markers=True,
        labels={"Week_num": "Week", "CumFinal": "Cumulative Final score"},
        title="Weekly Trend of Final score (Cumulative)"
    )
    fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)
    if "HWK Total" in selected_teams:
        df_overall_trend = df_filtered.sort_values("Week_num").groupby("Week_num").agg({"Final": "sum"}).reset_index()
        df_overall_trend["CumFinal"] = df_overall_trend["Final"].cumsum()
        fig_line.add_scatter(
            x=df_overall_trend["Week_num"],
            y=df_overall_trend["CumFinal"],
            mode='lines+markers',
            name="HWK Total",
            line=dict(color='black', dash='dash')
        )
else:
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
if selected_team_detail != "HWK Total":
    df_team = df[df["Team"] == selected_team_detail].copy()
else:
    df_team = df[df["Week_num"] == latest_week].copy()
    df_team = df_team.groupby("KPI").agg({
        "Actual_numeric": "mean",
        "Final": "mean",
        "Actual": "first"
    }).reset_index()

# (A) Last Week performance Details – 폰트 크기를 18px로 확대
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

# (B) Total Week Performance Detail – 폰트 크기를 18px로 확대
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
kpi_all = sorted(df["KPI"].unique())
# CSV 파일에 있는 모든 주차 데이터를 동적으로 표시하도록 설정
max_week = int(df["Week_num"].max())
weeks_to_show = list(range(1, max_week + 1))
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

table_df = pd.DataFrame(data_table)
index_order = [f"Week {w}" for w in weeks_to_show] + ["Average"]
table_df = table_df.reindex(index_order)
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
st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)
