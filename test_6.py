import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import unicodedata
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
        "vi": "Chọn Nhóm để xem chi tiết"
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

# KPI별 단위 매핑
KPI_UNITS = {
    "prs validation": "%",
    "6s_audit": "%",
    "aql_performance": "%",
    "b-grade": "%",
    "attendance": "%",
    "issue_tracking": "minutes",
    "shortage_cost": "$",
    "final score": ""
}

# KPI별 한글 표기 매핑 (lang == "ko"일 때 사용)
KPI_NAME_MAP = {
    "prs validation": {
        "ko": "포장 제품 5족 품질 검증 통과율",
        "en": "prs validation",
        "vi": "prs validation"
    },
    "6s_audit": {
        "ko": "6S 어딧 점수",
        "en": "6s_audit",
        "vi": "6s_audit"
    },
    "aql_performance": {
        "ko": "수검 리젝율",
        "en": "aql_performance",
        "vi": "aql_performance"
    },
    "b-grade": {
        "ko": "B-grade 발생율",
        "en": "b-grade",
        "vi": "b-grade"
    },
    "attendance": {
        "ko": "결근율",
        "en": "attendance",
        "vi": "attendance"
    },
    "issue_tracking": {
        "ko": "이슈 개선 소요 시간",
        "en": "issue_tracking",
        "vi": "issue_tracking"
    },
    "shortage_cost": {
        "ko": "부족분 금액",
        "en": "shortage_cost",
        "vi": "shortage_cost"
    },
    "final score": {
        "ko": "Final score",
        "en": "Final score",
        "vi": "Final score"
    }
}

def get_kpi_display_name(kpi_name: str, lang: str) -> str:
    key = kpi_name.lower()
    if key in KPI_NAME_MAP:
        return KPI_NAME_MAP[key].get(lang, kpi_name)
    else:
        return kpi_name

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
def remove_all_spaces(s: str) -> str:
    return re.sub(r'\s+', '', s)

def to_halfwidth(s: str) -> str:
    return unicodedata.normalize('NFKC', s)

@st.cache_data
def load_data():
    df = pd.read_csv("score.csv", sep="\t", encoding="utf-8")
    return df

def convert_to_numeric(x):
    try:
        if isinstance(x, str):
            if x.strip() == '-' or x.strip() == '':
                return np.nan
            x = x.replace(",", ".")
            num = re.sub(r'[^\d\.-]', '', x)
            return float(num)
        else:
            return x
    except:
        return np.nan

def get_kpi_unit(kpi_name: str) -> str:
    return KPI_UNITS.get(kpi_name.lower(), "")

def cumulative_performance(sub_df, kpi):
    if kpi.lower() == "shortage_cost":
        return sub_df["Actual_numeric"].sum()
    else:
        return sub_df["Actual_numeric"].mean()

def get_trend_emoticon(kpi, delta):
    if delta is None:
        return ""
    kpi_lower = kpi.lower()
    positive_better = ["prs validation", "6s_audit", "final score"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
    if kpi_lower in positive_better:
        if delta > 0:
            return "😀"
        elif delta < 0:
            return "😡"
        else:
            return ""
    elif kpi_lower in negative_better:
        if delta < 0:
            return "😀"
        elif delta > 0:
            return "😡"
        else:
            return ""
    else:
        if delta > 0:
            return "😀"
        elif delta < 0:
            return "😡"
        else:
            return ""

def render_custom_metric(col, label, value, delta_str, color="black"):
    html_metric = f"""
    <div style="font-size:14px; margin:5px; padding:5px;">
      <div style="font-weight:bold;">{label}</div>
      <div>{value}</div>
      <div style="color:{color};">{delta_str}</div>
    </div>
    """
    col.markdown(html_metric, unsafe_allow_html=True)

def format_final_label(row):
    return f"{row['Final']:.0f} point"

def get_range_comment(lang_code, start_week, end_week):
    if lang_code == "ko":
        return f"({start_week}주차~{end_week}주차 평균)"
    elif lang_code == "vi":
        return f"(Từ Tuần {start_week} đến Tuần {end_week} trung bình)"
    else:
        return f"(From Week {start_week} to Week {end_week} average)"

# --------------------------------------------------
# 4. 데이터 로드 및 전처리
# --------------------------------------------------
df = load_data()
df["Week"] = (
    df["Week"]
    .astype(str)
    .apply(to_halfwidth)
    .str.upper()
    .apply(remove_all_spaces)
)
df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x) else np.nan)
df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

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
min_week = int(df["Week_num"].min())
max_week = int(df["Week_num"].max())
selected_week_range = st.sidebar.slider(
    trans["select_week_range"][lang],
    min_week,
    max_week,
    (min_week, max_week),
    step=1
)
selected_team_detail = st.sidebar.selectbox(trans["select_team_details"][lang], options=team_list_extended, index=0)

# --------------------------------------------------
# 6. 데이터 필터링 (KPI, 주차 범위 적용)
# --------------------------------------------------
if selected_kpi == "Final score":
    df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])].copy()
else:
    df_filtered = df[(df["KPI"] == selected_kpi) & (df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])].copy()
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
    if "HWK Total" in selected_teams and not df_latest.empty:
        overall_actual = df_latest["Actual_numeric"].mean()
        overall_final = round(df_latest["Final"].mean())
        df_total = pd.DataFrame({
            "Team": ["HWK Total"],
            "Actual_numeric": [overall_actual],
            "Final": [overall_final],
            "Actual": [f"{overall_actual:.2f}"],
            "Week_num": [latest_week],
            "KPI": [selected_kpi]
        })
        df_latest = pd.concat([df_latest, df_total], ignore_index=True)
df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
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
    def make_bar_label(row):
        k_unit = get_kpi_unit(row["KPI"])
        val = row["Actual_numeric"]
        fin = row["Final"]
        return f"{val:.2f}{k_unit} ({fin} point)"
    df_comp["Label"] = df_comp.apply(make_bar_label, axis=1)
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
    if selected_kpi.lower() == "b-grade":
        y_label = "b-grade (%)"
    else:
        y_label = f"{selected_kpi} Value"
    df_trend_individual = df_filtered[df_filtered["Team"].isin([t for t in selected_teams if t != "HWK Total"])].copy()
    fig_line = px.line(
        df_trend_individual,
        x="Week_num",
        y="Actual_numeric",
        color="Team",
        markers=True,
        labels={"Week_num": "Week", "Actual_numeric": y_label},
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
if selected_kpi == "Final score":
    df_rank = df_rank.sort_values("Final", ascending=False)
else:
    df_rank = df_rank.sort_values("Actual_numeric", ascending=False)
top_n = 3 if len(df_rank) >= 3 else len(df_rank)
bottom_n = 3 if len(df_rank) >= 3 else len(df_rank)
top_df = df_rank.head(top_n).copy()
bottom_df = df_rank.tail(bottom_n).copy()
if selected_kpi == "Final score":
    bottom_df = bottom_df.sort_values("Final", ascending=True)
else:
    bottom_df = bottom_df.sort_values("Actual_numeric", ascending=True)
if selected_kpi == "Final score":
    top_df["Label"] = top_df.apply(lambda row: format_final_label(row), axis=1)
    bottom_df["Label"] = bottom_df.apply(lambda row: format_final_label(row), axis=1)
else:
    def make_bar_label2(row):
        k_unit = get_kpi_unit(row["KPI"])
        val = row["Actual_numeric"]
        fin = row["Final"]
        return f"{val:.2f}{k_unit} ({fin} point)"
    top_df["Label"] = top_df.apply(make_bar_label2, axis=1)
    bottom_df["Label"] = bottom_df.apply(make_bar_label2, axis=1)
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
# (A) 해당 팀 데이터 준비
if selected_team_detail != "HWK Total":
    df_team = df[df["Team"] == selected_team_detail].copy()
else:
    df_team = df[df["Week_num"] == latest_week].groupby("KPI").agg({
        "Actual_numeric": "mean",
        "Final": "mean",
        "Actual": "first"
    }).reset_index()
# (A) Last Week performance Details
if latest_week is not None:
    st.markdown(
        f"<div style='font-size:18px; font-weight:bold;'>{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}</div>",
        unsafe_allow_html=True
    )
    cols = st.columns(3)
    i = 0
    kpi_list_for_team = df_team["KPI"].unique()
    for kpi in kpi_list_for_team:
        kpi_lower = kpi.lower()
        kpi_unit = get_kpi_unit(kpi)
        kpi_display_name = get_kpi_display_name(kpi, lang)
        # HWK Total & shortage_cost
        if selected_team_detail == "HWK Total" and kpi_lower == "shortage_cost":
            df_last_raw = df[(df["Week_num"] == latest_week) & (df["KPI"].str.lower() == "shortage_cost")]
            latest_sum = df_last_raw["Actual_numeric"].sum() if not df_last_raw.empty else np.nan
            current_label = f"{latest_sum:.2f}{kpi_unit} (Week {latest_week} total)"
            df_prev_raw = df[(df["Week_num"] == (latest_week - 1)) & (df["KPI"].str.lower() == "shortage_cost")]
            prev_sum = df_prev_raw["Actual_numeric"].sum() if not df_prev_raw.empty else np.nan
            if pd.notna(latest_sum) and pd.notna(prev_sum):
                delta_actual = latest_sum - prev_sum
            else:
                delta_actual = None
            emoticon = get_trend_emoticon(kpi, delta_actual)
            if delta_actual is not None and delta_actual != 0:
                delta_str = f"{emoticon}{delta_actual:+.2f}{kpi_unit}"
            else:
                delta_str = "N/A"
            render_custom_metric(cols[i % 3], kpi_display_name, current_label, delta_str)
            i += 1
            continue
        if selected_team_detail != "HWK Total":
            df_last = df_team[(df_team["Week_num"] == latest_week) & (df_team["KPI"] == kpi)]
            df_prev = df_team[(df_team["Week_num"] == (latest_week - 1)) & (df_team["KPI"] == kpi)]
        else:
            df_last = df_team[df_team["KPI"] == kpi]
            df_prev_raw = df[df["Week_num"] == (latest_week - 1)].groupby("KPI").agg({
                "Actual_numeric": "mean",
                "Final": "mean",
                "Actual": "first"
            }).reset_index()
            df_prev = df_prev_raw[df_prev_raw["KPI"] == kpi]
        if not df_last.empty:
            row_last = df_last.iloc[0]
        else:
            continue
        if selected_team_detail != "HWK Total":
            curr_val_str = f"{row_last['Actual_numeric']:.2f}{kpi_unit}"
            current_label = f"{curr_val_str} ({int(round(row_last['Final']))} point)"
        else:
            curr_val_str = f"{row_last['Actual_numeric']:.2f}{kpi_unit}"
            current_label = f"{curr_val_str} ({int(round(row_last['Final']))} point)"
        if not df_prev.empty:
            row_prev = df_prev.iloc[0]
            if pd.notna(row_last["Actual_numeric"]) and pd.notna(row_prev["Actual_numeric"]):
                delta_actual = row_last["Actual_numeric"] - row_prev["Actual_numeric"]
            else:
                delta_actual = None
            if pd.notna(row_last["Final"]) and pd.notna(row_prev["Final"]):
                delta_final = int(round(row_last["Final"])) - int(round(row_prev["Final"]))
            else:
                delta_final = None
            if delta_actual is not None and delta_final is not None:
                emoticon = get_trend_emoticon(kpi, delta_actual)
                delta_str = f"{emoticon}{delta_actual:+.2f}{kpi_unit}({delta_final:+d} point)"
            else:
                delta_str = "N/A"
        else:
            delta_str = "N/A"
        render_custom_metric(cols[i % 3], kpi_display_name, current_label, delta_str)
        i += 1
# (B) Total Week Performance Detail
st.markdown("")
st.markdown(
    f"<div style='font-size:18px; font-weight:bold;'>{trans['total_week_details'][lang].format(team=selected_team_detail)}</div>",
    unsafe_allow_html=True
)
if selected_team_detail != "HWK Total":
    df_cum = df[(df["Team"] == selected_team_detail) & (df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])]
else:
    df_cum = df[(df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])]
df_cum_group = df_cum.groupby("KPI").apply(lambda x: cumulative_performance(x, x["KPI"].iloc[0])).reset_index(name="cum")
cols_total = st.columns(3)
i = 0
for kpi in df_cum_group["KPI"].unique():
    kpi_lower = kpi.lower()
    kpi_unit = get_kpi_unit(kpi)
    kpi_display_name = get_kpi_display_name(kpi, lang)
    # HWK Total 선택 시
    if selected_team_detail == "HWK Total":
        if kpi_lower == "shortage_cost":
            if latest_week is not None:
                df_latest_sc = df[(df["Week_num"] == latest_week) & (df["KPI"].str.lower() == "shortage_cost")]
                latest_sum = df_latest_sc["Actual_numeric"].sum() if not df_latest_sc.empty else np.nan
            else:
                latest_sum = np.nan
            df_all_sc = df_cum[df_cum["KPI"].str.lower() == "shortage_cost"]
            total_sum_sc = df_all_sc["Actual_numeric"].sum() if not df_all_sc.empty else np.nan
            unique_weeks = df_all_sc["Week_num"].unique() if not df_all_sc.empty else []
            weeks_count = len(unique_weeks)
            if weeks_count > 0:
                avg_weekly_sc = total_sum_sc / weeks_count
            else:
                avg_weekly_sc = np.nan
            cum_value = latest_sum
            best_value = avg_weekly_sc
            if pd.notna(cum_value) and pd.notna(best_value):
                delta = cum_value - best_value
            else:
                delta = None
            emoticon = get_trend_emoticon(kpi, delta)
            range_comment = get_range_comment(lang, selected_week_range[0], latest_week if latest_week else selected_week_range[1])
            if delta is not None:
                delta_str = f"{emoticon}{delta:+.2f}{kpi_unit} {range_comment}"
            else:
                delta_str = ""
            full_text = f"{cum_value:.2f}{kpi_unit}<br>{delta_str}"
            render_custom_metric(cols_total[i % 3], kpi_display_name, full_text, "")
            i += 1
        else:
            sub_df = df_cum[df_cum["KPI"] == kpi]
            cum_value = cumulative_performance(sub_df, kpi)
            # 1주차부터 최신주-1까지의 평균 계산
            if latest_week is not None and latest_week > selected_week_range[0]:
                df_cum_prev = df[(df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] < latest_week)]
                prev_avg = cumulative_performance(df_cum_prev[df_cum_prev["KPI"] == kpi], kpi)
            else:
                prev_avg = None
            if prev_avg is not None:
                delta = cum_value - prev_avg
            else:
                delta = None
            emoticon = get_trend_emoticon(kpi, delta)
            range_comment = get_range_comment(lang, selected_week_range[0], latest_week if latest_week else selected_week_range[1])
            line1 = f"{cum_value:.2f}{kpi_unit}"
            if delta is not None:
                line2 = f"{emoticon}{delta:+.2f}{kpi_unit} {range_comment}"
            else:
                line2 = "N/A"
            full_text = f"{line1}<br>{line2}"
            render_custom_metric(cols_total[i % 3], kpi_display_name, full_text, "")
            i += 1
    else:
        # 개별 팀 선택 시: 3줄 표기
        sub_df = df_cum[df_cum["KPI"] == kpi]
        cum_value = cumulative_performance(sub_df, kpi)
        # Ranking 계산: 전체 팀 데이터를 기준으로 selected_team_detail의 순위(동률 처리)
        team_cum = df[(df["KPI"] == kpi) & (df["Week_num"] >= selected_week_range[0]) & (df["Week_num"] <= selected_week_range[1])].groupby("Team").apply(lambda x: cumulative_performance(x, kpi)).reset_index(name="cum")
        if kpi_lower in ["prs validation", "6s_audit", "final score"]:
            sorted_df = team_cum.sort_values("cum", ascending=False).reset_index(drop=True)
        elif kpi_lower in ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]:
            sorted_df = team_cum.sort_values("cum", ascending=True).reset_index(drop=True)
        else:
            sorted_df = team_cum.sort_values("cum", ascending=False).reset_index(drop=True)
        # 동률 처리: 표준 경쟁 순위 (동률이면 같은 순위, 다음 순위는 동률 수 만큼 건너뜀)
        # 동률 처리: 표준 경쟁 순위 (동률이면 같은 순위, 다음 순위는 동률 수 만큼 건너뜀)
        ranks = []
        current_rank = 1
        for i_row, row in sorted_df.iterrows():
            if i_row == 0:
                ranks.append(current_rank)
            else:
                if row["cum"] == sorted_df.iloc[i_row-1]["cum"]:
                    ranks.append(current_rank)
                else:
                    current_rank = i_row + 1
                    ranks.append(current_rank)
        selected_rank = None
        for i_row, row in sorted_df.iterrows():
            if row["Team"] == selected_team_detail:
                selected_rank = ranks[i_row]
                break
        if selected_rank is not None:
            if selected_rank == 1:
                rank_str = f'<span style="color:blue;">Top {selected_rank}</span>'
            elif selected_rank == 7:
                rank_str = f'<span style="color:red;">Top {selected_rank}</span>'
            else:
                rank_str = f"Top {selected_rank}"
        else:
            rank_str = "N/A"

        if kpi_lower in ["prs validation", "6s_audit", "final score"]:
            best_value = sorted_df.iloc[0]["cum"] if not sorted_df.empty else 0
            delta = cum_value - best_value
        elif kpi_lower in ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]:
            best_value = sorted_df.iloc[0]["cum"] if not sorted_df.empty else 0
            delta = cum_value - best_value
        else:
            best_value = sorted_df.iloc[0]["cum"] if not sorted_df.empty else 0
            delta = cum_value - best_value
        emoticon = get_trend_emoticon(kpi, delta)
        range_comment = get_range_comment(lang, selected_week_range[0], latest_week if latest_week else selected_week_range[1])
        line1 = f"{cum_value:.2f}{kpi_unit}"
        line2 = rank_str
        if best_value != 0 and pd.notna(delta):
            line3 = f"{emoticon}{delta:+.2f}{kpi_unit} {range_comment}"
        else:
            line3 = ""
        full_text = f"{line1}<br>{line2}<br>{line3}"
        render_custom_metric(cols_total[i % 3], kpi_display_name, full_text, "")
        i += 1

# --------------------------------------------------
# 11. Detailed Data Table (행=주차, 열=KPI)
# --------------------------------------------------
st.markdown("")
st.markdown(trans["detailed_data"][lang])
kpi_all = sorted(df["KPI"].unique())
all_weeks = sorted(df["Week_num"].dropna().unique())
data_table = {}
for kpi in kpi_all:
    kpi_unit = get_kpi_unit(kpi)
    row_data = {}
    values = []
    finals = []
    for w in all_weeks:
        if selected_team_detail != "HWK Total":
            sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
            if not sub_df.empty:
                val = sub_df.iloc[0]["Actual_numeric"]
                final_val = sub_df.iloc[0]["Final"]
                formatted = f"{val:.2f}{kpi_unit}<br>({final_val} point)"
                row_data[f"Week {int(w)}"] = formatted
                values.append(val)
                finals.append(final_val)
            else:
                row_data[f"Week {int(w)}"] = "N/A"
        else:
            sub_df = df[(df["KPI"] == kpi) & (df["Week_num"] == w)]
            if not sub_df.empty:
                val = sub_df["Actual_numeric"].mean()
                final_val = sub_df["Final"].mean()
                formatted = f"{val:.2f}{kpi_unit}<br>({final_val:.2f} point)"
                row_data[f"Week {int(w)}"] = formatted
                values.append(val)
                finals.append(final_val)
            else:
                row_data[f"Week {int(w)}"] = "N/A"
    if values:
        avg_val = sum(values) / len(values)
        avg_final = sum(finals) / len(finals) if finals else 0
        row_data["Average"] = f"{avg_val:.2f}{kpi_unit}<br>({avg_final:.2f} point)"
    else:
        row_data["Average"] = "N/A"
    data_table[kpi] = row_data
table_df = pd.DataFrame(data_table)
index_order = [f"Week {int(w)}" for w in all_weeks] + ["Average"]
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
rename_cols = {}
for col in table_df.columns:
    rename_cols[col] = get_kpi_display_name(col, lang)
table_df.rename(columns=rename_cols, inplace=True)
st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)
