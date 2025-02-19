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
        "vi": "Phân tích xu hướng hiệu suất hàng tuần"
    },
    "top_bottom_rankings": {
        "en": "KPI Top/Bottom Team Rankings",
        "ko": "KPI 상위/하위 팀 순위",
        "vi": "Xếp hạng Nhóm KPI Cao/Thấp Nhất"
    },
    "last_week_details": {
        "en": "Last Week performance Details for {team} (Week {week})",
        "ko": "지난주 성과 상세보기: {team} (Week {week})",
        "vi": "Chi tiết hiệu suất tuần trước của {team} (Tuần {week})"
    },
    "total_week_details": {
        "en": "Total Week Performance Detail for {team} (All weeks)",
        "ko": "전체 주차 누적 실적 상세: {team} (All weeks)",
        "vi": "Chi tiết hiệu suất tổng tuần của {team} (Tất cả các tuần)"
    },
    "detailed_data": {
        "en": "Detailed Data for Selected Team",
        "ko": "선택된 팀의 상세 데이터",
        "vi": "Dữ liệu chi tiết cho nhóm đã chọn"
    },
    "select_kpi": {
        "en": "Select KPI",
        "ko": "KPI 선택",
        "vi": "Chọn KPI"
    },
    "select_teams": {
        "en": "Select Teams for Comparison",
        "ko": "비교할 팀 선택 (HWK Total 포함)",
        "vi": "Chọn nhóm để so sánh"
    },
    "select_team_details": {
        "en": "Select Team for Details",
        "ko": "상세 조회할 팀 선택 (HWK Total 포함)",
        "vi": "Chọn nhóm để xem chi tiết"
    },
    "select_week_range": {
        "en": "Select Week Range",
        "ko": "주차 범위 선택",
        "vi": "Chọn phạm vi tuần"
    },
    "language": {
        "en": "Language",
        "ko": "언어",
        "vi": "ngôn ngữ"
    },
    "avg_by_team": {
        "en": "Average {kpi} by Team",
        "ko": "팀별 {kpi} 평균",
        "vi": "Trung bình {kpi} theo nhóm"
    },
    "weekly_trend_title": {
        "en": "Weekly Trend of {kpi}",
        "ko": "{kpi} 주간 추이",
        "vi": "Xu hướng hàng tuần của {kpi}"
    },
    "top_teams": {
        "en": "Top {n} Teams - {kpi}",
        "ko": "{kpi} 상위 {n} 팀",
        "vi": "Top {n} nhóm - {kpi}"
    },
    "bottom_teams": {
        "en": "Bottom {n} Teams - {kpi}",
        "ko": "{kpi} 하위 {n} 팀",
        "vi": "Nhóm {n} thấp nhất - {kpi}"
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
    "5 prs validation": "%",
    "6s_audit": "%",
    "aql_performance": "%",
    "b-grade": "%",
    "attendance": "%",
    "issue_tracking": "minutes",
    "shortage_cost": "$",
    "final score": ""
}

# KPI별 한글 표기 매핑
KPI_NAME_MAP = {
    "5 prs validation": {
        "ko": "포장 완료 제품 5족 품질 검증 통과율",
        "en": "5 prs validation",
        "vi": "5 prs validation"
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
# 2. 우측 상단 언어 선택
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

def aggregator_for_kpi(df_sub: pd.DataFrame, kpi_name: str) -> float:
    kpi_lower = kpi_name.lower()
    if kpi_lower == "final score":
        return df_sub["Final"].sum()
    elif kpi_lower == "shortage_cost":
        return df_sub["Actual_numeric"].sum()
    else:
        return df_sub["Actual_numeric"].mean()

def cumulative_performance(sub_df, kpi):
    kpi_lower = kpi.lower()
    if kpi_lower == "final score":
        return sub_df["Final"].sum()
    elif kpi_lower == "shortage_cost":
        return sub_df["Actual_numeric"].sum()
    else:
        return sub_df["Actual_numeric"].mean()

def get_weekly_value_color(kpi, weekly_value, avg_value):
    positive_better = ["5 prs validation", "6s_audit", "final score"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
    if weekly_value is None or avg_value is None:
        return "black"
    if kpi.lower() in positive_better:
        return "blue" if weekly_value >= avg_value else "red"
    elif kpi.lower() in negative_better:
        return "blue" if weekly_value <= avg_value else "red"
    else:
        return "blue" if weekly_value >= avg_value else "red"

def get_trend_emoticon(kpi, delta):
    """실적 개선: 👍, 악화: 🥵"""
    if delta is None:
        return ""
    kpi_lower = kpi.lower()
    positive_better = ["5 prs validation", "6s_audit", "final score"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
    if kpi_lower in positive_better:
        if delta > 0:
            return "👍"
        elif delta < 0:
            return "🥵"
        else:
            return ""
    elif kpi_lower in negative_better:
        if delta < 0:
            return "👍"
        elif delta > 0:
            return "🥵"
        else:
            return ""
    else:
        if delta > 0:
            return "👍"
        elif delta < 0:
            return "🥵"
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
        return f"({start_week}주차~{end_week}주차 평균 대비)"
    elif lang_code == "vi":
        return f"(Từ Tuần {start_week} đến Tuần {end_week} trung bình so với)"
    else:
        return f"(From Week {start_week} to Week {end_week} average compared to)"

def format_value_with_unit(val, unit):
    if pd.isna(val):
        return "N/A"
    if unit == "$":
        return f"${val:.2f}"
    elif unit == "%" and not f"{val:.2f}".endswith("%"):
        return f"{val:.2f}{unit}"
    else:
        return f"{val:.2f}{unit}"

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
if "5 prs validation" not in kpi_options:
    kpi_options.append("5 prs validation")

selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)

team_list = sorted(df["Team"].unique())
team_list_extended = team_list.copy()
if "HWK Total" not in team_list_extended:
    team_list_extended.append("HWK Total")
selected_teams = st.sidebar.multiselect(
    trans["select_teams"][lang],
    options=team_list_extended,
    default=team_list
)

min_week = int(df["Week_num"].min())
max_week = int(df["Week_num"].max())
selected_week_range = st.sidebar.slider(
    trans["select_week_range"][lang],
    min_week,
    max_week,
    (min_week, max_week),
    step=1
)

selected_team_detail = st.sidebar.selectbox(
    trans["select_team_details"][lang],
    options=team_list_extended,
    index=0
)

start_week, end_week = sorted(selected_week_range)

# --------------------------------------------------
# 6. KPI/주차 범위 필터링
# --------------------------------------------------
kpi_lower = selected_kpi.lower()
if kpi_lower == "final score":
    df_filtered = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= end_week)].copy()
else:
    df_filtered = df[
        (df["KPI"].str.lower() == kpi_lower) &
        (df["Week_num"] >= start_week) &
        (df["Week_num"] <= end_week)
    ].copy()

if df_filtered.empty:
    st.warning("선택한 필터에 해당하는 데이터가 없습니다.")
    st.stop()

latest_week = df_filtered["Week_num"].max()
if pd.isna(latest_week):
    st.warning("해당 주차 범위에 데이터가 없습니다.")
    st.stop()
else:
    latest_week = int(latest_week)

# --------------------------------------------------
# 7. [1] KPI Performance Comparison by Team (바 차트)
# --------------------------------------------------
st.markdown(trans["kpi_comparison"][lang])

df_bar = df_filtered.groupby("Team").apply(lambda x: aggregator_for_kpi(x, selected_kpi)).reset_index(name="Value")
if "HWK Total" in selected_teams:
    if kpi_lower == "final score":
        total_val = df_filtered["Final"].sum()
    elif kpi_lower == "shortage_cost":
        total_val = df_filtered["Actual_numeric"].sum()
    else:
        total_val = df_filtered["Actual_numeric"].mean()
    df_total = pd.DataFrame({"Team": ["HWK Total"], "Value": [total_val]})
    df_bar = pd.concat([df_bar, df_total], ignore_index=True)

df_bar = df_bar[df_bar["Team"].isin(selected_teams)].copy()

def make_bar_label(team, val, kpi_name):
    k_unit = get_kpi_unit(kpi_name)
    if kpi_name.lower() == "final score":
        return f"{val:.0f} point"
    elif kpi_name.lower() == "shortage_cost":
        return f"${val:.2f}"
    else:
        if k_unit == "%":
            return f"{val:.2f}{k_unit}"
        else:
            return f"{val:.2f}{k_unit}"

df_bar["Label"] = df_bar.apply(lambda row: make_bar_label(row["Team"], row["Value"], selected_kpi), axis=1)

if kpi_lower == "final score":
    fig_bar = px.bar(
        df_bar,
        x="Team",
        y="Value",
        text="Label",
        labels={"Value": "Final score (Sum in selected range)"}
    )
else:
    fig_bar = px.bar(
        df_bar,
        x="Team",
        y="Value",
        text="Label",
        labels={"Value": trans["avg_by_team"][lang].format(kpi=selected_kpi)}
    )

fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# --------------------------------------------------
# 8. [2] Weekly Performance Trend Analysis (라인 차트)
# --------------------------------------------------
st.markdown(trans["weekly_trend"][lang])

if kpi_lower == "final score":
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
    fig_line.update_xaxes(tickmode='linear', tick0=start_week, dtick=1)

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

elif kpi_lower == "shortage_cost":
    df_trend_individual = df_filtered.groupby(["Team", "Week_num"]).agg({"Actual_numeric": "sum"}).reset_index()
    fig_line = px.line(
        df_trend_individual[df_trend_individual["Team"].isin([t for t in selected_teams if t != "HWK Total"])],
        x="Week_num",
        y="Actual_numeric",
        color="Team",
        markers=True,
        labels={"Week_num": "Week", "Actual_numeric": "Shortage cost (Sum)"},
        title=trans["weekly_trend_title"][lang].format(kpi=selected_kpi)
    )
    fig_line.update_xaxes(tickmode='linear', tick0=start_week, dtick=1)

    if "HWK Total" in selected_teams:
        df_overall_trend = df_filtered.groupby("Week_num").agg({"Actual_numeric": "sum"}).reset_index()
        fig_line.add_scatter(
            x=df_overall_trend["Week_num"],
            y=df_overall_trend["Actual_numeric"],
            mode='lines+markers',
            name="HWK Total",
            line=dict(color='black', dash='dash')
        )

else:
    df_trend_individual = df_filtered.groupby(["Team", "Week_num"]).agg({"Actual_numeric": "mean"}).reset_index()
    fig_line = px.line(
        df_trend_individual[df_trend_individual["Team"].isin([t for t in selected_teams if t != "HWK Total"])],
        x="Week_num",
        y="Actual_numeric",
        color="Team",
        markers=True,
        labels={"Week_num": "Week", "Actual_numeric": f"{selected_kpi} Value"},
        title=trans["weekly_trend_title"][lang].format(kpi=selected_kpi)
    )
    fig_line.update_xaxes(tickmode='linear', tick0=start_week, dtick=1)

    if "HWK Total" in selected_teams:
        df_overall_trend = df_filtered.groupby("Week_num").agg({"Actual_numeric": "mean"}).reset_index()
        fig_line.add_scatter(
            x=df_overall_trend["Week_num"],
            y=df_overall_trend["Actual_numeric"],
            mode='lines+markers',
            name="HWK Total",
            line=dict(color='black', dash='dash')
        )

st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

# --------------------------------------------------
# 9. [3] KPI Top/Bottom Team Rankings
# --------------------------------------------------
st.markdown(trans["top_bottom_rankings"][lang])

df_rank_base = df_filtered.copy()
if df_rank_base.empty:
    st.warning("Top/Bottom 분석을 위한 데이터가 없습니다.")
else:
    df_rank_agg = df_rank_base.groupby("Team").apply(lambda x: cumulative_performance(x, selected_kpi)).reset_index(name="cum")
    df_rank_agg = df_rank_agg[df_rank_agg["Team"] != "HWK Total"]

    if kpi_lower in ["5 prs validation", "6s_audit", "final score"]:
        df_rank_agg.sort_values("cum", ascending=False, inplace=True)
    elif kpi_lower in ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]:
        df_rank_agg.sort_values("cum", ascending=True, inplace=True)
    else:
        df_rank_agg.sort_values("cum", ascending=False, inplace=True)

    top_n = 3 if len(df_rank_agg) >= 3 else len(df_rank_agg)
    bottom_n = 3 if len(df_rank_agg) >= 3 else len(df_rank_agg)

    top_df = df_rank_agg.head(top_n).copy()
    bottom_df = df_rank_agg.tail(bottom_n).copy()

    if kpi_lower in ["5 prs validation", "6s_audit", "final score"]:
        bottom_df = bottom_df.sort_values("cum", ascending=True)
    else:
        bottom_df = bottom_df.sort_values("cum", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(trans["top_teams"][lang].format(n=top_n, kpi=selected_kpi))
        fig_top = px.bar(
            top_df,
            x="cum",
            y="Team",
            orientation="h",
            text="cum",
            labels={"cum": f"Aggregated {selected_kpi}", "Team": "Team"}
        )
        fig_top.update_traces(texttemplate="%{text:.2f}", textposition='inside')
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True, key="top_chart")

    with col2:
        st.subheader(trans["bottom_teams"][lang].format(n=bottom_n, kpi=selected_kpi))
        fig_bottom = px.bar(
            bottom_df,
            x="cum",
            y="Team",
            orientation="h",
            text="cum",
            labels={"cum": f"Aggregated {selected_kpi}", "Team": "Team"}
        )
        fig_bottom.update_traces(texttemplate="%{text:.2f}", textposition='inside', marker_color='red')
        fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# --------------------------------------------------
# 10. [4] Team-Specific KPI Detailed View (카드형 레이아웃)
# --------------------------------------------------
st.markdown("")

# (A) 공통 df_cum 생성
if selected_team_detail == "HWK Total":
    df_cum = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= end_week)]
    # ======== (수정) HWK Total 선택 시 shortage_cost는 합산, 그 외는 평균 ========
    df_team = (
        df_cum[df_cum["Week_num"] == latest_week]
        .groupby("KPI")
        .apply(lambda g: pd.Series({
            "Actual_numeric": (
                g["Actual_numeric"].sum()
                if g.name.lower() == "shortage_cost"
                else g["Actual_numeric"].mean()
            ),
            "Final": g["Final"].mean(),
            "Actual": g["Actual"].iloc[0] if not g.empty else None
        }))
        .reset_index()
    )
else:
    df_cum = df[
        (df["Team"] == selected_team_detail) &
        (df["Week_num"] >= start_week) &
        (df["Week_num"] <= end_week)
    ]
    df_team = df_cum[df_cum["Week_num"] == latest_week].copy()

# (B) 지난주 성과 상세보기
if latest_week is not None:
    st.markdown(
        f"<div style='font-size:18px; font-weight:bold;'>"
        f"{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}"
        f"</div>",
        unsafe_allow_html=True
    )

    if df_team.empty:
        st.warning(f"{selected_team_detail} 팀은 최신 주({latest_week}주차)에 데이터가 없습니다.")
    else:
        kpi_list_for_team = sorted(df_team["KPI"].unique(), key=str.lower)
        cols = st.columns(3)
        i = 0

        for kpi in kpi_list_for_team:
            kpi_lower = kpi.lower()
            kpi_unit = get_kpi_unit(kpi)

            # ----- HWK Total일 때: 2줄만 표시 (현재 값, Δ) -----
            if selected_team_detail == "HWK Total":
                row_kpi = df_team[df_team["KPI"] == kpi]
                if row_kpi.empty:
                    continue
                val_last = row_kpi.iloc[0]["Actual_numeric"]
                final_last = row_kpi.iloc[0]["Final"]

                # 이전 주(Week-1) 대비 값
                if latest_week > start_week:
                    df_prev_sc = df_cum[df_cum["Week_num"] == (latest_week - 1)]
                    df_prev_sc = df_prev_sc[df_prev_sc["KPI"].str.lower() == kpi_lower]
                    # shortage_cost는 합산, 그 외에는 평균
                    if kpi_lower == "shortage_cost":
                        val_prev = df_prev_sc["Actual_numeric"].sum() if not df_prev_sc.empty else np.nan
                    else:
                        val_prev = df_prev_sc["Actual_numeric"].mean() if not df_prev_sc.empty else np.nan
                else:
                    val_prev = np.nan

                # Line1: 현재 값
                curr_val_str = format_value_with_unit(val_last, kpi_unit)
                if pd.notna(final_last):
                    line1 = f"{curr_val_str} ({int(round(final_last))} point)"
                else:
                    line1 = curr_val_str

                # Line2: 이전 주 대비 증감 (Δ)
                if (not np.isnan(val_last)) and (not np.isnan(val_prev)):
                    delta_actual = val_last - val_prev
                    emoticon = get_trend_emoticon(kpi, delta_actual)
                    line2 = f"{emoticon}{format_value_with_unit(delta_actual, kpi_unit)}"
                else:
                    line2 = "N/A"

                full_text = f"{line1}<br>{line2}"
                render_custom_metric(cols[i % 3], get_kpi_display_name(kpi, lang), full_text, "")
                i += 1
                continue

            # ----- 일반 팀일 때: 3줄 (현재 값, 순위, Δ) -----
            df_last = df_team[(df_team["Week_num"] == latest_week) & (df_team["KPI"] == kpi)]
            if df_last.empty:
                continue
            row_last = df_last.iloc[0]
            val_last = row_last["Actual_numeric"]
            final_last = row_last["Final"]

            df_prev = df_cum[(df_cum["Week_num"] == (latest_week - 1)) & (df_cum["KPI"] == kpi)]
            if df_prev.empty:
                val_prev = np.nan
            else:
                val_prev = df_prev["Actual_numeric"].mean()

            # (1) 현재 값
            curr_val_str = format_value_with_unit(val_last, kpi_unit)
            if pd.notna(final_last):
                line1 = f"{curr_val_str} ({int(round(final_last))} point)"
            else:
                line1 = curr_val_str

            # (2) 순위
            df_sw_all = df[(df["Week_num"] == latest_week) & (df["KPI"].str.lower() == kpi_lower)].copy()
            if not df_sw_all.empty:
                df_sw_agg = df_sw_all.groupby("Team").apply(lambda x: aggregator_for_kpi(x, kpi)).reset_index(name="val")
                # HWK Total 제외
                df_sw_agg = df_sw_agg[df_sw_agg["Team"] != "HWK Total"]

                if kpi_lower in ["5 prs validation", "6s_audit", "final score"]:
                    df_sw_agg.sort_values("val", ascending=False, inplace=True)
                elif kpi_lower in ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]:
                    df_sw_agg.sort_values("val", ascending=True, inplace=True)
                else:
                    df_sw_agg.sort_values("val", ascending=False, inplace=True)

                ranks = []
                current_rank = 1
                for idx2 in range(len(df_sw_agg)):
                    if idx2 == 0:
                        ranks.append(current_rank)
                    else:
                        if df_sw_agg.iloc[idx2]["val"] == df_sw_agg.iloc[idx2 - 1]["val"]:
                            ranks.append(current_rank)
                        else:
                            current_rank = idx2 + 1
                            ranks.append(current_rank)
                df_sw_agg["Rank"] = ranks

                selected_rank = None
                for idx2 in range(len(df_sw_agg)):
                    if df_sw_agg.iloc[idx2]["Team"] == selected_team_detail:
                        selected_rank = df_sw_agg.iloc[idx2]["Rank"]
                        break

                if selected_rank is not None:
                    if selected_rank == 1:
                        line2 = '<span style="color:blue;">Top 1</span>'
                    elif selected_rank == 7:
                        line2 = '<span style="color:red;">Top 7</span>'
                    else:
                        line2 = f"Top {int(selected_rank)}"
                else:
                    line2 = "N/A"
            else:
                line2 = "N/A"

            # (3) Δ
            if (not np.isnan(val_last)) and (not np.isnan(val_prev)):
                delta_actual = val_last - val_prev
                emoticon = get_trend_emoticon(kpi, delta_actual)
                if pd.notna(final_last):
                    # point 차이
                    df_prev_fin = df_prev["Final"].mean() if not df_prev.empty else np.nan
                    if pd.notna(df_prev_fin):
                        delta_final = int(round(final_last)) - int(round(df_prev_fin))
                        line3 = f"{emoticon}{format_value_with_unit(delta_actual, kpi_unit)}({delta_final:+d} point)"
                    else:
                        line3 = f"{emoticon}{format_value_with_unit(delta_actual, kpi_unit)}"
                else:
                    line3 = f"{emoticon}{format_value_with_unit(delta_actual, kpi_unit)}"
            else:
                line3 = "N/A"

            full_text = f"{line1}<br>{line2}<br>{line3}"
            render_custom_metric(cols[i % 3], get_kpi_display_name(kpi, lang), full_text, "")
            i += 1

# (C) 전체 주차 누적 실적 상세
st.markdown("")
st.markdown(
    f"<div style='font-size:18px; font-weight:bold;'>"
    f"{trans['total_week_details'][lang].format(team=selected_team_detail)}"
    f"</div>",
    unsafe_allow_html=True
)

if df_cum.empty:
    st.warning(f"{selected_team_detail} 팀은 선택한 주차 범위({start_week}~{end_week})에 데이터가 없습니다.")
else:
    df_cum_group = df_cum.groupby("KPI").apply(lambda x: cumulative_performance(x, x["KPI"].iloc[0])).reset_index(name="cum")
    kpi_list_for_cum = sorted(df_cum_group["KPI"].unique(), key=str.lower)

    cols_total = st.columns(3)
    i = 0

    for kpi in kpi_list_for_cum:
        kpi_lower = kpi.lower()
        kpi_unit = get_kpi_unit(kpi)
        kpi_display_name = get_kpi_display_name(kpi, lang)
        
        # --- HWK Total: 2줄 표시 (현재 값, Δ)
        if selected_team_detail == "HWK Total":
            current_df = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= latest_week) & (df["KPI"].str.lower() == kpi_lower)]
            previous_df = df[(df["Week_num"] >= start_week) & (df["Week_num"] < latest_week) & (df["KPI"].str.lower() == kpi_lower)]
            
            # 현재 구간 (start_week ~ latest_week)
            if not current_df.empty:
                if kpi_lower in ["final score", "5 prs validation"]:
                    current_avg = current_df["Final"].mean()
                elif kpi_lower == "shortage_cost":
                    # shortage_cost는 합산
                    current_avg = current_df["Actual_numeric"].sum()
                else:
                    current_avg = current_df["Actual_numeric"].mean()
            else:
                current_avg = np.nan

            # 이전 구간 (start_week ~ latest_week-1)
            if not previous_df.empty:
                if kpi_lower in ["final score", "5 prs validation"]:
                    previous_avg = previous_df
