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
    """선택한 언어에 맞게 KPI명을 반환"""
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
    """CSV에서 데이터 불러오기"""
    df = pd.read_csv("score.csv", sep="\t", encoding="utf-8")
    return df

def convert_to_numeric(x):
    """문자열 수치를 float로 변환, 불가능하면 np.nan"""
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
    """KPI에 맞는 단위 반환"""
    return KPI_UNITS.get(kpi_name.lower(), "")

def cumulative_performance(sub_df, kpi):
    """특정 구간(sub_df)에 대한 KPI 평균(혹은 누적) 계산 예시"""
    return sub_df["Actual_numeric"].mean()

def get_weekly_value_color(kpi, weekly_value, avg_value):
    """주별 수치와 전체 평균 비교 후 색상 결정"""
    positive_better = ["prs validation", "6s_audit", "final score"]
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
    """증감(delta)에 따라 😀/😡 이모티콘 표시"""
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
    """메트릭(카드형) UI를 직접 HTML로 렌더링"""
    html_metric = f"""
    <div style="font-size:14px; margin:5px; padding:5px;">
      <div style="font-weight:bold;">{label}</div>
      <div>{value}</div>
      <div style="color:{color};">{delta_str}</div>
    </div>
    """
    col.markdown(html_metric, unsafe_allow_html=True)

def format_final_label(row):
    """최종점수를 바 차트용 라벨로 변환"""
    return f"{row['Final']:.0f} point"

def get_range_comment(lang_code, start_week, end_week):
    """증감 비교 시, 범위를 나타내는 문자열 반환"""
    if lang_code == "ko":
        return f"({start_week}주차~{end_week}주차 평균 대비)"
    elif lang_code == "vi":
        return f"(Từ Tuần {start_week} đến Tuần {end_week} trung bình so với)"
    else:
        return f"(From Week {start_week} to Week {end_week} average compared to)"

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
# 5. 사이드바 위젯 (필터) - 주차 범위, KPI, 팀 선택
# --------------------------------------------------
st.sidebar.header("Filter Options")

# KPI 목록
kpi_options = sorted(list(df["KPI"].unique()))
# 'Final score'가 없다면 추가
if "Final score" not in kpi_options:
    kpi_options.append("Final score")
selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)

# 팀 목록
team_list = sorted(df["Team"].unique())
team_list_extended = team_list.copy()
if "HWK Total" not in team_list_extended:
    team_list_extended.append("HWK Total")
selected_teams = st.sidebar.multiselect(
    trans["select_teams"][lang],
    options=team_list_extended,
    default=team_list
)

# 주차 범위 (min/max)
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

# 주차 범위 역순 방지 (시작 주차 > 끝 주차일 경우 정렬)
start_week, end_week = sorted(selected_week_range)

# --------------------------------------------------
# 6. KPI/주차 범위 필터링
# --------------------------------------------------
if selected_kpi == "Final score":
    df_filtered = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= end_week)].copy()
else:
    df_filtered = df[(df["KPI"] == selected_kpi) &
                     (df["Week_num"] >= start_week) &
                     (df["Week_num"] <= end_week)].copy()

# 필터 결과가 없으면 종료
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

if selected_kpi == "Final score":
    # 주차 범위 내에서 Final score를 팀별로 합산
    df_latest = df_filtered.groupby("Team").agg({"Final": "sum"}).reset_index()
    df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
    # "HWK Total"을 선택했다면 전체 합계를 한 행으로 추가
    if "HWK Total" in selected_teams:
        overall_final = df_latest["Final"].sum()
        df_total = pd.DataFrame({
            "Team": ["HWK Total"],
            "Final": [overall_final]
        })
        df_total["Label"] = df_total.apply(format_final_label, axis=1)
        df_latest = pd.concat([df_latest, df_total], ignore_index=True)

else:
    # 최신 주(latest_week)에 해당하는 데이터만 사용
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

# 팀 필터 적용
df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()

# 바 차트 생성
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
        if row["KPI"].lower() == "shortage_cost":
            return f"{k_unit}{val:.2f} ({fin} point)"
        else:
            return f"{val:.2f}{k_unit} ({fin} point)"

    if not df_comp.empty and "KPI" in df_comp.columns:
        df_comp["Label"] = df_comp.apply(make_bar_label, axis=1)
    else:
        df_comp["Label"] = ""

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
    # 주차별 Final score 누적합
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

    # HWK Total이 포함되면 전체 합계 라인 추가
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
    # b-grade는 y축 라벨을 별도 처리
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
    fig_line.update_xaxes(tickmode='linear', tick0=start_week, dtick=1)

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

# df_comp가 비어있지 않은지 확인
if df_comp.empty:
    st.warning("선택된 팀에 대한 데이터가 없습니다. Top/Bottom 분석을 건너뜁니다.")
else:
    # HWK Total 제외
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

    # 라벨 구성
    if selected_kpi == "Final score":
        top_df["Label"] = top_df.apply(lambda row: format_final_label(row), axis=1)
        bottom_df["Label"] = bottom_df.apply(lambda row: format_final_label(row), axis=1)
    else:
        def make_bar_label2(row):
            k_unit = get_kpi_unit(row["KPI"])
            val = row["Actual_numeric"]
            fin = row["Final"]
            if row["KPI"].lower() == "shortage_cost":
                return f"{k_unit}{val:.2f} ({fin} point)"
            else:
                return f"{val:.2f}{k_unit} ({fin} point)"
        top_df["Label"] = top_df.apply(make_bar_label2, axis=1)
        bottom_df["Label"] = bottom_df.apply(make_bar_label2, axis=1)

    col1, col2 = st.columns(2)

    # (개별 팀 선택 시) 전체 구간에서의 누적/평균 기준 랭킹
    if selected_team_detail != "HWK Total":
        team_cum = df[(df["KPI"] == selected_kpi) &
                      (df["Week_num"] >= start_week) &
                      (df["Week_num"] <= end_week)] \
                    .groupby("Team") \
                    .apply(lambda x: cumulative_performance(x, selected_kpi)) \
                    .reset_index(name="cum")

        # KPI에 따라 정렬 ascending 여부 결정
        if selected_kpi.lower() in ["prs validation", "6s_audit", "final score"]:
            sorted_df = team_cum.sort_values("cum", ascending=False).reset_index(drop=True)
        elif selected_kpi.lower() in ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]:
            sorted_df = team_cum.sort_values("cum", ascending=True).reset_index(drop=True)
        else:
            sorted_df = team_cum.sort_values("cum", ascending=False).reset_index(drop=True)

        # 동률 처리를 위한 랭킹 계산
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
                rank_str = '<span style="color:blue;">Top 1</span>'
            elif selected_rank == 7:
                rank_str = '<span style="color:red;">Top 7</span>'
            else:
                rank_str = f"Top {selected_rank}"
        else:
            rank_str = "N/A"
    else:
        rank_str = ""

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

# (A) 팀 데이터 준비
# "HWK Total"일 경우, 주차 범위 내 전체 데이터 df_cum 정의
if selected_team_detail == "HWK Total":
    df_cum = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= end_week)]
    # 마지막 주 기준 팀별 평균
    df_team = (
        df_filtered.groupby("KPI")
        .agg({"Actual_numeric": "mean", "Final": "mean", "Actual": "first"})
        .reset_index()
    )
else:
    # 특정 팀
    df_cum = df[
        (df["Team"] == selected_team_detail) &
        (df["Week_num"] >= start_week) &
        (df["Week_num"] <= end_week)
    ]
    # 최신 주 데이터만 추출
    df_team = df_cum[df_cum["Week_num"] == latest_week].copy()

# (A-1) 마지막 주 성과 상세보기
if latest_week is not None:
    st.markdown(
        f"<div style='font-size:18px; font-weight:bold;'>"
        f"{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}"
        f"</div>",
        unsafe_allow_html=True
    )

    # df_team이 비어있는 경우
    if df_team.empty:
        st.warning(f"{selected_team_detail} 팀은 최신 주({latest_week}주차)에 데이터가 없습니다.")
    else:
        cols = st.columns(3)
        i = 0

        kpi_list_for_team = df_team["KPI"].unique()
        for kpi in kpi_list_for_team:
            kpi_lower = kpi.lower()
            kpi_unit = get_kpi_unit(kpi)

            def format_value_with_unit(val, unit):
                # 단위가 %인데 값 끝에 %가 없으면 붙여준다
                if unit == "%" and not f"{val:.2f}".endswith("%"):
                    return f"{val:.2f}{unit}"
                return f"{val:.2f}{unit}"

            # HWK Total & shortage_cost
            if selected_team_detail == "HWK Total" and kpi_lower == "shortage_cost":
                df_cum_sc = df_cum[df_cum["KPI"].str.lower() == "shortage_cost"]
                if not df_cum_sc.empty:
                    cum_value = df_cum_sc["Actual_numeric"].mean()
                else:
                    cum_value = np.nan

                if latest_week > start_week:
                    df_cum_prev_sc = df[
                        (df["Week_num"] >= start_week) &
                        (df["Week_num"] < latest_week) &
                        (df["KPI"].str.lower() == "shortage_cost")
                    ]
                    prev_avg = df_cum_prev_sc["Actual_numeric"].mean() if not df_cum_prev_sc.empty else np.nan
                else:
                    prev_avg = None

                delta = cum_value - prev_avg if prev_avg is not None else None
                emoticon = get_trend_emoticon(kpi, delta)
                range_comment = get_range_comment(lang, start_week, latest_week if latest_week else end_week)
                line1 = f"{kpi_unit}{cum_value:.2f}"
                if delta is not None:
                    line2 = f"{emoticon}{kpi_unit}{delta:+.2f} {range_comment}"
                else:
                    line2 = "N/A"
                full_text = f"{line1}<br>{line2}"
                render_custom_metric(cols[i % 3], get_kpi_display_name(kpi, lang), full_text, "")
                i += 1
                continue

            # 일반 케이스
            if selected_team_detail != "HWK Total":
                df_last = df_team[(df_team["Week_num"] == latest_week) & (df_team["KPI"] == kpi)]
                df_prev = df_cum[(df_cum["Week_num"] == (latest_week - 1)) & (df_cum["KPI"] == kpi)]
            else:
                # HWK Total인 경우, df_team에는 마지막 주 평균, 이전 주는 df에서 동일 방식으로
                df_last = df_team[df_team["KPI"] == kpi]
                df_prev_raw = (
                    df[df["Week_num"] == (latest_week - 1)]
                    .groupby("KPI")
                    .agg({"Actual_numeric": "mean", "Final": "mean", "Actual": "first"})
                    .reset_index()
                )
                df_prev = df_prev_raw[df_prev_raw["KPI"] == kpi]

            if df_last.empty:
                # 해당 KPI 데이터가 없으면 스킵
                continue

            row_last = df_last.iloc[0]
            curr_val_str = format_value_with_unit(row_last["Actual_numeric"], kpi_unit)
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
                    delta_str = f"{emoticon}{format_value_with_unit(delta_actual, kpi_unit)}({delta_final:+d} point)"
                else:
                    delta_str = "N/A"
            else:
                delta_str = "N/A"

            render_custom_metric(cols[i % 3], get_kpi_display_name(kpi, lang), current_label, delta_str)
            i += 1

# (B) 전체 주차 누적(또는 평균) 성과 상세
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
    cols_total = st.columns(3)
    i = 0

    for kpi in df_cum_group["KPI"].unique():
        kpi_lower = kpi.lower()
        kpi_unit = get_kpi_unit(kpi)
        kpi_display_name = get_kpi_display_name(kpi, lang)

        # HWK Total & shortage_cost 처리
        if selected_team_detail == "HWK Total" and kpi_lower == "shortage_cost":
            df_cum_sc = df_cum[df_cum["KPI"].str.lower() == "shortage_cost"]
            if not df_cum_sc.empty:
                cum_value = df_cum_sc["Actual_numeric"].mean()
            else:
                cum_value = np.nan

            if latest_week > start_week:
                df_cum_prev_sc = df[
                    (df["Week_num"] >= start_week) &
                    (df["Week_num"] < latest_week) &
                    (df["KPI"].str.lower() == "shortage_cost")
                ]
                prev_avg = df_cum_prev_sc["Actual_numeric"].mean() if not df_cum_prev_sc.empty else np.nan
            else:
                prev_avg = None

            delta = cum_value - prev_avg if prev_avg is not None else None
            emoticon = get_trend_emoticon(kpi, delta)
            range_comment = get_range_comment(lang, start_week, latest_week if latest_week else end_week)
            line1 = f"{cum_value:.2f}{kpi_unit}"
            if delta is not None:
                line2 = f"{emoticon}{delta:+.2f}{kpi_unit} {range_comment}"
            else:
                line2 = "N/A"
            full_text = f"{line1}<br>{line2}"
            render_custom_metric(cols_total[i % 3], kpi_display_name, full_text, "")
            i += 1
            continue

        # 일반 케이스
        sub_df = df_cum[df_cum["KPI"] == kpi]
        cum_value = cumulative_performance(sub_df, kpi)

        # 전체 팀 대비 rank
        team_cum = df[(df["KPI"] == kpi) &
                      (df["Week_num"] >= start_week) &
                      (df["Week_num"] <= end_week)] \
                    .groupby("Team") \
                    .apply(lambda x: cumulative_performance(x, kpi)) \
                    .reset_index(name="cum")

        # KPI별 ascending 여부
        if kpi_lower in ["prs validation", "6s_audit", "final score"]:
            sorted_df = team_cum.sort_values("cum", ascending=False).reset_index(drop=True)
        elif kpi_lower in ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]:
            sorted_df = team_cum.sort_values("cum", ascending=True).reset_index(drop=True)
        else:
            sorted_df = team_cum.sort_values("cum", ascending=False).reset_index(drop=True)

        # 랭킹 계산
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
                rank_str = '<span style="color:blue;">Top 1</span>'
            elif selected_rank == 7:
                rank_str = '<span style="color:red;">Top 7</span>'
            else:
                rank_str = f"Top {selected_rank}"
        else:
            rank_str = "N/A"

        # best_value와의 차이(delta) 계산
        if not sorted_df.empty:
            best_value = sorted_df.iloc[0]["cum"]
        else:
            best_value = None

        if best_value is not None:
            delta = cum_value - best_value
        else:
            delta = None

        emoticon = get_trend_emoticon(kpi, delta)
        range_comment = get_range_comment(lang, start_week, latest_week if latest_week else end_week)
        line1 = f"{cum_value:.2f}{kpi_unit}"
        line2 = rank_str
        if best_value is not None and pd.notna(delta):
            line3 = f"{emoticon}{delta:+.2f}{kpi_unit} {range_comment}"
        else:
            line3 = ""

        full_text = f"{line1}<br>{line2}<br>{line3}"
        render_custom_metric(cols_total[i % 3], kpi_display_name, full_text, "")
        i += 1

# --------------------------------------------------
# 11. Detailed Data Table (행=주차, 열=KPI)
# --------------------------------------------------
st.markdown(trans["detailed_data"][lang])

kpi_all = sorted(df["KPI"].unique())
all_weeks = sorted(df["Week_num"].dropna().unique())

data_table = {}
for kpi in kpi_all:
    kpi_unit = get_kpi_unit(kpi)
    row_data = {}
    week_values = {}
    weekly_finals = {}

    for w in all_weeks:
        if selected_team_detail != "HWK Total":
            sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
            if not sub_df.empty:
                val = sub_df.iloc[0]["Actual_numeric"]
                final_val = sub_df.iloc[0]["Final"]
                week_values[w] = val
                weekly_finals[w] = final_val
            else:
                week_values[w] = None
                weekly_finals[w] = None
        else:
            # HWK Total이면 팀 구분 없이 평균값
            sub_df = df[(df["KPI"] == kpi) & (df["Week_num"] == w)]
            if not sub_df.empty:
                val = sub_df["Actual_numeric"].mean()
                final_val = sub_df["Final"].mean()
                week_values[w] = val
                weekly_finals[w] = final_val
            else:
                week_values[w] = None
                weekly_finals[w] = None

    valid_values = [v for v in week_values.values() if v is not None]
    avg_val = sum(valid_values) / len(valid_values) if valid_values else None

    valid_finals = [f for f in weekly_finals.values() if f is not None]
    avg_final = sum(valid_finals) / len(valid_finals) if valid_finals else None

    # 주차별 값 포매팅
    for w in all_weeks:
        val = week_values[w]
        final_val = weekly_finals[w]
        if val is not None and avg_val is not None:
            color = get_weekly_value_color(kpi, val, avg_val)
            formatted = f'<span style="color:{color};">{val:.2f}{kpi_unit}</span><br>({final_val:.1f} point)'
        else:
            formatted = "N/A"
        row_data[f"Week {int(w)}"] = formatted

    # 평균 행
    if avg_val is not None and avg_final is not None:
         row_data["Average"] = f"{avg_val:.2f}{kpi_unit}<br>({avg_final:.1f} point)"
    else:
         row_data["Average"] = "N/A"

    data_table[kpi] = row_data

table_df = pd.DataFrame(data_table)
index_order = [f"Week {int(w)}" for w in all_weeks] + ["Average"]
table_df = table_df.reindex(index_order)

# 인덱스(행)명 다국어 변환
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

# 열(KPI)명 다국어 변환
rename_cols = {}
for col in table_df.columns:
    rename_cols[col] = get_kpi_display_name(col, lang)
table_df.rename(columns=rename_cols, inplace=True)

# HTML로 렌더링
st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)
