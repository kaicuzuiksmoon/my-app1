# # # import streamlit as st
# # # import pandas as pd
# # # import plotly.express as px
# # # import numpy as np
# # # import re
# # # import os

# # # # --------------------------------------------------
# # # # 1. 페이지 설정 및 다국어 번역용 사전 정의
# # # # --------------------------------------------------
# # # st.set_page_config(page_title="HWK Quality competition Event", layout="wide")

# # # # 번역 사전 (영어/한글/베트남어)
# # # trans = {
# # #     "title": {
# # #         "en": "HWK Quality competition Event",
# # #         "ko": "HWK 품질 경쟁 이벤트",
# # #         "vi": "HWK sự kiện thi đua chất lượng"
# # #     },
# # #     "kpi_comparison": {
# # #         "en": "1. KPI Performance Comparison by Team",
# # #         "ko": "1. 팀별 KPI 성과 비교",
# # #         "vi": "So sánh Hiệu suất KPI theo Nhóm"
# # #     },
# # #     "weekly_trend": {
# # #         "en": "2. Weekly Performance Trend Analysis",
# # #         "ko": "2. 주간 성과 트렌드 분석",
# # #         "vi": "2. Phân tích Xu hướng Hiệu suất Hàng Tuần"
# # #     },
# # #     "top_bottom_rankings": {
# # #         "en": "3. KPI Top/Bottom Team Rankings",
# # #         "ko": "3. KPI 상위/하위 팀 순위",
# # #         "vi": "3. Xếp hạng Nhóm KPI Cao/Thấp Nhất"
# # #     },
# # #     "last_week_details": {
# # #         "en": "Last Week performance Details for {team} (Week {week})",
# # #         "ko": "지난주 성과 상세보기: {team} (Week {week})",
# # #         "vi": "Chi tiết Hiệu suất Tuần Trước của {team} (Tuần {week})"
# # #     },
# # #     "total_week_details": {
# # #         "en": "Total Week Performance Detail for {team} (All weeks)",
# # #         "ko": "전체 주차 누적 실적 상세: {team} (All weeks)",
# # #         "vi": "Chi tiết Hiệu suất Tổng Tuần của {team} (Tất cả các tuần)"
# # #     },
# # #     "detailed_data": {
# # #         "en": "Detailed Data for Selected Team",
# # #         "ko": "선택된 팀의 상세 데이터",
# # #         "vi": "Dữ liệu Chi tiết cho Nhóm Đã Chọn"
# # #     },
# # #     "select_kpi": {
# # #         "en": "Select KPI",
# # #         "ko": "KPI 선택",
# # #         "vi": "Chọn KPI"
# # #     },
# # #     "select_teams": {
# # #         "en": "Select Teams for Comparison",
# # #         "ko": "비교할 팀 선택 (HWK Total 포함)",
# # #         "vi": "Chọn Nhóm để So Sánh"
# # #     },
# # #     "select_team_details": {
# # #         "en": "Select Team for Details",
# # #         "ko": "상세 조회할 팀 선택 (HWK Total 포함)",
# # #         "vi": "chọn Nhóm để xem chi tiết"
# # #     },
# # #     "select_week_range": {
# # #         "en": "Select Week Range",
# # #         "ko": "주차 범위 선택",
# # #         "vi": "Chọn Phạm vi Tuần"
# # #     },
# # #     "language": {
# # #         "en": "Language",
# # #         "ko": "언어",
# # #         "vi": "ngôn ngữ"
# # #     },
# # #     "avg_by_team": {
# # #         "en": "Average {kpi} by Team",
# # #         "ko": "팀별 {kpi} 평균",
# # #         "vi": "Trung bình {kpi} theo Nhóm"
# # #     },
# # #     "weekly_trend_title": {
# # #         "en": "Weekly Trend of {kpi}",
# # #         "ko": "{kpi} 주간 추이",
# # #         "vi": "Xu hướng Hàng Tuần của {kpi}"
# # #     },
# # #     "top_teams": {
# # #         "en": "Top {n} Teams - {kpi}",
# # #         "ko": "{kpi} 상위 {n} 팀",
# # #         "vi": "Top {n} Nhóm - {kpi}"
# # #     },
# # #     "bottom_teams": {
# # #         "en": "Bottom {n} Teams - {kpi}",
# # #         "ko": "{kpi} 하위 {n} 팀",
# # #         "vi": "Nhóm {n} Thấp Nhất - {kpi}"
# # #     },
# # #     "week_col": {
# # #         "en": "Week {week}",
# # #         "ko": "{week}주차",
# # #         "vi": "Tuần {week}"
# # #     },
# # #     "average": {
# # #         "en": "Average",
# # #         "ko": "평균",
# # #         "vi": "Trung bình"
# # #     }
# # # }

# # # # --------------------------------------------------
# # # # 2. 우측 상단 언어 선택 (영어/한글/베트남어)
# # # # --------------------------------------------------
# # # col_title, col_lang = st.columns([4, 1])
# # # with col_lang:
# # #     lang = st.radio("Language / 언어 / ngôn ngữ", options=["en", "ko", "vi"], index=0, horizontal=True)

# # # st.title(trans["title"][lang])

# # # # --------------------------------------------------
# # # # 3. 유틸리티 함수 정의
# # # # --------------------------------------------------
# # # @st.cache_data
# # # def load_data():
# # #     # CSV 파일이 탭 구분자인 경우 sep="\t" 옵션을 사용합니다.
# # #     df = pd.read_csv("score.csv", sep="\t")
# # #     return df

# # # def convert_to_numeric(x):
# # #     try:
# # #         if isinstance(x, str):
# # #             if x.strip() == '-' or x.strip() == '':
# # #                 return np.nan
# # #             # 쉼표를 소수점으로 변환 (예: "0,02%" -> "0.02%")
# # #             x = x.replace(",", ".")
# # #             num = re.sub(r'[^\d\.-]', '', x)
# # #             return float(num)
# # #         else:
# # #             return x
# # #     except:
# # #         return np.nan

# # # def extract_unit(s):
# # #     if isinstance(s, str):
# # #         m = re.search(r'([^\d\.\-]+)$', s.strip())
# # #         return m.group(1).strip() if m else ""
# # #     else:
# # #         return ""

# # # def format_label(row):
# # #     unit = extract_unit(row["Actual"]) if pd.notnull(row.get("Actual", "")) else ""
# # #     return f"{row['Actual_numeric']:.2f}{unit} ({row['Final']} point)"

# # # def cumulative_performance(sub_df, kpi):
# # #     # shortage_cost는 누적합, 나머지는 평균 처리
# # #     if kpi.lower() == "shortage_cost":
# # #         return sub_df["Actual_numeric"].sum()
# # #     else:
# # #         return sub_df["Actual_numeric"].mean()

# # # def get_delta_color(kpi, delta):
# # #     if delta is None:
# # #         return "black"
# # #     kpi_lower = kpi.lower()
# # #     positive_better = ["prs validation", "6s_audit"]
# # #     negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
# # #     if kpi_lower in positive_better:
# # #         return "blue" if delta > 0 else "red" if delta < 0 else "black"
# # #     elif kpi_lower in negative_better:
# # #         return "blue" if delta < 0 else "red" if delta > 0 else "black"
# # #     else:
# # #         return "blue" if delta > 0 else "red" if delta < 0 else "black"

# # # def render_custom_metric(col, label, value, delta, delta_color):
# # #     html_metric = f"""
# # #     <div style="font-size:14px; margin:5px; padding:5px;">
# # #       <div style="font-weight:bold;">{label}</div>
# # #       <div>{value}</div>
# # #       <div style="color:{delta_color};">{delta}</div>
# # #     </div>
# # #     """
# # #     col.markdown(html_metric, unsafe_allow_html=True)

# # # def format_final_label(row):
# # #     return f"{row['Final']:.0f} point"

# # # # --------------------------------------------------
# # # # 4. 데이터 로드 및 전처리
# # # # --------------------------------------------------
# # # df = load_data()

# # # # "Week" 열의 앞뒤 공백 제거 및 대문자 통일 (예: " W4 " → "W4")
# # # df["Week"] = df["Week"].astype(str).str.strip().str.upper()

# # # # "Week" 열에서 숫자만 추출하여 Week_num 열 생성 (예: "W4" → 4)
# # # df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x) != '' else np.nan)
# # # df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
# # # df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

# # # # 디버깅: CSV 파일에서 추출된 주차 확인 (예: [1, 2, 3, 4])
# # # # st.write("CSV에 있는 주차:", sorted(df["Week_num"].dropna().unique()))

# # # # --------------------------------------------------
# # # # 5. 사이드바 위젯 (필터)
# # # # --------------------------------------------------
# # # st.sidebar.header("Filter Options")
# # # kpi_options = sorted(list(df["KPI"].unique()))
# # # if "Final score" not in kpi_options:
# # #     kpi_options.append("Final score")
# # # selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)
# # # team_list = sorted(df["Team"].unique())
# # # team_list_extended = team_list.copy()
# # # if "HWK Total" not in team_list_extended:
# # #     team_list_extended.append("HWK Total")
# # # selected_teams = st.sidebar.multiselect(trans["select_teams"][lang], options=team_list_extended, default=team_list)

# # # # 슬라이더는 CSV 전체의 주차(min, max)를 기준으로 정수(step=1) 단위로 설정
# # # selected_week_range = st.sidebar.slider(
# # #     trans["select_week_range"][lang],
# # #     int(df["Week_num"].min()),
# # #     int(df["Week_num"].max()),
# # #     (int(df["Week_num"].min()), int(df["Week_num"].max())),
# # #     step=1
# # # )
# # # selected_team_detail = st.sidebar.selectbox(trans["select_team_details"][lang], options=team_list_extended, index=0)

# # # # --------------------------------------------------
# # # # 6. 데이터 필터링 (KPI, 주차 범위 적용)
# # # # --------------------------------------------------
# # # if selected_kpi == "Final score":
# # #     df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & 
# # #                      (df["Week_num"] <= selected_week_range[1])].copy()
# # # else:
# # #     df_filtered = df[(df["KPI"] == selected_kpi) & 
# # #                      (df["Week_num"] >= selected_week_range[0]) & 
# # #                      (df["Week_num"] <= selected_week_range[1])].copy()

# # # # KPI가 Final score가 아닐 경우, 최신 주차를 기준으로 상세정보 활용
# # # if selected_kpi != "Final score":
# # #     if not df_filtered.empty:
# # #         latest_week = int(df_filtered["Week_num"].max())
# # #     else:
# # #         latest_week = None
# # # else:
# # #     if not df_filtered.empty:
# # #         latest_week = int(df_filtered["Week_num"].max())
# # #     else:
# # #         latest_week = None

# # # # --------------------------------------------------
# # # # 7. [1] KPI Performance Comparison by Team (바 차트)
# # # # --------------------------------------------------
# # # st.markdown(trans["kpi_comparison"][lang])
# # # if selected_kpi == "Final score":
# # #     df_latest = df_filtered.groupby("Team").agg({"Final": "sum"}).reset_index()
# # #     df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
# # #     if "HWK Total" in selected_teams:
# # #         overall_final = df_latest["Final"].sum()
# # #         df_total = pd.DataFrame({
# # #             "Team": ["HWK Total"],
# # #             "Final": [overall_final]
# # #         })
# # #         df_total["Label"] = df_total.apply(format_final_label, axis=1)
# # #         df_latest = pd.concat([df_latest, df_total], ignore_index=True)
# # # else:
# # #     df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()
# # #     if "HWK Total" in selected_teams:
# # #         df_overall = df_filtered[df_filtered["Week_num"] == latest_week].copy()
# # #         if not df_overall.empty:
# # #             overall_actual = df_overall["Actual_numeric"].mean()
# # #             overall_final = round(df_overall["Final"].mean())
# # #             overall_unit = extract_unit(df_overall.iloc[0]["Actual"])
# # #             df_total = pd.DataFrame({
# # #                 "Team": ["HWK Total"],
# # #                 "Actual_numeric": [overall_actual],
# # #                 "Final": [overall_final],
# # #                 "Actual": [f"{overall_actual:.2f}{overall_unit}"],
# # #                 "Week_num": [latest_week],
# # #                 "KPI": [selected_kpi]
# # #             })
# # #             df_latest = pd.concat([df_latest, df_total], ignore_index=True)
            
# # # if selected_kpi == "Final score":
# # #     df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
# # # else:
# # #     df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
# # #     df_comp["Label"] = df_comp.apply(format_label, axis=1)

# # # if selected_kpi == "Final score":
# # #     fig_bar = px.bar(
# # #         df_comp,
# # #         x="Team",
# # #         y="Final",
# # #         text="Label",
# # #         labels={"Final": "Final score by Team"}
# # #     )
# # #     fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
# # # else:
# # #     fig_bar = px.bar(
# # #         df_comp,
# # #         x="Team",
# # #         y="Actual_numeric",
# # #         text="Label",
# # #         labels={"Actual_numeric": trans["avg_by_team"][lang].format(kpi=selected_kpi)}
# # #     )
# # #     fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
# # # st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# # # # --------------------------------------------------
# # # # 8. [2] Weekly Performance Trend Analysis (라인 차트)
# # # # --------------------------------------------------
# # # st.markdown(trans["weekly_trend"][lang])
# # # if selected_kpi == "Final score":
# # #     df_trend_individual = df_filtered.sort_values("Week_num").groupby("Team").apply(
# # #         lambda x: x.assign(CumFinal=x["Final"].cumsum())
# # #     ).reset_index(drop=True)
# # #     fig_line = px.line(
# # #         df_trend_individual,
# # #         x="Week_num",
# # #         y="CumFinal",
# # #         color="Team",
# # #         markers=True,
# # #         labels={"Week_num": "Week", "CumFinal": "Cumulative Final score"},
# # #         title="Weekly Trend of Final score (Cumulative)"
# # #     )
# # #     fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)
# # #     if "HWK Total" in selected_teams:
# # #         df_overall_trend = df_filtered.sort_values("Week_num").groupby("Week_num").agg({"Final": "sum"}).reset_index()
# # #         df_overall_trend["CumFinal"] = df_overall_trend["Final"].cumsum()
# # #         fig_line.add_scatter(
# # #             x=df_overall_trend["Week_num"],
# # #             y=df_overall_trend["CumFinal"],
# # #             mode='lines+markers',
# # #             name="HWK Total",
# # #             line=dict(color='black', dash='dash')
# # #         )
# # # else:
# # #     # 만약 선택한 KPI가 b-grade라면 y축 레이블을 "%" 단위로 설정
# # #     if selected_kpi.lower() == "b-grade":
# # #         y_label = "b-grade (%)"
# # #     else:
# # #         y_label = f"{selected_kpi} Value"
# # #     df_trend_individual = df_filtered[df_filtered["Team"].isin([t for t in selected_teams if t != "HWK Total"])].copy()
# # #     fig_line = px.line(
# # #         df_trend_individual,
# # #         x="Week_num",
# # #         y="Actual_numeric",
# # #         color="Team",
# # #         markers=True,
# # #         labels={"Week_num": "Week", "Actual_numeric": y_label},
# # #         title=trans["weekly_trend_title"][lang].format(kpi=selected_kpi)
# # #     )
# # #     fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)
# # #     if "HWK Total" in selected_teams:
# # #         df_overall_trend = df_filtered.groupby("Week_num").agg({"Actual_numeric": "mean", "Final": "mean"}).reset_index()
# # #         fig_line.add_scatter(
# # #             x=df_overall_trend["Week_num"],
# # #             y=df_overall_trend["Actual_numeric"],
# # #             mode='lines+markers',
# # #             name="HWK Total",
# # #             line=dict(color='black', dash='dash')
# # #         )
# # # st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

# # # # --------------------------------------------------
# # # # 9. [3] KPI Top/Bottom Team Rankings (Top 3 / Bottom 3)
# # # # --------------------------------------------------
# # # st.markdown(trans["top_bottom_rankings"][lang])
# # # df_rank = df_comp[df_comp["Team"] != "HWK Total"].copy()
# # # df_rank = df_rank.sort_values("Actual_numeric" if selected_kpi != "Final score" else "Final", ascending=False)
# # # top_n = 3 if len(df_rank) >= 3 else len(df_rank)
# # # bottom_n = 3 if len(df_rank) >= 3 else len(df_rank)
# # # top_df = df_rank.head(top_n).copy()
# # # bottom_df = df_rank.tail(bottom_n).copy().sort_values("Actual_numeric" if selected_kpi != "Final score" else "Final", ascending=True)
# # # if selected_kpi == "Final score":
# # #     top_df["Label"] = top_df.apply(lambda row: format_final_label(row), axis=1)
# # #     bottom_df["Label"] = bottom_df.apply(lambda row: format_final_label(row), axis=1)
# # # else:
# # #     top_df["Label"] = top_df.apply(format_label, axis=1)
# # #     bottom_df["Label"] = bottom_df.apply(format_label, axis=1)
# # # col1, col2 = st.columns(2)
# # # with col1:
# # #     st.subheader(trans["top_teams"][lang].format(n=top_n, kpi=selected_kpi))
# # #     if selected_kpi == "Final score":
# # #         fig_top = px.bar(
# # #             top_df,
# # #             x="Final",
# # #             y="Team",
# # #             orientation="h",
# # #             text="Label",
# # #             labels={"Final": "Final score", "Team": "Team"}
# # #         )
# # #     else:
# # #         fig_top = px.bar(
# # #             top_df,
# # #             x="Actual_numeric",
# # #             y="Team",
# # #             orientation="h",
# # #             text="Label",
# # #             labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
# # #         )
# # #     fig_top.update_traces(texttemplate="%{text}", textposition='inside')
# # #     fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
# # #     st.plotly_chart(fig_top, use_container_width=True, key="top_chart")
# # # with col2:
# # #     st.subheader(trans["bottom_teams"][lang].format(n=bottom_n, kpi=selected_kpi))
# # #     if selected_kpi == "Final score":
# # #         fig_bottom = px.bar(
# # #             bottom_df,
# # #             x="Final",
# # #             y="Team",
# # #             orientation="h",
# # #             text="Label",
# # #             labels={"Final": "Final score", "Team": "Team"}
# # #         )
# # #     else:
# # #         fig_bottom = px.bar(
# # #             bottom_df,
# # #             x="Actual_numeric",
# # #             y="Team",
# # #             orientation="h",
# # #             text="Label",
# # #             labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
# # #         )
# # #     fig_bottom.update_traces(texttemplate="%{text}", textposition='inside', marker_color='red')
# # #     fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
# # #     st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# # # # --------------------------------------------------
# # # # 10. [4] Team-Specific KPI Detailed View (카드형 레이아웃)
# # # # --------------------------------------------------
# # # st.markdown("")
# # # if selected_team_detail != "HWK Total":
# # #     df_team = df[df["Team"] == selected_team_detail].copy()
# # # else:
# # #     df_team = df[df["Week_num"] == latest_week].copy()
# # #     df_team = df_team.groupby("KPI").agg({
# # #         "Actual_numeric": "mean",
# # #         "Final": "mean",
# # #         "Actual": "first"
# # #     }).reset_index()

# # # # (A) Last Week performance Details – 폰트 크기를 18px로 확대
# # # st.markdown(
# # #     f"<div style='font-size:18px; font-weight:bold;'>{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}</div>",
# # #     unsafe_allow_html=True
# # # )
# # # cols = st.columns(3)
# # # i = 0
# # # for kpi in df_team["KPI"].unique():
# # #     if selected_team_detail != "HWK Total":
# # #         df_last = df_team[df_team["Week_num"] == latest_week]
# # #         df_prev = df_team[df_team["Week_num"] == (latest_week - 1)]
# # #     else:
# # #         df_last = df_team.copy()
# # #         df_prev = df[df["Week_num"] == (latest_week - 1)].groupby("KPI").agg({
# # #             "Actual_numeric": "mean",
# # #             "Final": "mean",
# # #             "Actual": "first"
# # #         }).reset_index()
# # #     if not df_last[df_last["KPI"] == kpi].empty:
# # #         row_last = df_last[df_last["KPI"] == kpi].iloc[0]
# # #     else:
# # #         continue
# # #     current_label = format_label(row_last)
# # #     if not df_prev[df_prev["KPI"] == kpi].empty:
# # #         row_prev = df_prev[df_prev["KPI"] == kpi].iloc[0]
# # #         if pd.notna(row_last["Actual_numeric"]) and pd.notna(row_prev["Actual_numeric"]):
# # #             delta_actual = row_last["Actual_numeric"] - row_prev["Actual_numeric"]
# # #         else:
# # #             delta_actual = None
# # #         if pd.notna(row_last["Final"]) and pd.notna(row_prev["Final"]):
# # #             delta_final = int(round(row_last["Final"])) - int(round(row_prev["Final"]))
# # #         else:
# # #             delta_final = None
# # #         if delta_actual is not None and delta_final is not None:
# # #             kpi_lower = kpi.lower()
# # #             positive_better = ["prs validation", "6s_audit"]
# # #             if kpi_lower in positive_better:
# # #                 arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
# # #             else:
# # #                 arrow = "▲" if delta_actual < 0 else "▼" if delta_actual > 0 else ""
# # #             delta_str = f"{arrow}{delta_actual:+.2f}%({delta_final:+d} point)"
# # #         else:
# # #             delta_str = "N/A"
# # #     else:
# # #         delta_str = "N/A"
# # #         delta_actual = None
# # #     delta_color = get_delta_color(kpi, delta_actual)
# # #     render_custom_metric(cols[i % 3], kpi, current_label, delta_str, delta_color)
# # #     i += 1

# # # # (B) Total Week Performance Detail – 폰트 크기를 18px로 확대
# # # st.markdown("")
# # # st.markdown(
# # #     f"<div style='font-size:18px; font-weight:bold;'>{trans['total_week_details'][lang].format(team=selected_team_detail)}</div>",
# # #     unsafe_allow_html=True
# # # )
# # # if selected_team_detail != "HWK Total":
# # #     df_cum = df[(df["Team"] == selected_team_detail) & 
# # #                 (df["Week_num"] >= selected_week_range[0]) & 
# # #                 (df["Week_num"] <= selected_week_range[1])]
# # # else:
# # #     df_cum = df[(df["Week_num"] >= selected_week_range[0]) & 
# # #                 (df["Week_num"] <= selected_week_range[1])]
# # # df_cum_group = df_cum.groupby("KPI").apply(lambda x: cumulative_performance(x, x["KPI"].iloc[0])).reset_index(name="cum")
# # # cols_total = st.columns(3)
# # # i = 0
# # # for kpi in df_cum_group["KPI"].unique():
# # #     sub_df = df_cum[df_cum["KPI"] == kpi]
# # #     cum_value = cumulative_performance(sub_df, kpi)
# # #     team_cum = df[(df["KPI"] == kpi) & 
# # #                   (df["Week_num"] >= selected_week_range[0]) & 
# # #                   (df["Week_num"] <= selected_week_range[1])].groupby("Team").apply(lambda x: cumulative_performance(x, kpi)).reset_index(name="cum")
# # #     kpi_lower = kpi.lower()
# # #     positive_better = ["prs validation", "6s_audit"]
# # #     negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
# # #     if kpi_lower in positive_better:
# # #         best_value = team_cum["cum"].max() if not team_cum.empty else 0
# # #         delta = cum_value - best_value
# # #         arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""
# # #     elif kpi_lower in negative_better:
# # #         best_value = team_cum["cum"].min() if not team_cum.empty else 0
# # #         delta = cum_value - best_value
# # #         arrow = "▲" if delta < 0 else "▼" if delta > 0 else ""
# # #     else:
# # #         best_value = team_cum["cum"].max() if not team_cum.empty else 0
# # #         delta = cum_value - best_value
# # #         arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""
# # #     delta_str = f"{arrow}{abs(delta):+.2f} point" if best_value != 0 else ""
# # #     delta_color = get_delta_color(kpi, delta)
# # #     render_custom_metric(cols_total[i % 3], kpi, f"{cum_value:.2f}", delta_str, delta_color)
# # #     i += 1

# # # # --------------------------------------------------
# # # # 11. [5] Detailed Data Table (행과 열 전환: 행=주차, 열=KPI)
# # # # --------------------------------------------------
# # # st.markdown("")
# # # st.markdown(trans["detailed_data"][lang])
# # # kpi_all = sorted(df["KPI"].unique())
# # # # CSV 파일에 있는 모든 주차 데이터를 동적으로 표시하도록 설정
# # # max_week = int(df["Week_num"].max())
# # # weeks_to_show = list(range(1, max_week + 1))
# # # data_table = {}
# # # for kpi in kpi_all:
# # #     row_data = {}
# # #     values = []
# # #     finals = []
# # #     unit = ""
# # #     for w in weeks_to_show:
# # #         if selected_team_detail != "HWK Total":
# # #             sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
# # #             if not sub_df.empty:
# # #                 val = sub_df.iloc[0]["Actual_numeric"]
# # #                 final_val = sub_df.iloc[0]["Final"]
# # #                 unit = extract_unit(sub_df.iloc[0]["Actual"])
# # #                 formatted = f"{val:.2f}{unit}<br>({final_val} point)"
# # #                 row_data[f"Week {w}"] = formatted
# # #                 values.append(val)
# # #                 finals.append(final_val)
# # #             else:
# # #                 row_data[f"Week {w}"] = "N/A"
# # #         else:
# # #             # HWK Total: 전체 팀에 대한 평균 계산
# # #             sub_df = df[(df["KPI"] == kpi) & (df["Week_num"] == w)]
# # #             if not sub_df.empty:
# # #                 val = sub_df["Actual_numeric"].mean()
# # #                 final_val = sub_df["Final"].mean()
# # #                 unit = extract_unit(sub_df.iloc[0]["Actual"])
# # #                 formatted = f"{val:.2f}{unit}<br>({final_val:.2f} point)"
# # #                 row_data[f"Week {w}"] = formatted
# # #                 values.append(val)
# # #                 finals.append(final_val)
# # #             else:
# # #                 row_data[f"Week {w}"] = "N/A"
# # #     if values:
# # #         avg_val = sum(values) / len(values)
# # #         avg_final = sum(finals) / len(finals) if finals else 0
# # #         row_data["Average"] = f"{avg_val:.2f}{unit}<br>({avg_final:.2f} point)"
# # #     else:
# # #         row_data["Average"] = "N/A"
# # #     data_table[kpi] = row_data

# # # table_df = pd.DataFrame(data_table)
# # # index_order = [f"Week {w}" for w in weeks_to_show] + ["Average"]
# # # table_df = table_df.reindex(index_order)
# # # new_index = {}
# # # for idx in table_df.index:
# # #     if idx.startswith("Week"):
# # #         week_num = idx.split()[1]
# # #         new_index[idx] = trans["week_col"][lang].format(week=week_num)
# # #     elif idx == "Average":
# # #         new_index[idx] = trans["average"][lang]
# # #     else:
# # #         new_index[idx] = idx
# # # table_df.rename(index=new_index, inplace=True)
# # # st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)


# # import streamlit as st
# # import pandas as pd
# # import plotly.express as px
# # import numpy as np
# # import re
# # import os

# # # --------------------------------------------------
# # # 1. 페이지 설정 및 다국어 번역용 사전 정의
# # # --------------------------------------------------
# # st.set_page_config(page_title="HWK Quality competition Event", layout="wide")

# # # 번역 사전 (영어/한글/베트남어)
# # trans = {
# #     "title": {
# #         "en": "HWK Quality competition Event",
# #         "ko": "HWK 품질 경쟁 이벤트",
# #         "vi": "HWK sự kiện thi đua chất lượng"
# #     },
# #     "kpi_comparison": {
# #         "en": "1. KPI Performance Comparison by Team",
# #         "ko": "1. 팀별 KPI 성과 비교",
# #         "vi": "So sánh Hiệu suất KPI theo Nhóm"
# #     },
# #     "weekly_trend": {
# #         "en": "2. Weekly Performance Trend Analysis",
# #         "ko": "2. 주간 성과 트렌드 분석",
# #         "vi": "2. Phân tích Xu hướng Hiệu suất Hàng Tuần"
# #     },
# #     "top_bottom_rankings": {
# #         "en": "3. KPI Top/Bottom Team Rankings",
# #         "ko": "3. KPI 상위/하위 팀 순위",
# #         "vi": "3. Xếp hạng Nhóm KPI Cao/Thấp Nhất"
# #     },
# #     "last_week_details": {
# #         "en": "Last Week performance Details for {team} (Week {week})",
# #         "ko": "지난주 성과 상세보기: {team} (Week {week})",
# #         "vi": "Chi tiết Hiệu suất Tuần Trước của {team} (Tuần {week})"
# #     },
# #     "total_week_details": {
# #         "en": "Total Week Performance Detail for {team} (All weeks)",
# #         "ko": "전체 주차 누적 실적 상세: {team} (All weeks)",
# #         "vi": "Chi tiết Hiệu suất Tổng Tuần của {team} (Tất cả các tuần)"
# #     },
# #     "detailed_data": {
# #         "en": "Detailed Data for Selected Team",
# #         "ko": "선택된 팀의 상세 데이터",
# #         "vi": "Dữ liệu Chi tiết cho Nhóm Đã Chọn"
# #     },
# #     "select_kpi": {
# #         "en": "Select KPI",
# #         "ko": "KPI 선택",
# #         "vi": "Chọn KPI"
# #     },
# #     "select_teams": {
# #         "en": "Select Teams for Comparison",
# #         "ko": "비교할 팀 선택 (HWK Total 포함)",
# #         "vi": "Chọn Nhóm để So Sánh"
# #     },
# #     "select_team_details": {
# #         "en": "Select Team for Details",
# #         "ko": "상세 조회할 팀 선택 (HWK Total 포함)",
# #         "vi": "chọn Nhóm để xem chi tiết"
# #     },
# #     "select_week_range": {
# #         "en": "Select Week Range",
# #         "ko": "주차 범위 선택",
# #         "vi": "Chọn Phạm vi Tuần"
# #     },
# #     "language": {
# #         "en": "Language",
# #         "ko": "언어",
# #         "vi": "ngôn ngữ"
# #     },
# #     "avg_by_team": {
# #         "en": "Average {kpi} by Team",
# #         "ko": "팀별 {kpi} 평균",
# #         "vi": "Trung bình {kpi} theo Nhóm"
# #     },
# #     "weekly_trend_title": {
# #         "en": "Weekly Trend of {kpi}",
# #         "ko": "{kpi} 주간 추이",
# #         "vi": "Xu hướng Hàng Tuần của {kpi}"
# #     },
# #     "top_teams": {
# #         "en": "Top {n} Teams - {kpi}",
# #         "ko": "{kpi} 상위 {n} 팀",
# #         "vi": "Top {n} Nhóm - {kpi}"
# #     },
# #     "bottom_teams": {
# #         "en": "Bottom {n} Teams - {kpi}",
# #         "ko": "{kpi} 하위 {n} 팀",
# #         "vi": "Nhóm {n} Thấp Nhất - {kpi}"
# #     },
# #     "week_col": {
# #         "en": "Week {week}",
# #         "ko": "{week}주차",
# #         "vi": "Tuần {week}"
# #     },
# #     "average": {
# #         "en": "Average",
# #         "ko": "평균",
# #         "vi": "Trung bình"
# #     }
# # }

# # # --------------------------------------------------
# # # 2. 우측 상단 언어 선택 (영어/한글/베트남어)
# # # --------------------------------------------------
# # col_title, col_lang = st.columns([4, 1])
# # with col_lang:
# #     lang = st.radio("Language / 언어 / ngôn ngữ", options=["en", "ko", "vi"], index=0, horizontal=True)

# # st.title(trans["title"][lang])

# # # --------------------------------------------------
# # # 3. 유틸리티 함수 정의
# # # --------------------------------------------------
# # @st.cache_data
# # def load_data():
# #     # CSV 파일이 탭 구분자인 경우 sep="\t" 옵션을 사용합니다.
# #     df = pd.read_csv("score.csv", sep="\t")
# #     return df

# # def convert_to_numeric(x):
# #     try:
# #         if isinstance(x, str):
# #             if x.strip() == '-' or x.strip() == '':
# #                 return np.nan
# #             # 쉼표를 소수점으로 변환 (예: "0,02%" -> "0.02%")
# #             x = x.replace(",", ".")
# #             num = re.sub(r'[^\d\.-]', '', x)
# #             return float(num)
# #         else:
# #             return x
# #     except:
# #         return np.nan

# # def extract_unit(s):
# #     if isinstance(s, str):
# #         m = re.search(r'([^\d\.\-]+)$', s.strip())
# #         return m.group(1).strip() if m else ""
# #     else:
# #         return ""

# # def format_label(row):
# #     unit = extract_unit(row["Actual"]) if pd.notnull(row.get("Actual", "")) else ""
# #     return f"{row['Actual_numeric']:.2f}{unit} ({row['Final']} point)"

# # def cumulative_performance(sub_df, kpi):
# #     # shortage_cost는 누적합, 나머지는 평균 처리
# #     if kpi.lower() == "shortage_cost":
# #         return sub_df["Actual_numeric"].sum()
# #     else:
# #         return sub_df["Actual_numeric"].mean()

# # def get_delta_color(kpi, delta):
# #     if delta is None:
# #         return "black"
# #     kpi_lower = kpi.lower()
# #     positive_better = ["prs validation", "6s_audit"]
# #     negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
# #     if kpi_lower in positive_better:
# #         return "blue" if delta > 0 else "red" if delta < 0 else "black"
# #     elif kpi_lower in negative_better:
# #         return "blue" if delta < 0 else "red" if delta > 0 else "black"
# #     else:
# #         return "blue" if delta > 0 else "red" if delta < 0 else "black"

# # def render_custom_metric(col, label, value, delta, delta_color):
# #     html_metric = f"""
# #     <div style="font-size:14px; margin:5px; padding:5px;">
# #       <div style="font-weight:bold;">{label}</div>
# #       <div>{value}</div>
# #       <div style="color:{delta_color};">{delta}</div>
# #     </div>
# #     """
# #     col.markdown(html_metric, unsafe_allow_html=True)

# # def format_final_label(row):
# #     return f"{row['Final']:.0f} point"

# # # --------------------------------------------------
# # # 4. 데이터 로드 및 전처리
# # # --------------------------------------------------
# # df = load_data()

# # # (중요) Week 컬럼 전처리를 조금 더 강화
# # df["Week"] = df["Week"].astype(str)
# # # 앞뒤 공백 제거 + 대문자 + 중간 공백 제거
# # df["Week"] = df["Week"].str.strip().str.upper().str.replace(" ", "", regex=False)

# # # 이제 숫자만 추출하여 Week_num에 저장
# # df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x) else np.nan)

# # # "Actual_numeric"와 "Final" 처리
# # df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
# # df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

# # # 디버깅용: 실제로 어떤 주차들이 있는지 확인하고 싶다면 아래를 잠시 풀어서 확인해보세요.
# # # st.write("CSV에 있는 주차:", sorted(df["Week_num"].dropna().unique()))

# # # --------------------------------------------------
# # # 5. 사이드바 위젯 (필터)
# # # --------------------------------------------------
# # st.sidebar.header("Filter Options")

# # # KPI 목록
# # kpi_options = sorted(list(df["KPI"].unique()))
# # if "Final score" not in kpi_options:
# #     kpi_options.append("Final score")
# # selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)

# # # 팀 목록
# # team_list = sorted(df["Team"].unique())
# # team_list_extended = team_list.copy()
# # if "HWK Total" not in team_list_extended:
# #     team_list_extended.append("HWK Total")

# # selected_teams = st.sidebar.multiselect(trans["select_teams"][lang], options=team_list_extended, default=team_list)

# # # **주차 슬라이더**: CSV 전체 주차의 min, max를 활용
# # min_week = int(df["Week_num"].min())
# # max_week = int(df["Week_num"].max())

# # selected_week_range = st.sidebar.slider(
# #     trans["select_week_range"][lang],
# #     min_week,
# #     max_week,
# #     (min_week, max_week),
# #     step=1
# # )

# # selected_team_detail = st.sidebar.selectbox(trans["select_team_details"][lang], options=team_list_extended, index=0)

# # # --------------------------------------------------
# # # 6. 데이터 필터링 (KPI, 주차 범위 적용)
# # # --------------------------------------------------
# # if selected_kpi == "Final score":
# #     df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & 
# #                      (df["Week_num"] <= selected_week_range[1])].copy()
# # else:
# #     df_filtered = df[(df["KPI"] == selected_kpi) & 
# #                      (df["Week_num"] >= selected_week_range[0]) & 
# #                      (df["Week_num"] <= selected_week_range[1])].copy()

# # # KPI가 Final score가 아닐 경우, 최신 주차 파악
# # if not df_filtered.empty:
# #     latest_week = int(df_filtered["Week_num"].max())
# # else:
# #     latest_week = None

# # # --------------------------------------------------
# # # 7. [1] KPI Performance Comparison by Team (바 차트)
# # # --------------------------------------------------
# # st.markdown(trans["kpi_comparison"][lang])
# # if selected_kpi == "Final score":
# #     # 팀별 Final 값 합산
# #     df_latest = df_filtered.groupby("Team").agg({"Final": "sum"}).reset_index()
# #     df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
# #     # HWK Total이 포함되어 있다면 전체 합계 추가
# #     if "HWK Total" in selected_teams:
# #         overall_final = df_latest["Final"].sum()
# #         df_total = pd.DataFrame({
# #             "Team": ["HWK Total"],
# #             "Final": [overall_final]
# #         })
# #         df_total["Label"] = df_total.apply(format_final_label, axis=1)
# #         df_latest = pd.concat([df_latest, df_total], ignore_index=True)
# # else:
# #     # 최신 주차 데이터만
# #     df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()
# #     # HWK Total 포함 시 전체 평균(또는 합산) 추가
# #     if "HWK Total" in selected_teams and not df_latest.empty:
# #         overall_actual = df_latest["Actual_numeric"].mean()
# #         overall_final = round(df_latest["Final"].mean())
# #         overall_unit = extract_unit(df_latest.iloc[0]["Actual"])
# #         df_total = pd.DataFrame({
# #             "Team": ["HWK Total"],
# #             "Actual_numeric": [overall_actual],
# #             "Final": [overall_final],
# #             "Actual": [f"{overall_actual:.2f}{overall_unit}"],
# #             "Week_num": [latest_week],
# #             "KPI": [selected_kpi]
# #         })
# #         df_latest = pd.concat([df_latest, df_total], ignore_index=True)

# # # 선택한 팀만 필터링
# # df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
# # if selected_kpi != "Final score":
# #     df_comp["Label"] = df_comp.apply(format_label, axis=1)

# # # 바 차트
# # if selected_kpi == "Final score":
# #     fig_bar = px.bar(
# #         df_comp,
# #         x="Team",
# #         y="Final",
# #         text="Label",
# #         labels={"Final": "Final score by Team"}
# #     )
# #     fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
# # else:
# #     fig_bar = px.bar(
# #         df_comp,
# #         x="Team",
# #         y="Actual_numeric",
# #         text="Label",
# #         labels={"Actual_numeric": trans["avg_by_team"][lang].format(kpi=selected_kpi)}
# #     )
# #     fig_bar.update_traces(texttemplate="%{text}", textposition='inside')

# # st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# # # --------------------------------------------------
# # # 8. [2] Weekly Performance Trend Analysis (라인 차트)
# # # --------------------------------------------------
# # st.markdown(trans["weekly_trend"][lang])

# # if selected_kpi == "Final score":
# #     # 주차별로 Final을 누적합
# #     df_trend_individual = df_filtered.sort_values("Week_num").groupby("Team").apply(
# #         lambda x: x.assign(CumFinal=x["Final"].cumsum())
# #     ).reset_index(drop=True)

# #     fig_line = px.line(
# #         df_trend_individual,
# #         x="Week_num",
# #         y="CumFinal",
# #         color="Team",
# #         markers=True,
# #         labels={"Week_num": "Week", "CumFinal": "Cumulative Final score"},
# #         title="Weekly Trend of Final score (Cumulative)"
# #     )
# #     fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)

# #     # HWK Total도 누적합으로 표시
# #     if "HWK Total" in selected_teams:
# #         df_overall_trend = df_filtered.sort_values("Week_num").groupby("Week_num").agg({"Final": "sum"}).reset_index()
# #         df_overall_trend["CumFinal"] = df_overall_trend["Final"].cumsum()
# #         fig_line.add_scatter(
# #             x=df_overall_trend["Week_num"],
# #             y=df_overall_trend["CumFinal"],
# #             mode='lines+markers',
# #             name="HWK Total",
# #             line=dict(color='black', dash='dash')
# #         )
# # else:
# #     # KPI가 b-grade 같은 % 단위인 경우 y_label 표시
# #     if selected_kpi.lower() == "b-grade":
# #         y_label = "b-grade (%)"
# #     else:
# #         y_label = f"{selected_kpi} Value"

# #     df_trend_individual = df_filtered[df_filtered["Team"].isin([t for t in selected_teams if t != "HWK Total"])].copy()
# #     fig_line = px.line(
# #         df_trend_individual,
# #         x="Week_num",
# #         y="Actual_numeric",
# #         color="Team",
# #         markers=True,
# #         labels={"Week_num": "Week", "Actual_numeric": y_label},
# #         title=trans["weekly_trend_title"][lang].format(kpi=selected_kpi)
# #     )
# #     fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)

# #     # HWK Total = 평균
# #     if "HWK Total" in selected_teams:
# #         df_overall_trend = df_filtered.groupby("Week_num").agg({"Actual_numeric": "mean", "Final": "mean"}).reset_index()
# #         fig_line.add_scatter(
# #             x=df_overall_trend["Week_num"],
# #             y=df_overall_trend["Actual_numeric"],
# #             mode='lines+markers',
# #             name="HWK Total",
# #             line=dict(color='black', dash='dash')
# #         )

# # st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

# # # --------------------------------------------------
# # # 9. [3] KPI Top/Bottom Team Rankings (Top 3 / Bottom 3)
# # # --------------------------------------------------
# # st.markdown(trans["top_bottom_rankings"][lang])

# # df_rank = df_comp[df_comp["Team"] != "HWK Total"].copy()
# # if selected_kpi == "Final score":
# #     df_rank = df_rank.sort_values("Final", ascending=False)
# # else:
# #     df_rank = df_rank.sort_values("Actual_numeric", ascending=False)

# # top_n = 3 if len(df_rank) >= 3 else len(df_rank)
# # bottom_n = 3 if len(df_rank) >= 3 else len(df_rank)

# # top_df = df_rank.head(top_n).copy()
# # bottom_df = df_rank.tail(bottom_n).copy()

# # # Bottom은 오름차순 정렬해서 그려주면 보기 편함
# # if selected_kpi == "Final score":
# #     bottom_df = bottom_df.sort_values("Final", ascending=True)
# # else:
# #     bottom_df = bottom_df.sort_values("Actual_numeric", ascending=True)

# # # 라벨 생성
# # if selected_kpi == "Final score":
# #     top_df["Label"] = top_df.apply(lambda row: format_final_label(row), axis=1)
# #     bottom_df["Label"] = bottom_df.apply(lambda row: format_final_label(row), axis=1)
# # else:
# #     top_df["Label"] = top_df.apply(format_label, axis=1)
# #     bottom_df["Label"] = bottom_df.apply(format_label, axis=1)

# # col1, col2 = st.columns(2)
# # with col1:
# #     st.subheader(trans["top_teams"][lang].format(n=top_n, kpi=selected_kpi))
# #     if selected_kpi == "Final score":
# #         fig_top = px.bar(
# #             top_df,
# #             x="Final",
# #             y="Team",
# #             orientation="h",
# #             text="Label",
# #             labels={"Final": "Final score", "Team": "Team"}
# #         )
# #     else:
# #         fig_top = px.bar(
# #             top_df,
# #             x="Actual_numeric",
# #             y="Team",
# #             orientation="h",
# #             text="Label",
# #             labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
# #         )
# #     fig_top.update_traces(texttemplate="%{text}", textposition='inside')
# #     fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
# #     st.plotly_chart(fig_top, use_container_width=True, key="top_chart")

# # with col2:
# #     st.subheader(trans["bottom_teams"][lang].format(n=bottom_n, kpi=selected_kpi))
# #     if selected_kpi == "Final score":
# #         fig_bottom = px.bar(
# #             bottom_df,
# #             x="Final",
# #             y="Team",
# #             orientation="h",
# #             text="Label",
# #             labels={"Final": "Final score", "Team": "Team"}
# #         )
# #     else:
# #         fig_bottom = px.bar(
# #             bottom_df,
# #             x="Actual_numeric",
# #             y="Team",
# #             orientation="h",
# #             text="Label",
# #             labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
# #         )
# #     fig_bottom.update_traces(texttemplate="%{text}", textposition='inside', marker_color='red')
# #     fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
# #     st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# # # --------------------------------------------------
# # # 10. [4] Team-Specific KPI Detailed View (카드형 레이아웃)
# # # --------------------------------------------------
# # st.markdown("")
# # if selected_team_detail != "HWK Total":
# #     df_team = df[df["Team"] == selected_team_detail].copy()
# # else:
# #     # HWK Total을 선택하면 latest_week 데이터만 모아서 평균
# #     df_team = df[df["Week_num"] == latest_week].groupby("KPI").agg({
# #         "Actual_numeric": "mean",
# #         "Final": "mean",
# #         "Actual": "first"
# #     }).reset_index()

# # # (A) Last Week performance Details
# # if latest_week is not None:
# #     st.markdown(
# #         f"<div style='font-size:18px; font-weight:bold;'>{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}</div>",
# #         unsafe_allow_html=True
# #     )

# #     cols = st.columns(3)
# #     i = 0
# #     for kpi in df_team["KPI"].unique():
# #         if selected_team_detail != "HWK Total":
# #             df_last = df_team[(df_team["Week_num"] == latest_week) & (df_team["KPI"] == kpi)]
# #             df_prev = df_team[(df_team["Week_num"] == (latest_week - 1)) & (df_team["KPI"] == kpi)]
# #         else:
# #             # HWK Total일 경우 이미 groupby로 묶인 상태라 latest_week가 따로 없음
# #             df_last = df_team[df_team["KPI"] == kpi]
# #             # 이전주 데이터는 별도 groupby
# #             df_prev_raw = df[df["Week_num"] == (latest_week - 1)].groupby("KPI").agg({
# #                 "Actual_numeric": "mean",
# #                 "Final": "mean",
# #                 "Actual": "first"
# #             }).reset_index()
# #             df_prev = df_prev_raw[df_prev_raw["KPI"] == kpi]

# #         if not df_last.empty:
# #             row_last = df_last.iloc[0]
# #         else:
# #             continue

# #         current_label = format_label(row_last) if selected_team_detail != "HWK Total" else f"{row_last['Actual_numeric']:.2f}{extract_unit(row_last['Actual'])} ({int(round(row_last['Final']))} point)"

# #         if not df_prev.empty:
# #             row_prev = df_prev.iloc[0]
# #             if pd.notna(row_last["Actual_numeric"]) and pd.notna(row_prev["Actual_numeric"]):
# #                 delta_actual = row_last["Actual_numeric"] - row_prev["Actual_numeric"]
# #             else:
# #                 delta_actual = None

# #             if pd.notna(row_last["Final"]) and pd.notna(row_prev["Final"]):
# #                 delta_final = int(round(row_last["Final"])) - int(round(row_prev["Final"]))
# #             else:
# #                 delta_final = None

# #             if delta_actual is not None and delta_final is not None:
# #                 kpi_lower = kpi.lower()
# #                 positive_better = ["prs validation", "6s_audit"]
# #                 if kpi_lower in positive_better:
# #                     arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
# #                 else:
# #                     arrow = "▲" if delta_actual < 0 else "▼" if delta_actual > 0 else ""
# #                 # 예: ▲+1.20%(+2 point)
# #                 delta_str = f"{arrow}{delta_actual:+.2f}%({delta_final:+d} point)"
# #             else:
# #                 delta_str = "N/A"
# #         else:
# #             delta_str = "N/A"
# #             delta_actual = None

# #         delta_color = get_delta_color(kpi, delta_actual)
# #         render_custom_metric(cols[i % 3], kpi, current_label, delta_str, delta_color)
# #         i += 1

# # # (B) Total Week Performance Detail
# # st.markdown("")
# # st.markdown(
# #     f"<div style='font-size:18px; font-weight:bold;'>{trans['total_week_details'][lang].format(team=selected_team_detail)}</div>",
# #     unsafe_allow_html=True
# # )

# # if selected_team_detail != "HWK Total":
# #     df_cum = df[(df["Team"] == selected_team_detail) & 
# #                 (df["Week_num"] >= selected_week_range[0]) & 
# #                 (df["Week_num"] <= selected_week_range[1])]
# # else:
# #     df_cum = df[(df["Week_num"] >= selected_week_range[0]) & 
# #                 (df["Week_num"] <= selected_week_range[1])]

# # df_cum_group = df_cum.groupby("KPI").apply(lambda x: cumulative_performance(x, x["KPI"].iloc[0])).reset_index(name="cum")

# # cols_total = st.columns(3)
# # i = 0
# # for kpi in df_cum_group["KPI"].unique():
# #     sub_df = df_cum[df_cum["KPI"] == kpi]
# #     cum_value = cumulative_performance(sub_df, kpi)

# #     team_cum = df[(df["KPI"] == kpi) & 
# #                   (df["Week_num"] >= selected_week_range[0]) & 
# #                   (df["Week_num"] <= selected_week_range[1])].groupby("Team").apply(lambda x: cumulative_performance(x, kpi)).reset_index(name="cum")

# #     kpi_lower = kpi.lower()
# #     positive_better = ["prs validation", "6s_audit"]
# #     negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]

# #     if kpi_lower in positive_better:
# #         best_value = team_cum["cum"].max() if not team_cum.empty else 0
# #         delta = cum_value - best_value
# #         arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""
# #     elif kpi_lower in negative_better:
# #         best_value = team_cum["cum"].min() if not team_cum.empty else 0
# #         delta = cum_value - best_value
# #         arrow = "▲" if delta < 0 else "▼" if delta > 0 else ""
# #     else:
# #         best_value = team_cum["cum"].max() if not team_cum.empty else 0
# #         delta = cum_value - best_value
# #         arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""

# #     delta_str = f"{arrow}{abs(delta):+.2f} point" if best_value != 0 else ""
# #     delta_color = get_delta_color(kpi, delta)

# #     render_custom_metric(cols_total[i % 3], kpi, f"{cum_value:.2f}", delta_str, delta_color)
# #     i += 1

# # # --------------------------------------------------
# # # 11. [5] Detailed Data Table (행과 열 전환: 행=주차, 열=KPI)
# # # --------------------------------------------------
# # st.markdown("")
# # st.markdown(trans["detailed_data"][lang])

# # kpi_all = sorted(df["KPI"].unique())

# # # CSV 파일에 있는 모든 주차 데이터를 동적으로 표시
# # all_weeks = sorted(df["Week_num"].dropna().unique())
# # data_table = {}

# # for kpi in kpi_all:
# #     row_data = {}
# #     values = []
# #     finals = []
# #     unit = ""
# #     for w in all_weeks:
# #         if selected_team_detail != "HWK Total":
# #             sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
# #             if not sub_df.empty:
# #                 val = sub_df.iloc[0]["Actual_numeric"]
# #                 final_val = sub_df.iloc[0]["Final"]
# #                 unit = extract_unit(sub_df.iloc[0]["Actual"])
# #                 formatted = f"{val:.2f}{unit}<br>({final_val} point)"
# #                 row_data[f"Week {int(w)}"] = formatted
# #                 values.append(val)
# #                 finals.append(final_val)
# #             else:
# #                 row_data[f"Week {int(w)}"] = "N/A"
# #         else:
# #             # HWK Total이면 해당 주차 전체 팀 평균
# #             sub_df = df[(df["KPI"] == kpi) & (df["Week_num"] == w)]
# #             if not sub_df.empty:
# #                 val = sub_df["Actual_numeric"].mean()
# #                 final_val = sub_df["Final"].mean()
# #                 unit = extract_unit(sub_df.iloc[0]["Actual"])
# #                 formatted = f"{val:.2f}{unit}<br>({final_val:.2f} point)"
# #                 row_data[f"Week {int(w)}"] = formatted
# #                 values.append(val)
# #                 finals.append(final_val)
# #             else:
# #                 row_data[f"Week {int(w)}"] = "N/A"

# #     if values:
# #         avg_val = sum(values) / len(values)
# #         avg_final = sum(finals) / len(finals) if finals else 0
# #         row_data["Average"] = f"{avg_val:.2f}{unit}<br>({avg_final:.2f} point)"
# #     else:
# #         row_data["Average"] = "N/A"

# #     data_table[kpi] = row_data

# # table_df = pd.DataFrame(data_table)

# # # Week X와 Average 순서대로 행 재배치
# # index_order = [f"Week {int(w)}" for w in all_weeks] + ["Average"]
# # table_df = table_df.reindex(index_order)

# # # 인덱스 이름을 다국어로 교체
# # new_index = {}
# # for idx in table_df.index:
# #     if idx.startswith("Week"):
# #         week_num = idx.split()[1]
# #         new_index[idx] = trans["week_col"][lang].format(week=week_num)
# #     elif idx == "Average":
# #         new_index[idx] = trans["average"][lang]
# #     else:
# #         new_index[idx] = idx

# # table_df.rename(index=new_index, inplace=True)

# # # HTML 형태로 표시(escape=False로 <br> 등 허용)
# # st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import numpy as np
# import re
# import unicodedata
# import os

# # --------------------------------------------------
# # 1. 페이지 설정 및 다국어 번역용 사전 정의
# # --------------------------------------------------
# st.set_page_config(page_title="HWK Quality competition Event", layout="wide")

# # 번역 사전 (영어/한글/베트남어)
# trans = {
#     "title": {
#         "en": "HWK Quality competition Event",
#         "ko": "HWK 품질 경쟁 이벤트",
#         "vi": "HWK sự kiện thi đua chất lượng"
#     },
#     "kpi_comparison": {
#         "en": "1. KPI Performance Comparison by Team",
#         "ko": "1. 팀별 KPI 성과 비교",
#         "vi": "So sánh Hiệu suất KPI theo Nhóm"
#     },
#     "weekly_trend": {
#         "en": "2. Weekly Performance Trend Analysis",
#         "ko": "2. 주간 성과 트렌드 분석",
#         "vi": "2. Phân tích Xu hướng Hiệu suất Hàng Tuần"
#     },
#     "top_bottom_rankings": {
#         "en": "3. KPI Top/Bottom Team Rankings",
#         "ko": "3. KPI 상위/하위 팀 순위",
#         "vi": "3. Xếp hạng Nhóm KPI Cao/Thấp Nhất"
#     },
#     "last_week_details": {
#         "en": "Last Week performance Details for {team} (Week {week})",
#         "ko": "지난주 성과 상세보기: {team} (Week {week})",
#         "vi": "Chi tiết Hiệu suất Tuần Trước của {team} (Tuần {week})"
#     },
#     "total_week_details": {
#         "en": "Total Week Performance Detail for {team} (All weeks)",
#         "ko": "전체 주차 누적 실적 상세: {team} (All weeks)",
#         "vi": "Chi tiết Hiệu suất Tổng Tuần của {team} (Tất cả các tuần)"
#     },
#     "detailed_data": {
#         "en": "Detailed Data for Selected Team",
#         "ko": "선택된 팀의 상세 데이터",
#         "vi": "Dữ liệu Chi tiết cho Nhóm Đã Chọn"
#     },
#     "select_kpi": {
#         "en": "Select KPI",
#         "ko": "KPI 선택",
#         "vi": "Chọn KPI"
#     },
#     "select_teams": {
#         "en": "Select Teams for Comparison",
#         "ko": "비교할 팀 선택 (HWK Total 포함)",
#         "vi": "Chọn Nhóm để So Sánh"
#     },
#     "select_team_details": {
#         "en": "Select Team for Details",
#         "ko": "상세 조회할 팀 선택 (HWK Total 포함)",
#         "vi": "chọn Nhóm để xem chi tiết"
#     },
#     "select_week_range": {
#         "en": "Select Week Range",
#         "ko": "주차 범위 선택",
#         "vi": "Chọn Phạm vi Tuần"
#     },
#     "language": {
#         "en": "Language",
#         "ko": "언어",
#         "vi": "ngôn ngữ"
#     },
#     "avg_by_team": {
#         "en": "Average {kpi} by Team",
#         "ko": "팀별 {kpi} 평균",
#         "vi": "Trung bình {kpi} theo Nhóm"
#     },
#     "weekly_trend_title": {
#         "en": "Weekly Trend of {kpi}",
#         "ko": "{kpi} 주간 추이",
#         "vi": "Xu hướng Hàng Tuần của {kpi}"
#     },
#     "top_teams": {
#         "en": "Top {n} Teams - {kpi}",
#         "ko": "{kpi} 상위 {n} 팀",
#         "vi": "Top {n} Nhóm - {kpi}"
#     },
#     "bottom_teams": {
#         "en": "Bottom {n} Teams - {kpi}",
#         "ko": "{kpi} 하위 {n} 팀",
#         "vi": "Nhóm {n} Thấp Nhất - {kpi}"
#     },
#     "week_col": {
#         "en": "Week {week}",
#         "ko": "{week}주차",
#         "vi": "Tuần {week}"
#     },
#     "average": {
#         "en": "Average",
#         "ko": "평균",
#         "vi": "Trung bình"
#     }
# }

# # --------------------------------------------------
# # 2. 우측 상단 언어 선택 (영어/한글/베트남어)
# # --------------------------------------------------
# col_title, col_lang = st.columns([4, 1])
# with col_lang:
#     lang = st.radio("Language / 언어 / ngôn ngữ", options=["en", "ko", "vi"], index=0, horizontal=True)

# st.title(trans["title"][lang])

# # --------------------------------------------------
# # 3. 유틸리티 함수 정의
# # --------------------------------------------------

# def remove_all_spaces(s: str) -> str:
#     """모든 유니코드 공백(스페이스, 탭, NBSP 등)을 제거"""
#     return re.sub(r'\s+', '', s)

# def to_halfwidth(s: str) -> str:
#     """전각(Fullwidth) 문자를 반각(ASCII) 문자로 변환"""
#     return unicodedata.normalize('NFKC', s)

# @st.cache_data
# def load_data():
#     # CSV 파일 인코딩을 상황에 맞춰 지정 (예: 'utf-8', 'cp949' 등)
#     df = pd.read_csv("score.csv", sep="\t", encoding="utf-8")
#     return df

# def convert_to_numeric(x):
#     try:
#         if isinstance(x, str):
#             if x.strip() == '-' or x.strip() == '':
#                 return np.nan
#             # 쉼표를 소수점으로 변환 (예: "0,02%" -> "0.02%")
#             x = x.replace(",", ".")
#             num = re.sub(r'[^\d\.-]', '', x)
#             return float(num)
#         else:
#             return x
#     except:
#         return np.nan

# def extract_unit(s):
#     if isinstance(s, str):
#         m = re.search(r'([^\d\.\-]+)$', s.strip())
#         return m.group(1).strip() if m else ""
#     else:
#         return ""

# def format_label(row):
#     unit = extract_unit(row["Actual"]) if pd.notnull(row.get("Actual", "")) else ""
#     return f"{row['Actual_numeric']:.2f}{unit} ({row['Final']} point)"

# def cumulative_performance(sub_df, kpi):
#     # shortage_cost는 누적합, 나머지는 평균 처리
#     if kpi.lower() == "shortage_cost":
#         return sub_df["Actual_numeric"].sum()
#     else:
#         return sub_df["Actual_numeric"].mean()

# def get_delta_color(kpi, delta):
#     if delta is None:
#         return "black"
#     kpi_lower = kpi.lower()
#     positive_better = ["prs validation", "6s_audit"]
#     negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
#     if kpi_lower in positive_better:
#         return "blue" if delta > 0 else "red" if delta < 0 else "black"
#     elif kpi_lower in negative_better:
#         return "blue" if delta < 0 else "red" if delta > 0 else "black"
#     else:
#         return "blue" if delta > 0 else "red" if delta < 0 else "black"

# def render_custom_metric(col, label, value, delta, delta_color):
#     html_metric = f"""
#     <div style="font-size:14px; margin:5px; padding:5px;">
#       <div style="font-weight:bold;">{label}</div>
#       <div>{value}</div>
#       <div style="color:{delta_color};">{delta}</div>
#     </div>
#     """
#     col.markdown(html_metric, unsafe_allow_html=True)

# def format_final_label(row):
#     return f"{row['Final']:.0f} point"

# # --------------------------------------------------
# # 4. 데이터 로드 및 전처리
# # --------------------------------------------------
# df = load_data()

# # Week 컬럼 전처리 (전각→반각 + 유니코드 공백 제거 + 대문자화)
# df["Week"] = (
#     df["Week"]
#     .astype(str)
#     .apply(to_halfwidth)        # 전각 -> 반각 변환
#     .str.upper()                # 대문자
#     .apply(remove_all_spaces)   # 모든 유니코드 공백 제거
# )

# # 숫자만 추출하여 Week_num에 저장 (W5 -> 5, W10 -> 10 등)
# df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x) else np.nan)

# # "Actual_numeric"와 "Final" 처리
# df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
# df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

# # --------------------------------------------------
# # 5. 사이드바 위젯 (필터)
# # --------------------------------------------------
# st.sidebar.header("Filter Options")

# # KPI 목록
# kpi_options = sorted(list(df["KPI"].unique()))
# if "Final score" not in kpi_options:
#     kpi_options.append("Final score")
# selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)

# # 팀 목록
# team_list = sorted(df["Team"].unique())
# team_list_extended = team_list.copy()
# if "HWK Total" not in team_list_extended:
#     team_list_extended.append("HWK Total")

# selected_teams = st.sidebar.multiselect(trans["select_teams"][lang], options=team_list_extended, default=team_list)

# # 전체 주차의 최소/최대
# min_week = int(df["Week_num"].min())
# max_week = int(df["Week_num"].max())

# selected_week_range = st.sidebar.slider(
#     trans["select_week_range"][lang],
#     min_week,
#     max_week,
#     (min_week, max_week),
#     step=1
# )

# selected_team_detail = st.sidebar.selectbox(trans["select_team_details"][lang], options=team_list_extended, index=0)

# # --------------------------------------------------
# # 6. 데이터 필터링 (KPI, 주차 범위 적용)
# # --------------------------------------------------
# if selected_kpi == "Final score":
#     df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & 
#                      (df["Week_num"] <= selected_week_range[1])].copy()
# else:
#     df_filtered = df[(df["KPI"] == selected_kpi) & 
#                      (df["Week_num"] >= selected_week_range[0]) & 
#                      (df["Week_num"] <= selected_week_range[1])].copy()

# # KPI != Final score일 때 최신 주차
# if not df_filtered.empty:
#     latest_week = int(df_filtered["Week_num"].max())
# else:
#     latest_week = None

# # --------------------------------------------------
# # 7. [1] KPI Performance Comparison by Team (바 차트)
# # --------------------------------------------------
# st.markdown(trans["kpi_comparison"][lang])
# if selected_kpi == "Final score":
#     # 팀별 Final 값 합산
#     df_latest = df_filtered.groupby("Team").agg({"Final": "sum"}).reset_index()
#     df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
#     # HWK Total 포함 시 전체 합계
#     if "HWK Total" in selected_teams:
#         overall_final = df_latest["Final"].sum()
#         df_total = pd.DataFrame({
#             "Team": ["HWK Total"],
#             "Final": [overall_final]
#         })
#         df_total["Label"] = df_total.apply(format_final_label, axis=1)
#         df_latest = pd.concat([df_latest, df_total], ignore_index=True)
# else:
#     # 최신 주차 데이터만
#     df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()
#     # HWK Total 포함 시 전체 평균
#     if "HWK Total" in selected_teams and not df_latest.empty:
#         overall_actual = df_latest["Actual_numeric"].mean()
#         overall_final = round(df_latest["Final"].mean())
#         overall_unit = extract_unit(df_latest.iloc[0]["Actual"])
#         df_total = pd.DataFrame({
#             "Team": ["HWK Total"],
#             "Actual_numeric": [overall_actual],
#             "Final": [overall_final],
#             "Actual": [f"{overall_actual:.2f}{overall_unit}"],
#             "Week_num": [latest_week],
#             "KPI": [selected_kpi]
#         })
#         df_latest = pd.concat([df_latest, df_total], ignore_index=True)

# df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
# if selected_kpi != "Final score":
#     df_comp["Label"] = df_comp.apply(format_label, axis=1)

# if selected_kpi == "Final score":
#     fig_bar = px.bar(
#         df_comp,
#         x="Team",
#         y="Final",
#         text="Label",
#         labels={"Final": "Final score by Team"}
#     )
#     fig_bar.update_traces(texttemplate="%{text}", textposition='inside')
# else:
#     fig_bar = px.bar(
#         df_comp,
#         x="Team",
#         y="Actual_numeric",
#         text="Label",
#         labels={"Actual_numeric": trans["avg_by_team"][lang].format(kpi=selected_kpi)}
#     )
#     fig_bar.update_traces(texttemplate="%{text}", textposition='inside')

# st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

# # --------------------------------------------------
# # 8. [2] Weekly Performance Trend Analysis (라인 차트)
# # --------------------------------------------------
# st.markdown(trans["weekly_trend"][lang])

# if selected_kpi == "Final score":
#     df_trend_individual = df_filtered.sort_values("Week_num").groupby("Team").apply(
#         lambda x: x.assign(CumFinal=x["Final"].cumsum())
#     ).reset_index(drop=True)

#     fig_line = px.line(
#         df_trend_individual,
#         x="Week_num",
#         y="CumFinal",
#         color="Team",
#         markers=True,
#         labels={"Week_num": "Week", "CumFinal": "Cumulative Final score"},
#         title="Weekly Trend of Final score (Cumulative)"
#     )
#     fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)

#     if "HWK Total" in selected_teams:
#         df_overall_trend = df_filtered.sort_values("Week_num").groupby("Week_num").agg({"Final": "sum"}).reset_index()
#         df_overall_trend["CumFinal"] = df_overall_trend["Final"].cumsum()
#         fig_line.add_scatter(
#             x=df_overall_trend["Week_num"],
#             y=df_overall_trend["CumFinal"],
#             mode='lines+markers',
#             name="HWK Total",
#             line=dict(color='black', dash='dash')
#         )
# else:
#     if selected_kpi.lower() == "b-grade":
#         y_label = "b-grade (%)"
#     else:
#         y_label = f"{selected_kpi} Value"

#     df_trend_individual = df_filtered[df_filtered["Team"].isin([t for t in selected_teams if t != "HWK Total"])].copy()
#     fig_line = px.line(
#         df_trend_individual,
#         x="Week_num",
#         y="Actual_numeric",
#         color="Team",
#         markers=True,
#         labels={"Week_num": "Week", "Actual_numeric": y_label},
#         title=trans["weekly_trend_title"][lang].format(kpi=selected_kpi)
#     )
#     fig_line.update_xaxes(tickmode='linear', tick0=selected_week_range[0], dtick=1)

#     if "HWK Total" in selected_teams:
#         df_overall_trend = df_filtered.groupby("Week_num").agg({"Actual_numeric": "mean", "Final": "mean"}).reset_index()
#         fig_line.add_scatter(
#             x=df_overall_trend["Week_num"],
#             y=df_overall_trend["Actual_numeric"],
#             mode='lines+markers',
#             name="HWK Total",
#             line=dict(color='black', dash='dash')
#         )

# st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

# # --------------------------------------------------
# # 9. [3] KPI Top/Bottom Team Rankings (Top 3 / Bottom 3)
# # --------------------------------------------------
# st.markdown(trans["top_bottom_rankings"][lang])

# df_rank = df_comp[df_comp["Team"] != "HWK Total"].copy()
# if selected_kpi == "Final score":
#     df_rank = df_rank.sort_values("Final", ascending=False)
# else:
#     df_rank = df_rank.sort_values("Actual_numeric", ascending=False)

# top_n = 3 if len(df_rank) >= 3 else len(df_rank)
# bottom_n = 3 if len(df_rank) >= 3 else len(df_rank)

# top_df = df_rank.head(top_n).copy()
# bottom_df = df_rank.tail(bottom_n).copy()

# if selected_kpi == "Final score":
#     bottom_df = bottom_df.sort_values("Final", ascending=True)
# else:
#     bottom_df = bottom_df.sort_values("Actual_numeric", ascending=True)

# if selected_kpi == "Final score":
#     top_df["Label"] = top_df.apply(lambda row: format_final_label(row), axis=1)
#     bottom_df["Label"] = bottom_df.apply(lambda row: format_final_label(row), axis=1)
# else:
#     top_df["Label"] = top_df.apply(format_label, axis=1)
#     bottom_df["Label"] = bottom_df.apply(format_label, axis=1)

# col1, col2 = st.columns(2)
# with col1:
#     st.subheader(trans["top_teams"][lang].format(n=top_n, kpi=selected_kpi))
#     if selected_kpi == "Final score":
#         fig_top = px.bar(
#             top_df,
#             x="Final",
#             y="Team",
#             orientation="h",
#             text="Label",
#             labels={"Final": "Final score", "Team": "Team"}
#         )
#     else:
#         fig_top = px.bar(
#             top_df,
#             x="Actual_numeric",
#             y="Team",
#             orientation="h",
#             text="Label",
#             labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
#         )
#     fig_top.update_traces(texttemplate="%{text}", textposition='inside')
#     fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
#     st.plotly_chart(fig_top, use_container_width=True, key="top_chart")

# with col2:
#     st.subheader(trans["bottom_teams"][lang].format(n=bottom_n, kpi=selected_kpi))
#     if selected_kpi == "Final score":
#         fig_bottom = px.bar(
#             bottom_df,
#             x="Final",
#             y="Team",
#             orientation="h",
#             text="Label",
#             labels={"Final": "Final score", "Team": "Team"}
#         )
#     else:
#         fig_bottom = px.bar(
#             bottom_df,
#             x="Actual_numeric",
#             y="Team",
#             orientation="h",
#             text="Label",
#             labels={"Actual_numeric": f"Avg {selected_kpi} Value", "Team": "Team"}
#         )
#     fig_bottom.update_traces(texttemplate="%{text}", textposition='inside', marker_color='red')
#     fig_bottom.update_layout(yaxis={'categoryorder': 'total ascending'})
#     st.plotly_chart(fig_bottom, use_container_width=True, key="bottom_chart")

# # --------------------------------------------------
# # 10. [4] Team-Specific KPI Detailed View (카드형 레이아웃)
# # --------------------------------------------------
# st.markdown("")
# if selected_team_detail != "HWK Total":
#     df_team = df[df["Team"] == selected_team_detail].copy()
# else:
#     # HWK Total을 선택하면 latest_week 데이터만 모아서 평균
#     df_team = df[df["Week_num"] == latest_week].groupby("KPI").agg({
#         "Actual_numeric": "mean",
#         "Final": "mean",
#         "Actual": "first"
#     }).reset_index()

# # (A) Last Week performance Details
# if latest_week is not None:
#     st.markdown(
#         f"<div style='font-size:18px; font-weight:bold;'>{trans['last_week_details'][lang].format(team=selected_team_detail, week=latest_week)}</div>",
#         unsafe_allow_html=True
#     )

#     cols = st.columns(3)
#     i = 0
#     for kpi in df_team["KPI"].unique():
#         if selected_team_detail != "HWK Total":
#             df_last = df_team[(df_team["Week_num"] == latest_week) & (df_team["KPI"] == kpi)]
#             df_prev = df_team[(df_team["Week_num"] == (latest_week - 1)) & (df_team["KPI"] == kpi)]
#         else:
#             df_last = df_team[df_team["KPI"] == kpi]
#             df_prev_raw = df[df["Week_num"] == (latest_week - 1)].groupby("KPI").agg({
#                 "Actual_numeric": "mean",
#                 "Final": "mean",
#                 "Actual": "first"
#             }).reset_index()
#             df_prev = df_prev_raw[df_prev_raw["KPI"] == kpi]

#         if not df_last.empty:
#             row_last = df_last.iloc[0]
#         else:
#             continue

#         if selected_team_detail != "HWK Total":
#             current_label = format_label(row_last)
#         else:
#             current_label = f"{row_last['Actual_numeric']:.2f}{extract_unit(row_last['Actual'])} ({int(round(row_last['Final']))} point)"

#         if not df_prev.empty:
#             row_prev = df_prev.iloc[0]
#             if pd.notna(row_last["Actual_numeric"]) and pd.notna(row_prev["Actual_numeric"]):
#                 delta_actual = row_last["Actual_numeric"] - row_prev["Actual_numeric"]
#             else:
#                 delta_actual = None

#             if pd.notna(row_last["Final"]) and pd.notna(row_prev["Final"]):
#                 delta_final = int(round(row_last["Final"])) - int(round(row_prev["Final"]))
#             else:
#                 delta_final = None

#             if delta_actual is not None and delta_final is not None:
#                 kpi_lower = kpi.lower()
#                 positive_better = ["prs validation", "6s_audit"]
#                 if kpi_lower in positive_better:
#                     arrow = "▲" if delta_actual > 0 else "▼" if delta_actual < 0 else ""
#                 else:
#                     arrow = "▲" if delta_actual < 0 else "▼" if delta_actual > 0 else ""
#                 delta_str = f"{arrow}{delta_actual:+.2f}%({delta_final:+d} point)"
#             else:
#                 delta_str = "N/A"
#         else:
#             delta_str = "N/A"
#             delta_actual = None

#         delta_color = get_delta_color(kpi, delta_actual)
#         render_custom_metric(cols[i % 3], kpi, current_label, delta_str, delta_color)
#         i += 1

# # (B) Total Week Performance Detail
# st.markdown("")
# st.markdown(
#     f"<div style='font-size:18px; font-weight:bold;'>{trans['total_week_details'][lang].format(team=selected_team_detail)}</div>",
#     unsafe_allow_html=True
# )

# if selected_team_detail != "HWK Total":
#     df_cum = df[(df["Team"] == selected_team_detail) & 
#                 (df["Week_num"] >= selected_week_range[0]) & 
#                 (df["Week_num"] <= selected_week_range[1])]
# else:
#     df_cum = df[(df["Week_num"] >= selected_week_range[0]) & 
#                 (df["Week_num"] <= selected_week_range[1])]

# df_cum_group = df_cum.groupby("KPI").apply(lambda x: cumulative_performance(x, x["KPI"].iloc[0])).reset_index(name="cum")

# cols_total = st.columns(3)
# i = 0
# for kpi in df_cum_group["KPI"].unique():
#     sub_df = df_cum[df_cum["KPI"] == kpi]
#     cum_value = cumulative_performance(sub_df, kpi)

#     team_cum = df[(df["KPI"] == kpi) & 
#                   (df["Week_num"] >= selected_week_range[0]) & 
#                   (df["Week_num"] <= selected_week_range[1])].groupby("Team").apply(lambda x: cumulative_performance(x, kpi)).reset_index(name="cum")

#     kpi_lower = kpi.lower()
#     positive_better = ["prs validation", "6s_audit"]
#     negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]

#     if kpi_lower in positive_better:
#         best_value = team_cum["cum"].max() if not team_cum.empty else 0
#         delta = cum_value - best_value
#         arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""
#     elif kpi_lower in negative_better:
#         best_value = team_cum["cum"].min() if not team_cum.empty else 0
#         delta = cum_value - best_value
#         arrow = "▲" if delta < 0 else "▼" if delta > 0 else ""
#     else:
#         best_value = team_cum["cum"].max() if not team_cum.empty else 0
#         delta = cum_value - best_value
#         arrow = "▲" if delta > 0 else "▼" if delta < 0 else ""

#     delta_str = f"{arrow}{abs(delta):+.2f} point" if best_value != 0 else ""
#     delta_color = get_delta_color(kpi, delta)

#     render_custom_metric(cols_total[i % 3], kpi, f"{cum_value:.2f}", delta_str, delta_color)
#     i += 1

# # --------------------------------------------------
# # 11. [5] Detailed Data Table (행과 열 전환: 행=주차, 열=KPI)
# # --------------------------------------------------
# st.markdown("")
# st.markdown(trans["detailed_data"][lang])

# kpi_all = sorted(df["KPI"].unique())
# all_weeks = sorted(df["Week_num"].dropna().unique())

# data_table = {}
# for kpi in kpi_all:
#     row_data = {}
#     values = []
#     finals = []
#     unit = ""
#     for w in all_weeks:
#         if selected_team_detail != "HWK Total":
#             sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
#             if not sub_df.empty:
#                 val = sub_df.iloc[0]["Actual_numeric"]
#                 final_val = sub_df.iloc[0]["Final"]
#                 unit = extract_unit(sub_df.iloc[0]["Actual"])
#                 formatted = f"{val:.2f}{unit}<br>({final_val} point)"
#                 row_data[f"Week {int(w)}"] = formatted
#                 values.append(val)
#                 finals.append(final_val)
#             else:
#                 row_data[f"Week {int(w)}"] = "N/A"
#         else:
#             sub_df = df[(df["KPI"] == kpi) & (df["Week_num"] == w)]
#             if not sub_df.empty:
#                 val = sub_df["Actual_numeric"].mean()
#                 final_val = sub_df["Final"].mean()
#                 unit = extract_unit(sub_df.iloc[0]["Actual"])
#                 formatted = f"{val:.2f}{unit}<br>({final_val:.2f} point)"
#                 row_data[f"Week {int(w)}"] = formatted
#                 values.append(val)
#                 finals.append(final_val)
#             else:
#                 row_data[f"Week {int(w)}"] = "N/A"

#     if values:
#         avg_val = sum(values) / len(values)
#         avg_final = sum(finals) / len(finals) if finals else 0
#         row_data["Average"] = f"{avg_val:.2f}{unit}<br>({avg_final:.2f} point)"
#     else:
#         row_data["Average"] = "N/A"

#     data_table[kpi] = row_data

# table_df = pd.DataFrame(data_table)

# # Week X와 Average 순서대로 행 재배치
# index_order = [f"Week {int(w)}" for w in all_weeks] + ["Average"]
# table_df = table_df.reindex(index_order)

# # 인덱스 이름을 다국어로 교체
# new_index = {}
# for idx in table_df.index:
#     if idx.startswith("Week"):
#         week_num = idx.split()[1]
#         new_index[idx] = trans["week_col"][lang].format(week=week_num)
#     elif idx == "Average":
#         new_index[idx] = trans["average"][lang]
#     else:
#         new_index[idx] = idx

# table_df.rename(index=new_index, inplace=True)

# st.markdown(table_df.to_html(escape=False), unsafe_allow_html=True)

# #위에가 기존의, 잘 작동하던 코드임.

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

def remove_all_spaces(s: str) -> str:
    """모든 유니코드 공백(스페이스, 탭, NBSP 등)을 제거"""
    return re.sub(r'\s+', '', s)

def to_halfwidth(s: str) -> str:
    """전각(Fullwidth) 문자를 반각(ASCII) 문자로 변환"""
    return unicodedata.normalize('NFKC', s)

@st.cache_data
def load_data():
    # CSV 파일 인코딩을 상황에 맞춰 지정 (예: 'utf-8', 'cp949' 등)
    df = pd.read_csv("score.csv", sep="\t", encoding="utf-8")
    return df

def convert_to_numeric(x):
    try:
        if isinstance(x, str):
            if x.strip() == '-' or x.strip() == '':
                return np.nan
            # 쉼표를 소수점으로 변환 (예: "0,02%" -> "0.02%")
            x = x.replace(",", ".")
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
    # shortage_cost는 누적합, 나머지는 평균 처리 (기존 로직)
    if kpi.lower() == "shortage_cost":
        return sub_df["Actual_numeric"].sum()
    else:
        return sub_df["Actual_numeric"].mean()

# --- 새로 추가: 화살표 대신 이모티콘 반환 함수 ---
def get_trend_emoticon(kpi, delta):
    """
    모든 KPI에서 '위로 가는' 화살표(증가/감소)는 '😀'(긍정),
    '아래로 가는' 화살표는 '😡'(부정)으로 표시.
    
    단, KPI가 양(positive) 지표인지 음(negative) 지표인지에 따라
    delta의 부호가 '개선'을 의미하는지 달라짐.
    """
    if delta is None:
        return ""  # 변화 없음
    
    kpi_lower = kpi.lower()
    positive_better = ["prs validation", "6s_audit", "final score"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]
    
    # 양 지표: delta > 0 -> 😀, delta < 0 -> 😡
    if kpi_lower in positive_better:
        if delta > 0:
            return "😀"
        elif delta < 0:
            return "😡"
        else:
            return ""
    # 음 지표: delta < 0 -> 😀, delta > 0 -> 😡
    elif kpi_lower in negative_better:
        if delta < 0:
            return "😀"
        elif delta > 0:
            return "😡"
        else:
            return ""
    else:
        # 기타 KPI는 양 지표로 간주
        if delta > 0:
            return "😀"
        elif delta < 0:
            return "😡"
        else:
            return ""

def render_custom_metric(col, label, value, delta_str, color="black"):
    """
    실제 렌더링 시 delta_str 안에 이모티콘과 수치가 모두 포함되도록 처리.
    color는 일단 black으로 고정 (원하면 색상 변경 가능)
    """
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

# --------------------------------------------------
# 4. 데이터 로드 및 전처리
# --------------------------------------------------
df = load_data()

# Week 컬럼 전처리 (전각→반각 + 유니코드 공백 제거 + 대문자화)
df["Week"] = (
    df["Week"]
    .astype(str)
    .apply(to_halfwidth)        # 전각 -> 반각 변환
    .str.upper()                # 대문자
    .apply(remove_all_spaces)   # 모든 유니코드 공백 제거
)

# 숫자만 추출하여 Week_num에 저장 (W5 -> 5, W10 -> 10 등)
df["Week_num"] = df["Week"].apply(lambda x: int(re.sub(r'\D', '', x)) if re.sub(r'\D', '', x) else np.nan)

# "Actual_numeric"와 "Final" 처리
df["Actual_numeric"] = df["Actual"].apply(convert_to_numeric)
df["Final"] = pd.to_numeric(df["Final"], errors="coerce")

# --------------------------------------------------
# 5. 사이드바 위젯 (필터)
# --------------------------------------------------
st.sidebar.header("Filter Options")

# KPI 목록
kpi_options = sorted(list(df["KPI"].unique()))
if "Final score" not in kpi_options:
    kpi_options.append("Final score")
selected_kpi = st.sidebar.selectbox(trans["select_kpi"][lang], options=kpi_options)

# 팀 목록
team_list = sorted(df["Team"].unique())
team_list_extended = team_list.copy()
if "HWK Total" not in team_list_extended:
    team_list_extended.append("HWK Total")

selected_teams = st.sidebar.multiselect(trans["select_teams"][lang], options=team_list_extended, default=team_list)

# 전체 주차의 최소/최대
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
    df_filtered = df[(df["Week_num"] >= selected_week_range[0]) & 
                     (df["Week_num"] <= selected_week_range[1])].copy()
else:
    df_filtered = df[(df["KPI"] == selected_kpi) & 
                     (df["Week_num"] >= selected_week_range[0]) & 
                     (df["Week_num"] <= selected_week_range[1])].copy()

if not df_filtered.empty:
    latest_week = int(df_filtered["Week_num"].max())
else:
    latest_week = None

# --------------------------------------------------
# 7. [1] KPI Performance Comparison by Team (바 차트)
# --------------------------------------------------
st.markdown(trans["kpi_comparison"][lang])
if selected_kpi == "Final score":
    # 팀별 Final 값 합산
    df_latest = df_filtered.groupby("Team").agg({"Final": "sum"}).reset_index()
    df_latest["Label"] = df_latest.apply(format_final_label, axis=1)
    # HWK Total 포함 시 전체 합계
    if "HWK Total" in selected_teams:
        overall_final = df_latest["Final"].sum()
        df_total = pd.DataFrame({
            "Team": ["HWK Total"],
            "Final": [overall_final]
        })
        df_total["Label"] = df_total.apply(format_final_label, axis=1)
        df_latest = pd.concat([df_latest, df_total], ignore_index=True)
else:
    # 최신 주차 데이터만
    df_latest = df_filtered[df_filtered["Week_num"] == latest_week].copy()
    # HWK Total 포함 시 전체 평균
    if "HWK Total" in selected_teams and not df_latest.empty:
        overall_actual = df_latest["Actual_numeric"].mean()
        overall_final = round(df_latest["Final"].mean())
        overall_unit = extract_unit(df_latest.iloc[0]["Actual"])
        df_total = pd.DataFrame({
            "Team": ["HWK Total"],
            "Actual_numeric": [overall_actual],
            "Final": [overall_final],
            "Actual": [f"{overall_actual:.2f}{overall_unit}"],
            "Week_num": [latest_week],
            "KPI": [selected_kpi]
        })
        df_latest = pd.concat([df_latest, df_total], ignore_index=True)

df_comp = df_latest[df_latest["Team"].isin(selected_teams)].copy()
if selected_kpi != "Final score":
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

# (A) 해당 팀 데이터 준비
if selected_team_detail != "HWK Total":
    df_team = df[df["Team"] == selected_team_detail].copy()
else:
    # HWK Total: 최신 주 데이터만 모아서 KPI별 평균(또는 shortage_cost 제외) 
    # (단, shortage_cost는 아래 로직에서 별도 처리 가능하지만, 일단 기본 구조 유지)
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

        # ---- HWK Total & Shortage Cost인 경우: sum으로 집계해서 비교 ----
        if selected_team_detail == "HWK Total" and kpi_lower == "shortage_cost":
            # 최신 주(Week=latest_week) shortage cost 총합
            df_last_raw = df[(df["Week_num"] == latest_week) & (df["KPI"].str.lower() == "shortage_cost")]
            latest_sum = df_last_raw["Actual_numeric"].sum() if not df_last_raw.empty else np.nan
            current_label = f"{latest_sum:.2f} (Week {latest_week} total)"

            # 이전 주(Week=latest_week-1) shortage cost 총합
            df_prev_raw = df[(df["Week_num"] == (latest_week - 1)) & (df["KPI"].str.lower() == "shortage_cost")]
            prev_sum = df_prev_raw["Actual_numeric"].sum() if not df_prev_raw.empty else np.nan

            if pd.notna(latest_sum) and pd.notna(prev_sum):
                delta_actual = latest_sum - prev_sum
            else:
                delta_actual = None

            # shortage_cost는 Final 점수도 평균일 뿐이므로, 여기서는 delta_final 대신 실제 비용 증감만 표시
            # delta_str 만들기
            emoticon = get_trend_emoticon(kpi, delta_actual)
            if delta_actual is not None and delta_actual != 0:
                delta_str = f"{emoticon}{delta_actual:+.2f}"
            else:
                delta_str = "N/A"

            render_custom_metric(cols[i % 3], kpi, current_label, delta_str)
            i += 1
            continue
        # ---- 일반 케이스 (HWK Total이 아니거나 shortage_cost가 아닌 경우) ----
        if selected_team_detail != "HWK Total":
            df_last = df_team[(df_team["Week_num"] == latest_week) & (df_team["KPI"] == kpi)]
            df_prev = df_team[(df_team["Week_num"] == (latest_week - 1)) & (df_team["KPI"] == kpi)]
        else:
            # HWK Total & 다른 KPI
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

        # current_label
        if selected_team_detail != "HWK Total":
            current_label = format_label(row_last)
        else:
            # HWK Total & KPI != shortage_cost
            current_label = f"{row_last['Actual_numeric']:.2f}{extract_unit(row_last['Actual'])} ({int(round(row_last['Final']))} point)"

        # delta 계산
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
                # 예) 😀+1.50%(+2 point)
                # KPI가 % 단위인지 아닌지는 row_last["Actual"]에서 확인 가능
                unit = extract_unit(row_last["Actual"]) if pd.notnull(row_last.get("Actual", "")) else ""
                # 편의상 %가 있으면 % 표시, 아니면 그냥 수치
                if unit.strip() == "%":
                    delta_str = f"{emoticon}{delta_actual:+.2f}{unit}({delta_final:+d} point)"
                else:
                    delta_str = f"{emoticon}{delta_actual:+.2f}({delta_final:+d} point)"
            else:
                delta_str = "N/A"
        else:
            delta_str = "N/A"

        render_custom_metric(cols[i % 3], kpi, current_label, delta_str)
        i += 1

# (B) Total Week Performance Detail
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
    kpi_lower = kpi.lower()

    # --- HWK Total & shortage_cost인 경우: 최신 주 합계 vs 전체 주 평균 ---
    if selected_team_detail == "HWK Total" and kpi_lower == "shortage_cost":
        # (1) 최신 주 shortage_cost 합계
        if latest_week is not None:
            df_latest_sc = df[(df["Week_num"] == latest_week) & (df["KPI"].str.lower() == "shortage_cost")]
            latest_sum = df_latest_sc["Actual_numeric"].sum() if not df_latest_sc.empty else np.nan
        else:
            latest_sum = np.nan

        # (2) 전체 주차 범위 내 shortage_cost 합계 & 평균 주차 비용
        df_all_sc = df_cum[df_cum["KPI"].str.lower() == "shortage_cost"]
        total_sum_sc = df_all_sc["Actual_numeric"].sum() if not df_all_sc.empty else np.nan
        # 주차 수
        unique_weeks = df_all_sc["Week_num"].unique() if not df_all_sc.empty else []
        weeks_count = len(unique_weeks)
        if weeks_count > 0:
            avg_weekly_sc = total_sum_sc / weeks_count
        else:
            avg_weekly_sc = np.nan

        # cum_value: 최신 주 shortage cost 합계
        cum_value = latest_sum
        # 비교 대상: 평균 주 비용
        best_value = avg_weekly_sc

        if pd.notna(cum_value) and pd.notna(best_value):
            delta = cum_value - best_value
        else:
            delta = None

        emoticon = get_trend_emoticon(kpi, delta)
        if delta is not None:
            delta_str = f"{emoticon}{delta:+.2f}"
        else:
            delta_str = "N/A"

        render_custom_metric(cols_total[i % 3], kpi, f"{cum_value:.2f}", delta_str)
        i += 1
        continue

    # --- 일반 케이스 ---
    sub_df = df_cum[df_cum["KPI"] == kpi]
    cum_value = cumulative_performance(sub_df, kpi)

    team_cum = df[(df["KPI"] == kpi) & 
                  (df["Week_num"] >= selected_week_range[0]) & 
                  (df["Week_num"] <= selected_week_range[1])].groupby("Team").apply(lambda x: cumulative_performance(x, kpi)).reset_index(name="cum")

    positive_better = ["prs validation", "6s_audit", "final score"]
    negative_better = ["aql_performance", "b-grade", "attendance", "issue_tracking", "shortage_cost"]

    if kpi_lower in positive_better:
        best_value = team_cum["cum"].max() if not team_cum.empty else 0
        delta = cum_value - best_value
    elif kpi_lower in negative_better:
        best_value = team_cum["cum"].min() if not team_cum.empty else 0
        delta = cum_value - best_value
    else:
        best_value = team_cum["cum"].max() if not team_cum.empty else 0
        delta = cum_value - best_value

    emoticon = get_trend_emoticon(kpi, delta)
    if best_value != 0 and pd.notna(delta):
        delta_str = f"{emoticon}{delta:+.2f} point"
    else:
        delta_str = ""

    render_custom_metric(cols_total[i % 3], kpi, f"{cum_value:.2f}", delta_str)
    i += 1

# --------------------------------------------------
# 11. [5] Detailed Data Table (행과 열 전환: 행=주차, 열=KPI)
# --------------------------------------------------
st.markdown("")
st.markdown(trans["detailed_data"][lang])

kpi_all = sorted(df["KPI"].unique())
all_weeks = sorted(df["Week_num"].dropna().unique())

data_table = {}
for kpi in kpi_all:
    row_data = {}
    values = []
    finals = []
    unit = ""
    for w in all_weeks:
        if selected_team_detail != "HWK Total":
            sub_df = df[(df["KPI"] == kpi) & (df["Team"] == selected_team_detail) & (df["Week_num"] == w)]
            if not sub_df.empty:
                val = sub_df.iloc[0]["Actual_numeric"]
                final_val = sub_df.iloc[0]["Final"]
                unit = extract_unit(sub_df.iloc[0]["Actual"])
                formatted = f"{val:.2f}{unit}<br>({final_val} point)"
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
                unit = extract_unit(sub_df.iloc[0]["Actual"])
                formatted = f"{val:.2f}{unit}<br>({final_val:.2f} point)"
                row_data[f"Week {int(w)}"] = formatted
                values.append(val)
                finals.append(final_val)
            else:
                row_data[f"Week {int(w)}"] = "N/A"

    if values:
        avg_val = sum(values) / len(values)
        avg_final = sum(finals) / len(finals) if finals else 0
        row_data["Average"] = f"{avg_val:.2f}{unit}<br>({avg_final:.2f} point)"
    else:
        row_data["Average"] = "N/A"

    data_table[kpi] = row_data

table_df = pd.DataFrame(data_table)

# Week X와 Average 순서대로 행 재배치
index_order = [f"Week {int(w)}" for w in all_weeks] + ["Average"]
table_df = table_df.reindex(index_order)

# 인덱스 이름을 다국어로 교체
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
