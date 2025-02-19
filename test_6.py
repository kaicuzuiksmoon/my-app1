# --------------------------------------------------
# 1) 주차 범위 정리 및 필터
# --------------------------------------------------
start_week, end_week = sorted(selected_week_range)

df_filtered = None
if selected_kpi == "Final score":
    df_filtered = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= end_week)].copy()
else:
    df_filtered = df[(df["KPI"] == selected_kpi) &
                     (df["Week_num"] >= start_week) &
                     (df["Week_num"] <= end_week)].copy()

if df_filtered.empty:
    st.warning("선택한 필터에 해당하는 데이터가 없습니다.")
    st.stop()

# latest_week 구하기
latest_week = df_filtered["Week_num"].max()
if pd.isna(latest_week):
    st.warning("해당 주차 범위에 데이터가 없습니다.")
    st.stop()
else:
    latest_week = int(latest_week)

# --------------------------------------------------
# 2) KPI 비교/차트 로직 (df_comp 등)
# --------------------------------------------------
# ... (기존 로직과 동일) ...
# df_latest, df_comp, bar 차트, line 차트 등

# --------------------------------------------------
# 3) Top/Bottom 랭킹
# --------------------------------------------------
if df_comp.empty:
    st.warning("선택된 팀에 대한 데이터가 없습니다. Top/Bottom 분석을 건너뜁니다.")
else:
    # df_rank 계산, 차트 생성 로직
    # ...

# --------------------------------------------------
# 4) (A) 마지막 주 상세보기
# --------------------------------------------------

# 4-1) HWK Total인지 여부에 따라 df_cum 먼저 정의
if selected_team_detail == "HWK Total":
    df_cum = df[(df["Week_num"] >= start_week) & (df["Week_num"] <= end_week)]
    # df_team도 "마지막 주" 기준으로 가공한 값
    df_team = (df_filtered.groupby("KPI")
                         .agg({"Actual_numeric":"mean", "Final":"mean", "Actual":"first"})
                         .reset_index())
else:
    df_cum = df[(df["Team"] == selected_team_detail) &
                (df["Week_num"] >= start_week) &
                (df["Week_num"] <= end_week)]
    df_team = df_cum[df_cum["Week_num"] == latest_week].copy()

# df_team이 비어 있으면 상세보기 스킵
if df_team.empty:
    st.warning(f"{selected_team_detail} 팀은 선택한 주차 범위({start_week}~{end_week})에 데이터가 없습니다.")
else:
    # 여기서부터 마지막 주 카드형 상세보기 로직 진행
    # shortage_cost면 df_cum_sc = df_cum[...] 사용 가능

# --------------------------------------------------
# 5) (B) 전체 주차 상세보기
# --------------------------------------------------
# df_cum 이미 정의됨
if df_cum.empty:
    st.warning(f"{selected_team_detail} 팀은 선택한 주차 범위({start_week}~{end_week})에 누적 데이터가 없습니다.")
else:
    # 누적 성과, rank 계산 등
    # ...

# --------------------------------------------------
# 6) 상세 테이블
# --------------------------------------------------
# 여기서도 df_team 또는 df_cum이 empty면 건너뛰거나 안전하게 처리
