import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go

# ----------------------------------------------------
# 0) CSV Load & Preprocessing
# ----------------------------------------------------
@st.cache_data
def load_data():
    """
    Reads 'score.csv' and preprocesses:
      - Keep columns: Week, Team, KPI, Actual, Final
      - Remove non-numeric characters from Actual
      - Convert Actual, Final to float
      - Drop rows where Actual or Final is NaN
      - Extract numeric part of Week (W1 -> 1) as Week_num
    """
    # You may need to adjust this path according to your GitHub/Streamlit structure
    csv_path = "score.csv"
    df = pd.read_csv(csv_path)

    # Extract only numbers, '.' or '-' from Actual
    df['Actual'] = (
        df['Actual'].astype(str)
        .str.replace(r'[^0-9.\-]', '', regex=True)
    )
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')

    df['Final'] = pd.to_numeric(df['Final'], errors='coerce')

    # Drop rows if Actual or Final is NaN
    df.dropna(subset=['Actual', 'Final'], inplace=True)

    # Extract integer part from Week (W1 -> 1)
    df['Week_num'] = df['Week'].str.extract(r'(\d+)').astype(int)

    return df

# ----------------------------------------------------
# 1) Load data
# ----------------------------------------------------
df = load_data()

# ----------------------------------------------------
# 2) Define lists
# ----------------------------------------------------
all_kpis = sorted(df['KPI'].dropna().unique())
all_teams = sorted(df['Team'].dropna().unique())
unique_weeks = sorted(df['Week'].unique(), key=lambda w: int(w[1:]))

week_options = unique_weeks + ["Total"]

# ----------------------------------------------------
# 3) Sidebar filters (common)
# ----------------------------------------------------
st.sidebar.title("Common Filters")
selected_kpi = st.sidebar.selectbox("Select a KPI (Target)", options=all_kpis)
selected_week_or_total = st.sidebar.selectbox("Select Week (or 'Total')", options=week_options)
selected_team = st.sidebar.selectbox("Select a Team", options=all_teams)

# ----------------------------------------------------
# 4) Pivot - Final (sum) & Actual (mean) for selected_kpi
# ----------------------------------------------------
df_kpi_final = df[df['KPI'] == selected_kpi].copy()
grouped_final = df_kpi_final.groupby(['Week','Team'], as_index=False)['Final'].sum()
pivot_final = grouped_final.pivot(index='Team', columns='Week', values='Final').fillna(0)
pivot_final['Total'] = pivot_final.sum(axis=1)

df_kpi_act = df[df['KPI'] == selected_kpi].copy()
grouped_act = df_kpi_act.groupby(['Week','Team'], as_index=False)['Actual'].mean()
pivot_act = grouped_act.pivot(index='Team', columns='Week', values='Actual').fillna(0)
pivot_act['Average'] = pivot_act.mean(axis=1)

# ----------------------------------------------------
# Row1: "Target : ... best score : ... No. 1 team : ..."
# ----------------------------------------------------
st.subheader("Row1: Max Score Info")
if pivot_final.empty:
    st.write("No Final data.")
else:
    max_total = pivot_final['Total'].max()
    max_team = pivot_final['Total'].idxmax()
    st.markdown(f"""Target : {selected_kpi}  
best score : {max_total:.2f}  
No. 1 team : {max_team}""")

# ----------------------------------------------------
# Row2: Final Score chart
# ----------------------------------------------------
st.subheader("Row2: Final Score Chart")
if pivot_final.empty or (selected_week_or_total not in pivot_final.columns):
    st.write("No data for this chart.")
else:
    bar_df_final = pivot_final[[selected_week_or_total]].reset_index()
    bar_df_final.columns = ['Team', 'Final_Value']
    chart_final = (
        alt.Chart(bar_df_final)
        .mark_bar()
        .encode(
            x=alt.X('Team:N', sort=None),
            y=alt.Y('Final_Value:Q'),
            tooltip=['Team','Final_Value']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart_final, use_container_width=False)

# ----------------------------------------------------
# Row3: Final table
# ----------------------------------------------------
st.subheader("Row3: Final Table")
if pivot_final.empty:
    st.write("No Final data.")
else:
    st.dataframe(pivot_final.style.format(precision=2))

# ----------------------------------------------------
# Row4: Actual chart
# ----------------------------------------------------
st.subheader("Row4: Actual Chart")
actual_col = selected_week_or_total
if actual_col == "Total":
    actual_col = "Average"
if pivot_act.empty or (actual_col not in pivot_act.columns):
    st.write("No data for this chart.")
else:
    bar_df_act = pivot_act[[actual_col]].reset_index()
    bar_df_act.columns = ['Team','Actual_Value']
    chart_act = (
        alt.Chart(bar_df_act)
        .mark_bar(color='orange')
        .encode(
            x=alt.X('Team:N', sort=None),
            y=alt.Y('Actual_Value:Q'),
            tooltip=['Team','Actual_Value']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart_act, use_container_width=False)

# ----------------------------------------------------
# Row5: Actual table
# ----------------------------------------------------
st.subheader("Row5: Actual Table")
if pivot_act.empty:
    st.write("No Actual data.")
else:
    st.dataframe(pivot_act.style.format(precision=2))

# ----------------------------------------------------
# Row6: Rank chart (using selected_team)
# ----------------------------------------------------
st.subheader("Row6: Rank Chart")
df_rank = df.copy()
df_rank['Rank'] = df_rank.groupby(['Week','KPI'])['Final'].rank(method='dense', ascending=False)
df_team_rank = df_rank[df_rank['Team'] == selected_team].copy()
df_team_rank = df_team_rank.sort_values(['KPI','Week_num'])

if df_team_rank.empty:
    st.write("No Rank data for the selected team.")
else:
    line_chart = (
        alt.Chart(df_team_rank)
        .mark_line(point=True)
        .encode(
            x=alt.X('Week_num:Q', axis=alt.Axis(title='Weeknum')),
            y=alt.Y('Rank:Q', sort='descending', axis=alt.Axis(title='Rank (1=best)')),
            color=alt.Color('KPI:N', legend=alt.Legend(title='KPI')),
            tooltip=['Week','KPI','Final','Rank']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(line_chart, use_container_width=False)

# ----------------------------------------------------
# Row7: Rank table
# ----------------------------------------------------
st.subheader("Row7: Rank Table")
if df_team_rank.empty:
    st.write("No Rank data.")
else:
    rank_pivot = df_team_rank.pivot(index='KPI', columns='Week', values='Rank')
    rank_pivot['Total'] = rank_pivot.sum(axis=1)
    st.dataframe(rank_pivot.fillna("").style.format(precision=2))

# ----------------------------------------------------
# Row8: Radar chart (selected_team vs. overall max)
# ----------------------------------------------------
st.subheader("Row8: Radar Chart")
if df_team_rank.empty:
    st.write("No data for Radar Chart.")
else:
    df_final_only = df.copy()
    team_mean = (df_final_only[df_final_only['Team'] == selected_team]
                 .groupby('KPI')['Final'].mean()
                 .rename('Team_Mean')
                 .reset_index())
    max_final = (df_final_only.groupby('KPI')['Final'].max()
                 .rename('Max_Final')
                 .reset_index())
    radar_df = pd.merge(team_mean, max_final, on='KPI', how='outer').fillna(0)
    if radar_df.empty:
        st.write("No radar data.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_df['Max_Final'],
            theta=radar_df['KPI'],
            fill='toself',
            name='Max'
        ))
        fig.add_trace(go.Scatterpolar(
            r=radar_df['Team_Mean'],
            theta=radar_df['KPI'],
            fill='toself',
            name=f"{selected_team} Mean"
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, radar_df['Max_Final'].max() * 1.1]
                )
            ),
            showlegend=True,
            width=700,
            height=400
        )
        st.plotly_chart(fig, use_container_width=False)

# ----------------------------------------------------
# Row9: The team's Actual table
# ----------------------------------------------------
st.subheader("Row9: The Team's Actual Table")
df_team_act = df[df['Team'] == selected_team].copy()
if df_team_act.empty:
    st.write("No Actual data for this team.")
else:
    team_act_pivot = df_team_act.pivot(index='KPI', columns='Week', values='Actual')
    st.dataframe(team_act_pivot.fillna("").style.format(precision=2))

# ----------------------------------------------------
# Row10: Overall Summation Chart (all KPI)
# ----------------------------------------------------
st.subheader("Row10: Overall Summation Chart")
df_total_all_kpi = df.groupby(['Week','Team'], as_index=False)['Final'].sum()

if df_total_all_kpi.empty:
    st.write("No overall data.")
else:
    chart_overall = (
        alt.Chart(df_total_all_kpi)
        .mark_bar()
        .encode(
            x=alt.X('Team:N', sort='-y'),
            y=alt.Y('Final:Q', stack='zero'),
            color='Week:N',
            tooltip=['Team', 'Week', 'Final']
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart_overall, use_container_width=False)

# ----------------------------------------------------
# Row11: Overall Summation Table & Ranking
# ----------------------------------------------------
st.subheader("Row11: Overall Summation Table")
if df_total_all_kpi.empty:
    st.write("No overall data.")
else:
    pivot_total_all = df_total_all_kpi.pivot(index='Team', columns='Week', values='Final').fillna(0)
    pivot_total_all['GrandTotal'] = pivot_total_all.sum(axis=1)

    st.write("Overall Summation Table (all KPI, all weeks)")
    st.dataframe(pivot_total_all.style.format(precision=2))

    # Sort descending by GrandTotal
    ranking_df = pivot_total_all[['GrandTotal']].copy()
    ranking_df = ranking_df.sort_values('GrandTotal', ascending=False)
    ranking_df['Rank'] = ranking_df['GrandTotal'].rank(method='dense', ascending=False).astype(int)
    st.dataframe(ranking_df.style.format(precision=2))

st.write("---")
st.write("End of layout version 1 by HWK QIP.")
