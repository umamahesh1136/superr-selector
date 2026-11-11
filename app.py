import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SupeRR Selector | Clutch Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# CUSTOM CSS FOR SLEEK SPORTS ANALYTICS UI
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.25;
        word-wrap: break-word;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #ec4899;
        margin-top: 4px;
        font-weight: 500;
    }
    .method-box {
        background-color: #0f172a;
        border-left: 4px solid #ec4899;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DATA LOADING & CACHING
# ------------------------------------------------------------------------------
@st.cache_data
def load_base_data():
    candidates = [
        "master_cricket_data.csv.gz",
        "/content/master_cricket_data.csv.gz",
        "/content/master_cricket_data.csv",
        "master_cricket_data.csv"
    ]
    data_path = next((p for p in candidates if os.path.exists(p)), None)
    if not data_path:
        return None, set()

    df = pd.read_csv(data_path, low_memory=False)

    # Extract season year cleanly
    df['season_year'] = pd.to_numeric(
        df['season'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
    )
    if df['season_year'].isna().any() and 'match_date' in df.columns:
        parsed_dates = pd.to_datetime(df['match_date'], errors='coerce', dayfirst=True)
        df['season_year'] = df['season_year'].fillna(parsed_dates.dt.year)

    # Categorize League: IPL vs T20I
    comp_str = df['comp_name'].astype(str) if 'comp_name' in df.columns else pd.Series('', index=df.index)
    df['comp_type'] = np.where(comp_str.str.contains(r'\bipl\b', case=False, na=False), 'IPL', 'T20I')

    # Identify Pak & Ban players ONLY by team affiliation (NEVER by tournament name)
    pak_ban_ids = set()

    if 'is_striker_pak_ban' in df.columns:
        pak_ban_ids.update(df[df['is_striker_pak_ban'] == True]['striker_id'].dropna().unique())
    if 'is_bowler_pak_ban' in df.columns:
        pak_ban_ids.update(df[df['is_bowler_pak_ban'] == True]['bowler_id'].dropna().unique())

    search_root = "/content" if os.path.exists("/content") else "."
    all_csvs = glob.glob(f"{search_root}/**/*.csv", recursive=True)

    mapping_files = [f for f in all_csvs if 'mapping' in os.path.basename(f).lower()]
    teams_files = [
        f for f in all_csvs
        if 'team' in os.path.basename(f).lower()
        and 'mapping' not in os.path.basename(f).lower()
        and 'master' not in os.path.basename(f).lower()
    ]

    if mapping_files:
        map_df = pd.read_csv(mapping_files[0])
        if 'team_name' in map_df.columns and 'player_id' in map_df.columns:
            pb_mask = map_df['team_name'].astype(str).str.contains(r'\bpakistan\b|\bbangladesh\b', case=False, na=False)
            pak_ban_ids.update(map_df[pb_mask]['player_id'].dropna().unique())

    if teams_files:
        t_df = pd.read_csv(teams_files[0])
        if 'team_name' in t_df.columns and 'team_id' in t_df.columns:
            pb_team_ids = t_df[
                t_df['team_name'].astype(str).str.contains(r'\bpakistan\b|\bbangladesh\b', case=False, na=False)
            ]['team_id'].unique()
            if 'batting_team_id' in df.columns:
                pak_ban_ids.update(df[df['batting_team_id'].isin(pb_team_ids)]['striker_id'].dropna().unique())
            if 'bowling_team_id' in df.columns:
                pak_ban_ids.update(df[df['bowling_team_id'].isin(pb_team_ids)]['bowler_id'].dropna().unique())

    # Fallback: use result + winner_id directly from master file
    if len(pak_ban_ids) == 0 and 'result' in df.columns and 'winner_id' in df.columns:
        win_mask = df['result'].astype(str).str.contains(r'(pakistan|bangladesh).*?\bwon\b', case=False, na=False)
        pb_tids = {tid for tid in df[win_mask]['winner_id'].dropna().unique() if str(tid) != '0'}
        if pb_tids:
            if 'batting_team_id' in df.columns:
                pak_ban_ids.update(df[df['batting_team_id'].isin(pb_tids)]['striker_id'].dropna().unique())
            if 'bowling_team_id' in df.columns:
                pak_ban_ids.update(df[df['bowling_team_id'].isin(pb_tids)]['bowler_id'].dropna().unique())

    # Standardize columns
    over_col = 'overid' if 'overid' in df.columns else ('over_id' if 'over_id' in df.columns else 'over')
    ball_col = 'over_ball' if 'over_ball' in df.columns else ('ball_id' if 'ball_id' in df.columns else 'ball')
    df['over_num'] = pd.to_numeric(df[over_col], errors='coerce').fillna(1).astype(int)

    if 'runs' in df.columns:
        df['runs_batter'] = pd.to_numeric(df['runs'], errors='coerce').fillna(0)
    elif 'runs_off_bat' in df.columns:
        df['runs_batter'] = pd.to_numeric(df['runs_off_bat'], errors='coerce').fillna(0)
    else:
        df['runs_batter'] = 0

    df['extras'] = pd.to_numeric(df['extras'], errors='coerce').fillna(0) if 'extras' in df.columns else 0
    df['total_runs_ball'] = df['runs_batter'] + df['extras']

    df['is_wicket'] = df['is_wicket'].astype(str).str.upper().isin(['TRUE', '1'])
    df['is_four'] = df['is_four'].astype(str).str.upper().isin(['TRUE', '1'])
    df['is_six'] = df['is_six'].astype(str).str.upper().isin(['TRUE', '1'])
    df['is_dot'] = (df['total_runs_ball'] == 0) & (~df['is_wicket'])

    sort_cols = ['match_id', 'innings', over_col]
    if ball_col in df.columns:
        sort_cols.append(ball_col)
    df.sort_values(by=sort_cols, inplace=True)

    # Reconstruct Match State
    df['cum_runs'] = df.groupby(['match_id', 'innings'])['total_runs_ball'].cumsum()
    df['cum_wickets'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    df['balls_bowled_inn'] = df.groupby(['match_id', 'innings']).cumcount() + 1
    df['balls_remaining'] = (120 - df['balls_bowled_inn']).clip(lower=0)
    df['wickets_in_hand'] = (10 - df['cum_wickets']).clip(lower=0)

    first_inn_totals = df[df['innings'] == 1].groupby('match_id')['total_runs_ball'].sum().to_dict()
    df['target'] = df['match_id'].map(first_inn_totals).fillna(160) + 1
    df['runs_needed'] = df['target'] - df['cum_runs']
    df['rrr'] = np.where(
        (df['innings'] == 2) & (df['balls_remaining'] > 0),
        (df['runs_needed'] / df['balls_remaining']) * 6,
        0.0
    )

    return df, pak_ban_ids


def compute_metrics(df_filtered, pak_ban_ids, exclude_pak_ban, min_balls, rrr_threshold, apply_reliability):
    cond_chase = (
        (df_filtered['innings'] == 2) &
        (df_filtered['runs_needed'] > 0) &
        (df_filtered['balls_remaining'] <= 60) &
        (df_filtered['rrr'] <= 24.0) &
        ((df_filtered['rrr'] >= rrr_threshold) | ((df_filtered['wickets_in_hand'] <= 4) & (df_filtered['rrr'] >= 6.0)))
    )
    cond_death = (df_filtered['innings'] == 1) & (df_filtered['over_num'] >= 16) & (df_filtered['over_num'] <= 20)
    cond_crisis = (df_filtered['over_num'] <= 6) & (df_filtered['cum_wickets'] >= 3)

    stage_series = df_filtered['stage'].astype(str) if 'stage' in df_filtered.columns else pd.Series('', index=df_filtered.index)
    cond_playoff = stage_series.str.contains(r'qualifier|eliminator|final|semi', case=False, na=False)
    cond_super_over = df_filtered['innings'] > 2

    df_work = df_filtered.copy()
    df_work['is_pressure'] = cond_chase | cond_death | cond_crisis | cond_playoff | cond_super_over

    trigger_counts = {
        "2nd Innings High-Stakes Chase": int(cond_chase.sum()),
        "1st Innings Death Overs (16-20)": int(cond_death.sum()),
        "Powerplay Collapse (3+ Wkts in 1-6)": int(cond_crisis.sum()),
        "Playoffs / Knockouts / Super Overs": int((cond_playoff | cond_super_over).sum()),
        "Normal / Baseline Deliveries": int((~df_work['is_pressure']).sum())
    }

    # --- BATTING ---
    bat_all = df_work.groupby(['striker_id', 'striker_name']).agg(
        total_balls=('runs_batter', 'count'),
        total_runs=('runs_batter', 'sum'),
        matches=('match_id', 'nunique'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        dismissals=('is_wicket', 'sum')
    ).reset_index()
    bat_all['base_sr'] = (bat_all['total_runs'] / bat_all['total_balls']) * 100
    bat_all['average'] = bat_all['total_runs'] / np.maximum(bat_all['dismissals'], 1)

    bat_press = df_work[df_work['is_pressure']].groupby(['striker_id', 'striker_name']).agg(
        press_balls=('runs_batter', 'count'),
        press_runs=('runs_batter', 'sum'),
        press_fours=('is_four', 'sum'),
        press_sixes=('is_six', 'sum'),
        press_dots=('is_dot', 'sum')
    ).reset_index()

    bat_stats = bat_all.merge(bat_press, on=['striker_id', 'striker_name'], how='inner')
    bat_stats['press_sr'] = (bat_stats['press_runs'] / bat_stats['press_balls']) * 100
    bat_stats['delta_sr'] = bat_stats['press_sr'] - bat_stats['base_sr']
    bat_stats['press_boundary_pct'] = ((bat_stats['press_fours'] + bat_stats['press_sixes']) / bat_stats['press_balls']) * 100
    bat_stats['press_dot_pct'] = (bat_stats['press_dots'] / bat_stats['press_balls']) * 100

    if exclude_pak_ban and len(pak_ban_ids) > 0:
        bat_stats = bat_stats[~bat_stats['striker_id'].isin(pak_ban_ids)]
        bat_all_trad = bat_all[~bat_all['striker_id'].isin(pak_ban_ids)]
    else:
        bat_all_trad = bat_all

    elig_bat = bat_stats[bat_stats['press_balls'] >= min_balls].copy()
    raw_bat_score = (
        (elig_bat['press_sr'] * 0.45) +
        (elig_bat['delta_sr'] * 0.25) +
        (elig_bat['press_boundary_pct'] * 0.30) -
        (elig_bat['press_dot_pct'] * 0.20)
    )
    if apply_reliability:
        conf_bat = elig_bat['press_balls'] / (elig_bat['press_balls'] + 60.0)
        mean_bat = raw_bat_score.mean() if len(raw_bat_score) > 0 else 0
        elig_bat['pressure_score'] = (conf_bat * raw_bat_score) + ((1 - conf_bat) * mean_bat)
    else:
        elig_bat['pressure_score'] = raw_bat_score

    elig_bat = elig_bat.sort_values('pressure_score', ascending=False).reset_index(drop=True)
    elig_bat.index += 1

    # --- BOWLING ---
    bowl_all = df_work.groupby(['bowler_id', 'bowler_name']).agg(
        total_balls=('total_runs_ball', 'count'),
        total_runs_conceded=('total_runs_ball', 'sum'),
        matches=('match_id', 'nunique'),
        total_wickets=('is_wicket', 'sum')
    ).reset_index()
    bowl_all['base_econ'] = (bowl_all['total_runs_conceded'] / bowl_all['total_balls']) * 6
    bowl_all['overs'] = bowl_all['total_balls'] // 6 + (bowl_all['total_balls'] % 6) / 10
    bowl_all['average'] = bowl_all['total_runs_conceded'] / np.maximum(bowl_all['total_wickets'], 1)
    bowl_all['strike_rate'] = bowl_all['total_balls'] / np.maximum(bowl_all['total_wickets'], 1)

    bowl_press = df_work[df_work['is_pressure']].groupby(['bowler_id', 'bowler_name']).agg(
        press_balls=('total_runs_ball', 'count'),
        press_runs_conceded=('total_runs_ball', 'sum'),
        press_wickets=('is_wicket', 'sum'),
        press_dots=('is_dot', 'sum')
    ).reset_index()

    bowl_stats = bowl_all.merge(bowl_press, on=['bowler_id', 'bowler_name'], how='inner')
    bowl_stats['press_econ'] = (bowl_stats['press_runs_conceded'] / bowl_stats['press_balls']) * 6
    bowl_stats['delta_econ'] = bowl_stats['base_econ'] - bowl_stats['press_econ']
    bowl_stats['press_dot_pct'] = (bowl_stats['press_dots'] / bowl_stats['press_balls']) * 100

    if exclude_pak_ban and len(pak_ban_ids) > 0:
        bowl_stats = bowl_stats[~bowl_stats['bowler_id'].isin(pak_ban_ids)]
        bowl_all_trad = bowl_all[~bowl_all['bowler_id'].isin(pak_ban_ids)]
    else:
        bowl_all_trad = bowl_all

    elig_bowl = bowl_stats[bowl_stats['press_balls'] >= min_balls].copy()
    raw_bowl_score = (
        ((12 - elig_bowl['press_econ']) * 6.0) +
        (elig_bowl['delta_econ'] * 4.0) +
        (elig_bowl['press_dot_pct'] * 0.5) +
        ((elig_bowl['press_wickets'] / elig_bowl['press_balls']) * 200)
    )
    if apply_reliability:
        conf_bowl = elig_bowl['press_balls'] / (elig_bowl['press_balls'] + 60.0)
        mean_bowl = raw_bowl_score.mean() if len(raw_bowl_score) > 0 else 0
        elig_bowl['pressure_score'] = (conf_bowl * raw_bowl_score) + ((1 - conf_bowl) * mean_bowl)
    else:
        elig_bowl['pressure_score'] = raw_bowl_score

    elig_bowl = elig_bowl.sort_values('pressure_score', ascending=False).reset_index(drop=True)
    elig_bowl.index += 1

    most_runs = bat_all_trad.sort_values('total_runs', ascending=False).reset_index(drop=True)
    most_runs.index += 1

    most_wickets = bowl_all_trad.sort_values('total_wickets', ascending=False).reset_index(drop=True)
    most_wickets.index += 1

    return df_work, elig_bat, elig_bowl, most_runs, most_wickets, trigger_counts


# ------------------------------------------------------------------------------
# HEADER & FILTERS
# ------------------------------------------------------------------------------
st.markdown("# 🏏 SupeRR Selector — Clutch Pressure Analytics")
st.caption("Evaluating player reliability in high-pressure T20 match situations compared to their personal baseline.")

df, pak_ban_ids = load_base_data()
if df is None:
    st.error("Could not find `master_cricket_data.csv` or `master_cricket_data.csv.gz`.")
    st.stop()

# Top Filter Row
f_col1, f_col2 = st.columns([2, 1])
available_years = sorted([int(y) for y in df['season_year'].dropna().unique()])
default_years = [y for y in available_years if y >= 2024] or available_years[-2:]

with f_col1:
    selected_years = st.multiselect(
        "📅 Select Years / Seasons",
        options=available_years,
        default=default_years
    )
with f_col2:
    comp_filter = st.selectbox(
        "🏆 Competition Filter",
        ["All (IPL + T20I)", "IPL Only", "T20I Only"]
    )

if not selected_years:
    st.warning("Please select at least one year above.")
    st.stop()

# Sidebar Controls
st.sidebar.header("⚙️ Model Tuning")
exclude_pb = st.sidebar.checkbox(
    f"Exclude Pak & Ban Players ({len(pak_ban_ids)} found)",
    value=True,
    help="Removes Pak/Ban players from final Top-20 rankings while keeping all balls played against them."
)
apply_reliability = st.sidebar.checkbox(
    "Apply Sample-Size Reliability Weighting",
    value=True,
    help="Prevents small-sample flukes (e.g. 55 balls) from beating consistent 200+ ball performers."
)
min_balls = st.sidebar.slider("Min Pressure Balls Cutoff", 20, 200, 60, step=10)
rrr_threshold = st.sidebar.slider("Chase RRR Pressure Threshold", 7.5, 12.0, 9.0, step=0.5)
top_n = st.sidebar.selectbox("Show Top N Players", [10, 15, 20], index=2)

# Apply Top Filters
df_filtered = df[df['season_year'].isin(selected_years)]
if comp_filter == "IPL Only":
    df_filtered = df_filtered[df_filtered['comp_type'] == 'IPL']
elif comp_filter == "T20I Only":
    df_filtered = df_filtered[df_filtered['comp_type'] == 'T20I']

df_work, elig_bat, elig_bowl, most_runs, most_wickets, trigger_counts = compute_metrics(
    df_filtered, pak_ban_ids, exclude_pb, min_balls, rrr_threshold, apply_reliability
)

# Custom Full-Width KPI Cards
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Deliveries Analyzed</div>
        <div class="kpi-value">{len(df_work):,}</div>
        <div class="kpi-sub">Seasons: {', '.join(map(str, sorted(selected_years)))}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    pct_p = df_work['is_pressure'].mean() * 100 if len(df_work) > 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">High-Pressure Balls</div>
        <div class="kpi-value">{df_work['is_pressure'].sum():,}</div>
        <div class="kpi-sub">{pct_p:.1f}% of total balls</div>
    </div>""", unsafe_allow_html=True)

with k3:
    top_b_name = elig_bat.iloc[0]['striker_name'] if len(elig_bat) > 0 else "N/A"
    top_b_sub = f"SR: {elig_bat.iloc[0]['press_sr']:.1f} ({elig_bat.iloc[0]['delta_sr']:+.1f} Δ)" if len(elig_bat) > 0 else "-"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">#1 Clutch Batsman</div>
        <div class="kpi-value">{top_b_name}</div>
        <div class="kpi-sub">{top_b_sub}</div>
    </div>""", unsafe_allow_html=True)

with k4:
    top_w_name = elig_bowl.iloc[0]['bowler_name'] if len(elig_bowl) > 0 else "N/A"
    top_w_sub = f"Econ: {elig_bowl.iloc[0]['press_econ']:.2f} ({elig_bowl.iloc[0]['delta_econ']:+.2f} tighter)" if len(elig_bowl) > 0 else "-"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">#1 Clutch Bowler</div>
        <div class="kpi-value">{top_w_name}</div>
        <div class="kpi-sub">{top_w_sub}</div>
    </div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# THREE DEDICATED TABS: BATSMEN, BOWLERS, METHODOLOGY
# ------------------------------------------------------------------------------
tab_bat, tab_bowl, tab_method = st.tabs([
    "🏏 Batsmen",
    "🎯 Bowlers",
    "📐 Methodology"
])

# ==============================================================================
# TAB 1: BATSMEN
# ==============================================================================
with tab_bat:
    st.subheader(f"🔥 Top {top_n} Batsmen Under Pressure")
    bat_view = elig_bat.head(top_n)[[
        'striker_name', 'press_balls', 'press_runs', 'press_sr', 'base_sr',
        'delta_sr', 'press_boundary_pct', 'press_dot_pct', 'pressure_score'
    ]].copy()
    bat_view.columns = [
        'Batsman', 'Press. Balls', 'Press. Runs', 'Press. SR', 'Base SR',
        'SR Lift (Δ)', 'Boundary %', 'Dot %', 'Clutch Score'
    ]
    st.dataframe(
        bat_view,
        use_container_width=True,
        column_config={
            "Press. SR": st.column_config.NumberColumn(format="%.1f"),
            "Base SR": st.column_config.NumberColumn(format="%.1f"),
            "SR Lift (Δ)": st.column_config.NumberColumn(format="%+.1f"),
            "Boundary %": st.column_config.NumberColumn(format="%.1f%%"),
            "Dot %": st.column_config.NumberColumn(format="%.1f%%"),
            "Clutch Score": st.column_config.ProgressColumn(
                format="%.1f",
                min_value=float(bat_view["Clutch Score"].min() * 0.8) if len(bat_view) else 0,
                max_value=float(bat_view["Clutch Score"].max() * 1.05) if len(bat_view) else 100
            )
        }
    )

    if len(elig_bat) > 0:
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            fig_bat = go.Figure()
            fig_bat.add_trace(go.Bar(name='Baseline SR', x=bat_view['Batsman'], y=bat_view['Base SR'], marker_color='#475569'))
            fig_bat.add_trace(go.Bar(name='Pressure SR', x=bat_view['Batsman'], y=bat_view['Press. SR'], marker_color='#ec4899'))
            fig_bat.update_layout(
                barmode='group', title="Baseline SR vs. Pressure SR",
                xaxis_tickangle=-35, height=400, margin=dict(t=40, b=80)
            )
            st.plotly_chart(fig_bat, use_container_width=True)

        with c_chart2:
            top_scatter = elig_bat.head(40)
            fig_scat_b = px.scatter(
                top_scatter, x='base_sr', y='press_sr', size='press_balls',
                hover_name='striker_name', color='delta_sr',
                color_continuous_scale='RdYlGn',
                labels={'base_sr': 'Baseline SR', 'press_sr': 'Pressure SR', 'delta_sr': 'SR Lift'},
                title="Clutch Quadrant (Above Line = Elevates Under Pressure)"
            )
            min_v = min(top_scatter['base_sr'].min(), top_scatter['press_sr'].min()) - 5
            max_v = max(top_scatter['base_sr'].max(), top_scatter['press_sr'].max()) + 5
            fig_scat_b.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="white", dash="dash", width=1))
            fig_scat_b.update_layout(height=400, margin=dict(t=40, b=40))
            st.plotly_chart(fig_scat_b, use_container_width=True)

    st.divider()
    col_b1, col_b2 = st.columns([1.4, 1])
    with col_b1:
        st.subheader("📊 Most Runs (Overall Stats)")
        runs_view = most_runs.head(15)[['striker_name', 'matches', 'total_runs', 'total_balls', 'base_sr', 'average', 'fours', 'sixes']].copy()
        runs_view.columns = ['Batsman', 'Matches', 'Runs', 'Balls', 'SR', 'Avg', '4s', '6s']
        st.dataframe(runs_view.round(2), use_container_width=True)
    with col_b2:
        st.subheader("🔍 Batsman Deep-Dive")
        if len(elig_bat) > 0:
            player_b = st.selectbox("Select Qualified Batsman", elig_bat['striker_name'].tolist())
            row_b = elig_bat[elig_bat['striker_name'] == player_b].iloc[0]
            r1, r2 = st.columns(2)
            r1.metric("Pressure SR", f"{row_b['press_sr']:.1f}", f"{row_b['delta_sr']:+.1f} vs Base")
            r2.metric("Clutch Score", f"{row_b['pressure_score']:.2f}")
            r3, r4 = st.columns(2)
            r3.metric("Boundary %", f"{row_b['press_boundary_pct']:.1f}%")
            r4.metric("Dot Ball %", f"{row_b['press_dot_pct']:.1f}%")

# ==============================================================================
# TAB 2: BOWLERS
# ==============================================================================
with tab_bowl:
    st.subheader(f"🎯 Top {top_n} Bowlers Under Pressure")
    bowl_view = elig_bowl.head(top_n)[[
        'bowler_name', 'press_balls', 'press_econ', 'base_econ',
        'delta_econ', 'press_dot_pct', 'press_wickets', 'pressure_score'
    ]].copy()
    bowl_view.columns = [
        'Bowler', 'Press. Balls', 'Press. Econ', 'Base Econ',
        'Econ Saved (Δ)', 'Dot %', 'Wickets', 'Clutch Score'
    ]
    st.dataframe(
        bowl_view,
        use_container_width=True,
        column_config={
            "Press. Econ": st.column_config.NumberColumn(format="%.2f"),
            "Base Econ": st.column_config.NumberColumn(format="%.2f"),
            "Econ Saved (Δ)": st.column_config.NumberColumn(format="%+.2f"),
            "Dot %": st.column_config.NumberColumn(format="%.1f%%"),
            "Clutch Score": st.column_config.ProgressColumn(
                format="%.1f",
                min_value=float(bowl_view["Clutch Score"].min() * 0.8) if len(bowl_view) else 0,
                max_value=float(bowl_view["Clutch Score"].max() * 1.05) if len(bowl_view) else 100
            )
        }
    )

    if len(elig_bowl) > 0:
        w_chart1, w_chart2 = st.columns(2)
        with w_chart1:
            fig_bowl = go.Figure()
            fig_bowl.add_trace(go.Bar(name='Baseline Econ', x=bowl_view['Bowler'], y=bowl_view['Base Econ'], marker_color='#475569'))
            fig_bowl.add_trace(go.Bar(name='Pressure Econ', x=bowl_view['Bowler'], y=bowl_view['Press. Econ'], marker_color='#10b981'))
            fig_bowl.update_layout(
                barmode='group', title="Baseline Economy vs. Pressure Economy (Lower is Better)",
                xaxis_tickangle=-35, height=400, margin=dict(t=40, b=80)
            )
            st.plotly_chart(fig_bowl, use_container_width=True)

        with w_chart2:
            top_scat_w = elig_bowl.head(40)
            fig_scat_w = px.scatter(
                top_scat_w, x='base_econ', y='press_econ', size='press_balls',
                hover_name='bowler_name', color='delta_econ',
                color_continuous_scale='RdYlGn',
                labels={'base_econ': 'Baseline Economy', 'press_econ': 'Pressure Economy', 'delta_econ': 'Econ Tightened'},
                title="Clutch Quadrant (Below Line = Tighter Under Pressure)"
            )
            min_e = min(top_scat_w['base_econ'].min(), top_scat_w['press_econ'].min()) - 0.5
            max_e = max(top_scat_w['base_econ'].max(), top_scat_w['press_econ'].max()) + 0.5
            fig_scat_w.add_shape(type="line", x0=min_e, y0=min_e, x1=max_e, y1=max_e, line=dict(color="white", dash="dash", width=1))
            fig_scat_w.update_layout(height=400, margin=dict(t=40, b=40))
            st.plotly_chart(fig_scat_w, use_container_width=True)

    st.divider()
    col_w1, col_w2 = st.columns([1.4, 1])
    with col_w1:
        st.subheader("📊 Most Wickets (Overall Stats)")
        wkts_view = most_wickets.head(15)[['bowler_name', 'matches', 'overs', 'total_wickets', 'base_econ', 'average', 'strike_rate']].copy()
        wkts_view.columns = ['Bowler', 'Matches', 'Overs', 'Wickets', 'Econ', 'Avg', 'SR']
        st.dataframe(wkts_view.round(2), use_container_width=True)
    with col_w2:
        st.subheader("🔍 Bowler Deep-Dive")
        if len(elig_bowl) > 0:
            player_w = st.selectbox("Select Qualified Bowler", elig_bowl['bowler_name'].tolist())
            row_w = elig_bowl[elig_bowl['bowler_name'] == player_w].iloc[0]
            w1, w2 = st.columns(2)
            w1.metric("Pressure Economy", f"{row_w['press_econ']:.2f}", f"{row_w['delta_econ']:+.2f} tighter")
            w2.metric("Clutch Score", f"{row_w['pressure_score']:.2f}")
            w3, w4 = st.columns(2)
            w3.metric("Dot Ball %", f"{row_w['press_dot_pct']:.1f}%")
            w4.metric("Pressure Wickets", f"{int(row_w['press_wickets'])}")

# ==============================================================================
# TAB 3: METHODOLOGY
# ==============================================================================
with tab_method:
    st.subheader("📐 How Pressure & Clutch Reliability Are Calculated")

    m_col1, m_col2 = st.columns([1.2, 1])
    with m_col1:
        st.markdown("### 1. Defining a High-Pressure Delivery")
        st.markdown(f"""
        <div class="method-box">
            <b>Condition 1 — 2nd Innings High-Stakes Chase:</b><br>
            Overs 11–20 (≤ 60 balls remaining), target still active, competitive game (RRR ≤ 24.0), and either <b>RRR ≥ {rrr_threshold}</b> OR <b>≤ 4 wickets in hand</b> with RRR ≥ 6.0.
        </div>
        <div class="method-box">
            <b>Condition 2 — 1st Innings Death Overs:</b><br>
            Overs 16–20 when batting sides must maximize totals and bowlers execute yorkers/variations under extreme boundary threat.
        </div>
        <div class="method-box">
            <b>Condition 3 — Powerplay Crisis Recovery:</b><br>
            Overs 1–6 in either innings when <b>3 or more wickets</b> have already fallen, testing crisis composure.
        </div>
        <div class="method-box">
            <b>Condition 4 — Playoffs, Knockouts & Super Overs:</b><br>
            All deliveries in Qualifiers, Eliminators, Semi-Finals, Finals, or Super Overs where stakes are absolute.
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("### Live Delivery Breakdown")
        trig_df = pd.DataFrame({
            "Match Context": list(trigger_counts.keys()),
            "Deliveries": list(trigger_counts.values())
        })
        fig_trig = px.bar(
            trig_df, x="Deliveries", y="Match Context", orientation='h',
            color="Match Context", text_auto='.2s',
            color_discrete_sequence=['#ec4899', '#f43f5e', '#f59e0b', '#8b5cf6', '#475569']
        )
        fig_trig.update_layout(showlegend=False, height=380, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_trig, use_container_width=True)

    st.divider()
    st.markdown("### 2. Performance Measurement & Baseline Comparison")
    f_c1, f_c2 = st.columns(2)

    with f_c1:
        st.markdown("#### 🏏 Batting Clutch Formula")
        st.latex(r"\text{Raw Score} = 0.45(\text{SR}_{\text{press}}) + 0.25(\Delta \text{SR}) + 0.30(\text{Boundary\%}) - 0.20(\text{Dot\%})")
        st.markdown("""
        - **$\Delta \text{SR} = \text{SR}_{\text{press}} - \text{SR}_{\text{base}}$**: Rewards players who accelerate under pressure compared to their own baseline.
        - **Boundary %**: Measures six- and four-hitting ability when field pressure peaks.
        - **Dot % Penalty**: Penalizes stalling momentum during chases or death overs.
        """)

    with f_c2:
        st.markdown("#### 🎯 Bowling Clutch Formula")
        st.latex(r"\text{Raw Score} = 6(12 - \text{Econ}_{\text{press}}) + 4(\Delta \text{Econ}) + 0.5(\text{Dot\%}) + 200\left(\frac{\text{Wickets}}{\text{Balls}}\right)")
        st.markdown("""
        - **$\Delta \text{Econ} = \text{Econ}_{\text{base}} - \text{Econ}_{\text{press}}$**: Positive values indicate bowlers who tighten up and concede *fewer* runs under pressure.
        - **Dot %**: Measures run-choking control in crunch overs.
        - **Wicket Threat**: Rewards game-breaking breakthroughs per ball bowled.
        """)

    st.divider()
    st.markdown("### 3. Fair Comparisons & Reliability Safeguards")
    st.markdown(f"""
    - **Minimum Qualification Cutoff (`{min_balls}` balls)**: Prevents small-sample outliers (e.g., 10-ball cameos) from distorting rankings.
    - **Bayesian Volume Confidence Weighting (`{'Enabled' if apply_reliability else 'Disabled'}`)**:
      $$\\text{{Confidence}} = \\frac{{\\text{{Pressure Balls}}}}{{\\text{{Pressure Balls}} + 60}}$$
      Blends a player's raw score slightly toward the cohort average when sample sizes are smaller, ensuring proven high-volume performers rank fairly.
    - **Opponent Preservation ("vs Them" Stats)**: Pakistan & Bangladesh players are excluded from candidate rankings (`{'Enabled' if exclude_pb else 'Disabled'}`), while all deliveries bowled **to or against** them remain counted in full for eligible players.
    """)
