import os
import re
import glob
from datetime import datetime
from PIL import Image

import pandas as pd
import numpy as np
import streamlit as st

# --- Device name mapping ---
# Mapping from SM codes to friendly names
DEVICE_LABELS = {
    'SM01': "Outdoor Reference",
    'SM02': "Altar-Main",
    'SM03': "Chapel-Main",
    'SM04': "Sanctuary North-Crawlspace",
    'SM05': "Sanctuary South-Crawlspace",
}

# --- Helper functions ---
def load_and_clean_file(path):
    """Load a device export file (.csv or .xlsx) and clean column names."""
    fn = os.path.basename(path)
    match = re.match(r"(SM\d+)_export_.*\.(csv|xlsx)", fn, re.IGNORECASE)
    device = match.group(1) if match else "Unknown"

    if fn.lower().endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={
        df.columns[0]: 'Timestamp',
        df.columns[1]: 'Temp_F',
        df.columns[2]: 'RH'
    })
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Device'] = device
    return df


def find_contiguous_nans(mask):
    gaps, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            gaps.append((start, i-1)); start = None
    if start is not None:
        gaps.append((start, len(mask)-1))
    return gaps


def fill_and_flag(series, max_gap=10, n_neighbors=4):
    orig = series.copy()
    s = series.copy()
    mask = s.isna().values
    idxs = s.index
    for start, end in find_contiguous_nans(mask):
        if (end - start + 1) <= max_gap:
            left = max(start - n_neighbors, 0)
            right = min(end + n_neighbors, len(idxs)-1)
            segment = s.iloc[left:right+1]
            s.iloc[left:right+1] = segment.interpolate()
    interpolated = orig.isna() & s.notna()
    return s, interpolated



# --- Streamlit App ---
st.set_page_config(page_title='St Matthias: 2025 Environmental Data', layout='wide')
# Display logo above the title using a path relative to this script so it
# works regardless of the working directory from which Streamlit is run.
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "Logo.png")
if os.path.exists(logo_path):
    logoImage = Image.open(logo_path)
    st.image(logoImage)
else:
    st.warning(f"Logo not found at {logo_path}")

st.header('St Matthias: 2025 Environmental Data')

# Settings header
st.sidebar.title('Settings')
#st.sidebar.image("./Logo.png", use_container_width=True)

# Constants
FOLDER = './data'

# Date selectors
date_cols = st.sidebar.columns(2)
date_cols[0].write('Start date')
start_date = date_cols[0].date_input("Start Date", value=datetime(2025, 1, 1), label_visibility="collapsed")
date_cols[1].write('End date')
end_date   = date_cols[1].date_input("End Date", value=datetime.today(), label_visibility="collapsed")

# Load Data button
if st.sidebar.button('Load Data'):
    pattern_csv = os.path.join(FOLDER, 'SM*_export_*.csv')
    pattern_xlsx = os.path.join(FOLDER, 'SM*_export_*.xlsx')
    files = glob.glob(pattern_csv) + glob.glob(pattern_xlsx)
    device_dfs = {load_and_clean_file(f)['Device'].iloc[0]: load_and_clean_file(f) for f in files}

    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_times = device_dfs[master].sort_values('Timestamp')['Timestamp']

    records = []
    for dev, df in device_dfs.items():
        tmp = df.set_index('Timestamp').reindex(
            master_times, method='nearest', tolerance=pd.Timedelta(minutes=30)
        )
        filled_t, flag_t = fill_and_flag(tmp['Temp_F'])
        filled_r, flag_r = fill_and_flag(tmp['RH'])
        tmp['Temp_F']      = filled_t
        tmp['RH']          = filled_r
        tmp['Interpolated']= flag_t
        tmp['Device']      = dev
        # Map to friendly name
        tmp['DeviceName']  = DEVICE_LABELS.get(dev, dev)
        records.append(tmp.reset_index().rename(columns={'index':'Timestamp'}))

    df_all = pd.concat(records, ignore_index=True)
    st.session_state.df_all    = df_all
    st.session_state.devices   = sorted(df_all['Device'])

# Device groupings
devices    = st.session_state.get('devices', [])
main       = [f'SM{i:02d}' for i in range(2,4)]
crawlspace = [f'SM{i:02d}' for i in range(4,6)]
outdoor    = ['SM01']

# Grouped checkboxes
def group_ui(group, label):
    st.sidebar.markdown(f'**{label}**')
    col1, col2 = st.sidebar.columns(2)
    if col1.button(f'Select All {label}'):
        for d in group:
            if d in devices: st.session_state[f'chk_{d}'] = True
    if col2.button(f'Deselect All {label}'):
        for d in group:
            if d in devices: st.session_state[f'chk_{d}'] = False
    for d in group:
        if d in devices:
            key = f'chk_{d}'
            st.session_state.setdefault(key, True)
            label = DEVICE_LABELS.get(d, d)
            st.sidebar.checkbox(label, key=key)

group_ui(main,       'Main')
group_ui(crawlspace, 'Crawlspace')
# Outdoor Reference (no select/deselect buttons)
st.sidebar.markdown("**Outdoor Reference**")
for d in outdoor:
    if d in devices:
        key = f"chk_{d}"
        st.session_state.setdefault(key, True)
        st.sidebar.checkbox(DEVICE_LABELS.get(d, d), key=key)

selected = [d for d in devices if st.session_state.get(f'chk_{d}')]

# --- KPI Target Descriptions ---
with st.sidebar.expander('KPI Band Descriptions', expanded=False):
    desc_avg_temp = st.text_input('Avg Temp Target',
                                  'Quarterly average between 68°F and 75°F')
    desc_temp_swing = st.text_input('Temp Swing Target',
                                    'Difference between max and min under 5°F')
    desc_avg_rh = st.text_input('Avg RH Target',
                                'Comfort range 30–60% RH')
    desc_rh_var = st.text_input('RH Variability Target',
                                'Standard deviation below 10%')
    desc_corr = st.text_input('Correlation Target',
                              'Pearson correlation with outdoor > 0.8')

# Compile & Display
if st.sidebar.button('Compile'):
    if 'df_all' not in st.session_state:
        st.error('Please load data first.')
    else:
        df = st.session_state.df_all
        df = df[df['Device'].isin(selected)]
        df = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

        indoor_df = df[df['Device'] != 'SM01']
        avg_temp = indoor_df['Temp_F'].mean()
        temp_swing = indoor_df['Temp_F'].max() - indoor_df['Temp_F'].min()
        avg_rh = indoor_df['RH'].mean()
        rh_var = indoor_df['RH'].std()

        # Correlation of each indoor sensor vs outdoor reference
        outdoor_df = df[df['Device'] == 'SM01'][['Timestamp', 'Temp_F']].rename(columns={'Temp_F': 'Temp_out'})
        corr_results = {}
        for dev in indoor_df['Device'].unique():
            dev_df = indoor_df[indoor_df['Device'] == dev][['Timestamp', 'Temp_F']]
            merged = dev_df.merge(outdoor_df, on='Timestamp')
            corr = merged['Temp_F'].corr(merged['Temp_out']) if not merged.empty else np.nan
            corr_results[dev] = corr

        # Evaluation against bands
        def temp_status(val):
            if 68 <= val <= 75:
                return 'Pass'
            if val < 68 - 2 or val > 75 + 2:
                return 'Flag'
            return 'Check'

        swing_status = 'Area for improvement' if temp_swing > 5 else 'Pass'
        rh_status = 'Area for improvement' if rh_var > 10 else 'Pass'

        rows = [
            {
                'KPI': 'Average Temp (°F)',
                'Value': f"{avg_temp:.2f}",
                'Status': temp_status(avg_temp),
                'Target': desc_avg_temp,
            },
            {
                'KPI': 'Temp Swing (°F)',
                'Value': f"{temp_swing:.2f}",
                'Status': swing_status,
                'Target': desc_temp_swing,
            },
            {
                'KPI': 'Average RH (%)',
                'Value': f"{avg_rh:.2f}",
                'Status': 'n/a',
                'Target': desc_avg_rh,
            },
            {
                'KPI': 'RH Variability (%)',
                'Value': f"{rh_var:.2f}",
                'Status': rh_status,
                'Target': desc_rh_var,
            },
        ]

        for dev, corr in corr_results.items():
            rows.append({
                'KPI': f"Corr {DEVICE_LABELS.get(dev, dev)} vs Outdoor",
                'Value': f"{corr:.2f}" if pd.notna(corr) else 'n/a',
                'Status': 'Calibrate' if pd.notna(corr) and corr < 0.8 else 'OK',
                'Target': desc_corr,
            })

        st.header('Quarterly KPI Summary')
        st.table(pd.DataFrame(rows))

else:
    st.info("Use 'Load Data' then 'Compile' to display the KPI summary.")
