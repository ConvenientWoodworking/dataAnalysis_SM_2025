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
@@ -45,237 +44,162 @@ def find_contiguous_nans(mask):
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

# Constants
FOLDER = './data'

# Date selectors
date_cols = st.sidebar.columns(2)
date_cols[0].write('Start date')
start_date = date_cols[0].date_input(
    "Start Date", value=datetime(2025, 1, 1), label_visibility="collapsed"
)
date_cols[1].write('End date')
end_date = date_cols[1].date_input(
    "End Date", value=datetime.today(), label_visibility="collapsed"
)

# Device groupings
main = [f"SM{i:02d}" for i in range(2, 4)]
crawlspace = [f"SM{i:02d}" for i in range(4, 6)]
location_map = {d: "Main" for d in main}
location_map.update({d: "Crawlspace" for d in crawlspace})

# Hardcoded KPI band descriptions
desc_avg_temp = "Quarterly average between 68°F and 75°F"
desc_temp_swing = "Difference between max and min under 5°F"
desc_avg_rh = "Comfort range 30–60% RH"
desc_rh_var = "Standard deviation below 10%"

# Analyze & Display
if st.sidebar.button('Analyze'):
    pattern_csv = os.path.join(FOLDER, 'SM*_export_*.csv')
    pattern_xlsx = os.path.join(FOLDER, 'SM*_export_*.xlsx')
    files = glob.glob(pattern_csv) + glob.glob(pattern_xlsx)
    device_dfs = {
        load_and_clean_file(f)['Device'].iloc[0]: load_and_clean_file(f)
        for f in files
    }

    master = max(device_dfs, key=lambda d: len(device_dfs[d]))
    master_times = device_dfs[master].sort_values('Timestamp')['Timestamp']

    records = []
    for dev, df in device_dfs.items():
        tmp = df.set_index('Timestamp').reindex(
            master_times, method='nearest', tolerance=pd.Timedelta(minutes=30)
        )
        filled_t, flag_t = fill_and_flag(tmp['Temp_F'])
        filled_r, _ = fill_and_flag(tmp['RH'])
        tmp['Temp_F'] = filled_t
        tmp['RH'] = filled_r
        tmp['Interpolated'] = flag_t
        tmp['Device'] = dev
        tmp['DeviceName'] = DEVICE_LABELS.get(dev, dev)
        tmp['Location'] = location_map.get(dev, 'Outdoor')
        records.append(tmp.reset_index().rename(columns={'index': 'Timestamp'}))

    df_all = pd.concat(records, ignore_index=True)
    df_all = df_all[
        (df_all['Timestamp'].dt.date >= start_date)
        & (df_all['Timestamp'].dt.date <= end_date)
    ]

    indoor_df = df_all[df_all['Location'] != 'Outdoor']

    def temp_status(val: float) -> str:
        if 68 <= val <= 75:
            return 'Pass'
        if val < 68 - 2 or val > 75 + 2:
            return 'Flag'
        return 'Check'

    rows = []
    for location in sorted(indoor_df['Location'].unique()):
        loc_df = indoor_df[indoor_df['Location'] == location]
        avg_temp = loc_df['Temp_F'].mean()
        temp_swing = loc_df['Temp_F'].max() - loc_df['Temp_F'].min()
        avg_rh = loc_df['RH'].mean()
        rh_var = loc_df['RH'].std()

        swing_status = 'Area for improvement' if temp_swing > 5 else 'Pass'
        rh_status = 'Area for improvement' if rh_var > 10 else 'Pass'

        rows.extend(
            [
                {
                    'Location': location,
                    'KPI': 'Average Temp (°F)',
                    'Value': f"{avg_temp:.2f}",
                    'Status': temp_status(avg_temp),
                    'Target': desc_avg_temp,
                },
                {
                    'Location': location,
                    'KPI': 'Temp Swing (°F)',
                    'Value': f"{temp_swing:.2f}",
                    'Status': swing_status,
                    'Target': desc_temp_swing,
                },
                {
                    'Location': location,
                    'KPI': 'Average RH (%)',
                    'Value': f"{avg_rh:.2f}",
                    'Status': 'n/a',
                    'Target': desc_avg_rh,
                },
                {
                    'Location': location,
                    'KPI': 'RH Variability (%)',
                    'Value': f"{rh_var:.2f}",
                    'Status': rh_status,
                    'Target': desc_rh_var,
                },
            ]
        )

    st.header('Quarterly KPI Summary')
    st.table(pd.DataFrame(rows))

else:
    st.info('Set your date range and click Analyze to display the KPI summary.')
