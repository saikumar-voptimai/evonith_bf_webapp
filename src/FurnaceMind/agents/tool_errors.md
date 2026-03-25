# FurnaceMind Tool Errors & Learnings

This file is auto-updated when tool execution fails during AI Co-Operate sessions.
Each entry records the tool, parameters, error, and timestamp for debugging and improvement.

---


## 2026-02-28T06:50:46.452204+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "import pandas as pd\nimport plotly.express as px\n\ndf = pd.read_csv('current_furnace_data.csv')\n\n# Selecting the relevant columns for ETACO\nfig = px.line(df, x=df.index, y='Process Params - BF2_BODY_ETACO', title='ETACO Variation Over Last 8 Hours')\nfig.update_layout(xaxis_title='Time', yaxis_title='ETACO')"}`

**Error:**

```
Disallowed token in code: \bimport\b
```


## 2026-02-28T06:50:50.928949+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "import pandas as pd\nimport plotly.express as px\ndf = pd.read_csv('current_furnace_data.csv')\nfig = px.line(df, x=df.index, y='Process Params - BF2_BODY_ETACO', title='ETACO Variation Over Last 8 Hours')\nfig.update_layout(xaxis_title='Time', yaxis_title='ETACO')"}`

**Error:**

```
Disallowed token in code: \bimport\b
```


## 2026-03-24T11:47:57.823027+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "import pandas as pd\nimport plotly.express as px\ndf = pd.read_csv('current_furnace_data.csv')\n\n# Selecting the relevant columns for ETACO\nfig = px.line(df, x=df.index, y='Process Params - BF2_BODY_ETACO', title='ETACO Variation Over Last 8 Hours')\nfig.update_layout(xaxis_title='Time', yaxis_title='ETACO')\nfig.show()"}`

**Error:**

```
Disallowed token in code: \bimport\b
```


## 2026-03-24T11:52:50.274747+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "import pandas as pd\nimport plotly.express as px\ndf = pd.read_csv('current_furnace_data.csv')\n\n# Extract the most recent bunker report row\nrecent_bunker = df.iloc[0]\n\n# Extract silicon percentage from the most recent bunker report\n# The silicon percentage is available for multiple ore types; here we take the average of the major ore components\n# Major ore components: ore_1, ore_2, ore_3, ore_4, ore_5, ore_6, ore_7, ore_9, ore_10, ore_11\nmajor_ore_silicon_columns = [\n    'Offline[Bunker Report] - ore_1_pct_sio2',\n    'Offline[Bunker Report] - ore_2_pct_sio2',\n    'Offline[Bunker Report] - ore_3_pct_sio2',\n    'Offline[Bunker Report] - ore_4_pct_sio2',\n    'Offline[Bunker Report] - ore_5_pct_sio2',\n    'Offline[Bunker Report] - ore_6_pct_sio2',\n    'Offline[Bunker Report] - ore_7_pct_sio2',\n    'Offline[Bunker Report] - ore_9_pct_sio2',\n    'Offline[Bunker Report] - ore_10_pct_sio2',\n    'Offline[Bunker Report] - ore_11_pct_sio2'\n]\n\n# Calculate average silicon percentage from major ore components\nmajor_ore_silicon = recent_bunker[major_ore_silicon_columns].dropna().mean()\n\n# Extract hot metal silicon from the most recent DPR report\n# Hot metal silicon is not directly available in DPR; we need to look at HM_SLAG report\n# Let's fetch the latest HM_SLAG data\n"}`

**Error:**

```
Disallowed token in code: \bimport\b
```


## 2026-03-24T11:53:01.279248+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "df = pd.read_csv('current_furnace_data.csv')\n\n# Extract the most recent bunker report row\nrecent_bunker = df.iloc[0]\n\n# Extract silicon percentage from the most recent bunker report\n# The silicon percentage is available for multiple ore types; here we take the average of the major ore components\n# Major ore components: ore_1, ore_2, ore_3, ore_4, ore_5, ore_6, ore_7, ore_9, ore_10, ore_11\nmajor_ore_silicon_columns = [\n    'Offline[Bunker Report] - ore_1_pct_sio2',\n    'Offline[Bunker Report] - ore_2_pct_sio2',\n    'Offline[Bunker Report] - ore_3_pct_sio2',\n    'Offline[Bunker Report] - ore_4_pct_sio2',\n    'Offline[Bunker Report] - ore_5_pct_sio2',\n    'Offline[Bunker Report] - ore_6_pct_sio2',\n    'Offline[Bunker Report] - ore_7_pct_sio2',\n    'Offline[Bunker Report] - ore_9_pct_sio2',\n    'Offline[Bunker Report] - ore_10_pct_sio2',\n    'Offline[Bunker Report] - ore_11_pct_sio2'\n]\n\n# Calculate average silicon percentage from major ore components\nmajor_ore_silicon = recent_bunker[major_ore_silicon_columns].dropna().mean()\n\n# Extract hot metal silicon from the most recent DPR report\n# Hot metal silicon is not directly available in DPR; we need to look at HM_SLAG report\n# Let's fetch the latest HM_SLAG data\n"}`

**Error:**

```
"None of [Index(['Offline[Bunker Report] - ore_1_pct_sio2',\n       'Offline[Bunker Report] - ore_2_pct_sio2',\n       'Offline[Bunker Report] - ore_3_pct_sio2',\n       'Offline[Bunker Report] - ore_4_pct_sio2',\n       'Offline[Bunker Report] - ore_5_pct_sio2',\n       'Offline[Bunker Report] - ore_6_pct_sio2',\n       'Offline[Bunker Report] - ore_7_pct_sio2',\n       'Offline[Bunker Report] - ore_9_pct_sio2',\n       'Offline[Bunker Report] - ore_10_pct_sio2',\n       'Offline[Bunker Report] - ore_11_pct_sio2'],\n      dtype='object')] are in the [index]"
```


## 2026-03-25T11:47:51.487413+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "import pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\n\n# Load the offline HM_SLAG data\ndf = pd.read_csv('current_furnace_data.csv')\n\n# Filter for last 7 days and find March 23 11AM data\nlast_7_days = df.iloc[-7:]\nmar23_11am = last_7_days.iloc[-2]  # Assuming this is March 23 data\n\n# Create the plot\nfig = go.Figure()\n\n# Plot last 7 days Si% with marker for March 23 11AM\nfig.add_trace(go.Scatter(x=last_7_days.index, y=last_7_days['Offline[HM & Slag] - chem_pct_si'],\n                         mode='lines+markers', name='Si% (last 7 days)', line=dict(color='blue')))\n\n# Highlight March 23 11AM\nfig.add_trace(go.Scatter(x=[mar23_11am.name], y=[mar23_11am['Offline[HM & Slag] - chem_pct_si']], \n                         mode='markers', name='Mar 23 11AM', marker=dict(size=10, color='red', symbol='x')))\n\n# Add reference line at 0.2%\nfig.add_hline(y=0.2, line_dash='dash', line_color='green', annotation_text='0.2% Reference')\n\n# Title and labels\nfig.update_layout(title='Hot Metal Silicon Content - Last 7 Days', \n                   xaxis_title='Sample Index (hourly)', yaxis_title='Silicon Content (%)',\n                   showlegend=True)\n\nfig.show()"}`

**Error:**

```
Disallowed token in code: \bimport\b
```


## 2026-03-25T12:38:47.457184+00:00
**Tool:** execute_python_plot

**Params:** `{"code": "import pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\n\ndf = pd.read_csv('current_furnace_data.csv', index_col=0, parse_dates=True)\n\n# Find coke rate column\ncoke_col = 'Process Params - coke_rate'\n\n# Sort by coke rate (ascending) to find best (lowest) conditions\ndf_sorted = df.sort_values(coke_col)\n\n# Get top 10 best coke rate periods\ntop_10_best = df_sorted.head(10)\n\n# Statistics\nprint(\"=== COKE RATE ANALYSIS - LAST 7 DAYS ===\\n\")\nprint(f\"Coke Rate Range: {df[coke_col].min():.1f} - {df[coke_col].max():.1f} kg/tHM\")\nprint(f\"Average Coke Rate: {df[coke_col].mean():.1f} kg/tHM\")\nprint(f\"Best Coke Rate: {df[coke_col].min():.1f} kg/tHM\")\nprint(f\"Worst Coke Rate: {df[coke_col].max():.1f} kg/tHM\")\n\nprint(\"\\n=== TOP 10 BEST COKE RATE PERIODS ===\")\nfor idx, row in top_10_best.iterrows():\n    print(f\"\\n{idx.strftime('%Y-%m-%d %H:%M')}: Coke={row[coke_col]:.1f} kg/tHM\")\n    print(f\"  PCI={row['Process Params - coal_rate_actual_value']:.1f} kg/tHM\")\n    print(f\"  Fuel Rate={row['Process Params - fuel_rate']:.1f} kg/tHM\")\n    print(f\"  ETA CO={row['Process Params - BF2_BODY_ETACO']:.1f}%\")\n    print(f\"  Production={row['Process Params - production_per_hour']:.1f} t/hr\")\n    print(f\"  HB Temp={row['Process Params - BF2_PROC Hot Blast Temp']:.0f}°C\")\n    print(f\"  HB Volume={row['Process Params - BF2_PROC Hot Blast Volume']:.0f} Nm³/h\")\n    print(f\"  O2 Enrichment={row['Process Params - oxygen_enrichment_pct']:.2f}%\")\n\n# Create visualization\nfig = go.Figure()\n\n# Time series of coke rate\nfig.add_trace(go.Scatter(x=df.index, y=df[coke_col], mode='lines', name='Coke Rate', line=dict(color='blue')))\n\n# Highlight best periods\nfig.add_trace(go.Scatter(x=top_10_best.index, y=top_10_best[coke_col], \n                         mode='markers', name='Top 10 Best', marker=dict(size=10, color='green', symbol='star')))\n\nfig.update_layout(title='Coke Rate - Last 7 Days (Best periods highlighted)', \n                   xaxis_title='Time', yaxis_title='Coke Rate (kg/tHM)')…"}`

**Error:**

```
Disallowed token in code: \bimport\b
```
