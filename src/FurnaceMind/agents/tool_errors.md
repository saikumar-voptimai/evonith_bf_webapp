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
