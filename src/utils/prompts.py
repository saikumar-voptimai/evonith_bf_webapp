from typing import Optional, Dict, Callable
import yaml
from pathlib import Path
import warnings
import pandas as pd
from config.config_loader import load_config
from utils.packets import DFPacket

def _load_mapping_config() -> Dict:
    """
    Load mapping config robustly:
     - try config_loader with 'mapping.yml'
     - fallback to 'mappings.yml'
     - fallback to directly reading src/config/mappings.yml or src/config/mapping.yml by path
     - return empty dict if nothing found (and warn)
    """
    try:
        return load_config("mapping.yml")
    except FileNotFoundError:
        pass
    try:
        return load_config("mappings.yml")
    except FileNotFoundError:
        pass

    # try filesystem locations relative to repo
    repo_config_dir = Path(__file__).resolve().parents[1] / "config"
    for name in ("mapping.yml", "mappings.yml"):
        p = repo_config_dir / name
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                warnings.warn(f"Failed to parse mapping file {p}", RuntimeWarning)

    warnings.warn("mapping.yml / mappings.yml not found; using empty config_map", RuntimeWarning)
    return {}

# load once
config_map = _load_mapping_config()


class Prompts:
    """
    Object-oriented wrapper for prompt builders.
    Use instance methods to get prompts (keeps same content as previous functions).
    """

    def __init__(self) -> None:
        # no state required currently
        pass

    def recommendation_system_prompt(self, target_output: str, optimal_solution: str, new_df: pd.DataFrame) -> str:
        """
        Build a recommendation prompt for an optimisation result.

        Args:
            target_output: The name or description of the target metric to improve.
            optimal_solution: Text or structure describing the computed optimal solution.
            new_df: Current operating point as a pandas DataFrame (will be included in the prompt).

        Returns:
            A multi-line string prompt instructing the model to produce concise,
            actionable recommendations based on the optimisation and current state.
        """
        # build by concatenation to avoid accidental format collisions with braces in text
        part1 = """

    You are a blast furnace burden advisor. Analyze the impact of process parameters and raw material composition on **Unit Cost**.
    I have priorly done this analysis and found key drivers and best practices. Now generate the findings
    without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
    Report everything as Markdown only. 

    Based on the optimisation results, provide specific recommendations to improve """
        part2 = str(target_output)
        part3 = """.
    - Optimal solution computed using current methodology """
        part4 = str(optimal_solution)
        part5 = """.
    - Current operating point """
        part6 = str(new_df.to_dict())
        part7 = """.
    - List the top 3-5 control parameters to adjust, with their new values. Note that we say 
    the furnace is already optimally operated if the target output change is less than 1%.

    - Provide a brief rationale for each recommendation.
    - Use bullet points for clarity.
    - Avoid vague statements; be specific and data-driven.

    Previous data analysis observations only for your reference (do not repeat in output):
    

    # 🔍 High-Confidence Findings (Statistically Significant Drivers)

    **Sign convention:**  
    - Negative = higher value → lower fuel rate (fuel-saving)  
    - Positive = higher value → higher fuel rate (fuel-raising)  
    (Standardized OLS β shown for relative effect size; all significant with *p* < 0.05)

    ---

    ## 1️⃣ Strongest Levers During Apr–Jun 2024 (All Models Agree)

    ### **Hot Blast Pressure (NEGATIVE)**
    - Spearman ρ ≈ −0.49; RF importance notable; OLS significant negative.  
    - Apr–Jun higher by **+0.033 bar** (*p* ~ 3.0e-73).  
    ➡️ Higher wind pressure strongly linked to lower fuel rate.

    ---

    ### **Hot Blast Volume (NEGATIVE)**
    - Negative direction in OLS and correlations; non-trivial RF importance.  
    - Apr–Jun higher by **+6,543 Nm³/hr** (*p* ≪ 1e-10).  
    ➡️ More wind volume aligned with the low-fuel window.

    ---

    ### **Flux Addition Rate (FLUX_MT) (NEGATIVE)**
    - Spearman ρ ≈ −0.27; RF ranked high; OLS coefficient negative.  
    - Apr–Jun higher by **+0.57 t/h** (*p* ≈ 1.4e-5).  
    ➡️ More flux correlated with lower fuel at current burden chemistry.

    ---

    ### **Sinter Basicity (NEGATIVE)**
    - RF and OLS show lower basicity aligns with lower fuel.  
    - Apr–Jun slightly lower (−0.0019; *p* ≈ 0.20).  
    ➡️ Small but consistent with fuel-saving effect.

    ---

    ### **Coke Ash % (POSITIVE)**
    - Higher ash increases fuel demand.  
    - Apr–Jun lower by **−0.38 %** (*p* ≪ 1e-10).  
    ➡️ Lower coke ash supported the fuel reduction.

    ---

    ### **Na₂O in Sinter (POSITIVE)**
    - Higher Na₂O raises fuel demand.  
    - Apr–Jun lower by **−0.0011 %** (*p* ≪ 1e-20).  
    ➡️ Keeping Na₂O low is beneficial.

    ---

    📌 **Net Effect (Apr–Jun 2024):**  
    - **Fuel-saving ↑:** Wind pressure, wind volume, flux rate  
    - **Fuel-saving ↓:** Coke ash, Sinter Na₂O, Sinter basicity  

    ---

    ## 2️⃣ Levers That Moved *Against* the Fuel Decrease

    ### **Injected Fuel (ActualKg/Thm., PCI) (POSITIVE, Strongest)**
    - RF importance ~0.73; β ≈ +4.93.  
    - Apr–Jun higher by **+15.8 kg/thm** (*p* ≪ 1e-10).  
    ➡️ PCI raised Act. Fuel Rate, but was offset by wind/material gains.

    ---

    ### **Hot Blast Temperature °C (POSITIVE)**
    - Positive correlation with fuel rate.  
    - Apr–Jun higher by **+13 °C** (*p* ≪ 1e-200).  
    ➡️ Likely operational coupling, not causal.

    ---

    ### **Sinter Al₂O₃ % (POSITIVE)**
    - Higher alumina raised fuel.  
    - Apr–Jun slightly higher (+0.076 %; *p* ≪ 1e-60).  
    ➡️ Reducing Al₂O₃ helps.

    ---

    ## ✅ Which Variables Drove the Low Fuel Rate?

    **Fuel-reducing (↑ increased):**
    - Hot Blast Pressure  
    - Hot Blast Volume  
    - Flux Rate  

    **Fuel-reducing (↓ decreased):**
    - Coke Ash %  
    - Sinter Na₂O %  
    - Sinter Basicity  

    **Counter-direction (increased fuel, but offset):**
    - PCI (Actual Fuel Rate)  
    - Hot Blast Temp °C  
    - Sinter Al₂O₃ %  

    ---

    ## 📌 Actionable Operating Guidance (Data-Driven)

    - **Maintain higher HB pressure & wind volume** within safe limits.  
    - **Sustain or increase flux rate** while balancing slag chemistry.  
    - **Procure lower coke ash** (≤ Apr–Jun median).  
    - **Keep Na₂O in sinter low** via blending & fines control.  
    - **Maintain slightly lower basicity** (avoid over-basic burdens).  
    - **Be cautious with HBT** – correlation is operational, not causal.  
    - **PCI**: If the goal is absolute min. fuel, consider steady or reduced PCI at high wind/low-ash conditions.

    ---

    # ⚖️ Optimization for Unit Fuel Cost

    ### Definition
    Fuel_CostEq (kgCokeEq/thm) = Coke Rate + 0.53 * PCI
    ---

    ## 🔑 Key Results
    - **Apr–Jun 2024 Avg:** **483.0 kgCokeEq/thm**  
    - **Best Historic 7-Day Run (B7D):** **473.4 kgCokeEq/thm**  
    (≈ **−9.6** vs Apr–Jun)  
    - **Production:** A–J **89.96 t/h** vs B7D **82.51 t/h**  
    ➡️ B7D was cheaper but ran slower.

    ---

    ## 📉 What Drove Low Cost in Apr–Jun 2024

    **Operational Drivers**
    - HB Pressure ↑ → Cost ↓ (β ≈ −1.78, *p*≪0.001)  
    - Wind Volume ↑ → Cost ↓  
    - Flux Rate ↑ → Cost ↓ (β ≈ −1.08, *p*≪0.001)  
    - PCI ↑ → Cost ↓ (−0.35 kgCokeEq saved per +1 kg PCI)  
    - HB Temp ↑ → Cost ↑ (+0.15 per 1 °C)

    **Burden Chemistry**
    - Lower K₂O & Na₂O → Cost ↓  
    - Lower FeO, SiO₂, TiO₂ → Cost ↓  
    - Higher Al₂O₃ → Cost ↑

    **Sensitivities**
    - O₂ Enrichment: −2.58 kg/thm per +1 %  
    - HB Pressure: −4.4 per +0.05 bar  
    - Flux: −0.86 per +1 t/h  
    - HB Temp: +1.53 per +10 °C  
    - Wind Volume: −2.5 per +5,000 Nm³/h  

    ---

    ## 🔁 How the Best 7-Day Run Got Cheaper

    **Cost-Reducing Changes (B7D vs Apr–Jun):**
    - PCI +5.95 kg/thm (saved cost)  
    - O₂ Enrichment +0.93 % abs.  
    - Cleaner burden: Ore Al₂O₃ −1.80 %, Ore K₂O −0.019 %, Sinter K₂O −0.042 %  

    **Offsets (fuel-raising moves but outweighed):**
    - HB Pressure −0.18 bar, Wind −16,400 Nm³/h  
    - Flux −1.36 t/h  
    - HB Temp −12.4 °C (helpful)

    **Trade-Off:** Lower throughput (−7.45 t/h).

    ---

    # 📊 Operating Envelope for Low Cost

    **Based on lowest-cost decile (not strict limits):**

    - PCI: **135–143 kg/thm**  
    - O₂ Enrichment: **2.55–3.56 %**  
    - HB Pressure: **2.65–2.70 bar**  
    - HB Temp: **1149–1164 °C**  
    - Flux: **18.6–22.9 t/h**  
    - Wind: **≥115k Nm³/h**  
    - Sinter K₂O: **≤0.11 %**  
    - Sinter Na₂O: **0.045–0.053 %**  
    - Ore K₂O: **≤0.031 %**  
    - Ore Al₂O₃: **≥3.3 % (maintain slag balance)**

    📌 Practical: To minimize cost, **lean on PCI + O₂**, **keep HB pressure up**, **moderate HB temp**, **add flux**, and **suppress alkalis**.

    ---

    # ✅ Action Checklist

    1. **Target PCI** where substitution ratio (Coke/PCI = 0.53).  
    (You are at −0.88 to −0.91 → PCI is cost-saving).  
    2. **Hold HB pressure high** (2.65–2.70 bar).  
    3. **Increase O₂ enrichment** (3.0–3.5 % if feasible).  
    4. **Maintain flux ≥ 19 t/h**.  
    5. **Manage chemistry**: Keep K₂O/Na₂O low, blend ore to reduce alkalis.  
    6. **Throughput trade-off**: B7D was cheaper but slower → optimize cost *and* t/h jointly.
    """
        return part1 + part2 + part3 + part4 + part5 + part6 + part7

    def build_review_prompt(self, target_label: str, df1: pd.DataFrame, df2: pd.DataFrame | None, best_snap: dict) -> str:
        """
        Construct a concise review prompt comparing two timeframes and referencing a historical best.

        Args:
            target_label: Human-readable metric label used as key in METRIC_MAP.
            df1: DataFrame for timeframe 1.
            df2: Optional DataFrame for timeframe 2.
            best_snap: Optional dict with keys 'value' and 'when' describing historical best.

        Returns:
            A prompt string summarising inputs and instructing the model to produce a short,
            actionable review (drivers, recommendations, gap to best).

        Notes:
            This method expects METRIC_MAP and df_packet to be available in the module namespace.
        """
        # safe lookup
        metric = config_map.get("METRIC_MAP", {}).get(target_label)
        if metric is None:
            raise KeyError(f"Metric mapping for '{target_label}' not found in mappings.yml (METRIC_MAP)")

        # use DFPacket instance to render packets
        packetter = DFPacket()
        pkt1 = packetter.packet(df1[[metric]].dropna()) if metric in df1.columns else "_(Metric absent in CSV)_"
        pkt2 = packetter.packet(df2[[metric]].dropna()) if (df2 is not None and metric in df2.columns) else "_(No Timeframe 2)_"
        best_line = f"{best_snap['value']:.3f} at {best_snap['when']}" if best_snap else "N/A"

        return f"""
                You are a senior blast furnace advisor. Be concise, numeric, and actionable.

                # Target
                - Metric: **{target_label}** (`{metric}`)
                - Historical best (Apr–Jun 2024): **{best_line}** (ETA CO → max; Total Fuel → min)

                # Data
                ## Timeframe 1
                {pkt1}

                ## Timeframe 2
                {pkt2}

                # Output
                0) If you sense the furnace is shutdown, reply "Furnace is shutdown, no data available."
                1) Executive verdict (2–3 lines).
                2) Drivers of difference (hot blast temp/vol/press, O₂, steam, PCI, top pressure, permeability, silicon, heatloads, etc.).
                3) Recommendations (setpoint nudges + rationale + trade-offs).
                4) Gap to historical best and what to emulate.
                5) Comparison of last 2days operation vs 7days vs 30days (if available).
                """

    def build_unitcost_prompt(self) -> str:
        """
        Return a standard Unit Cost analysis prompt.

        Returns:
            A long multi-line string instructing the model to analyze drivers of Unit Cost
            and provide findings, guidance and sensitivities in Markdown.
        """
        return """
            You are a blast furnace burden advisor. Analyze the impact of process parameters and raw material composition on **Unit Cost**.
            I have priorly done this analysis and found key drivers and best practices. Now generate the findings
            without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
            Report everything as Markdown only. 

            # 🔍 High-Confidence Findings (Statistically Significant Drivers)

            **Sign convention:**  
            - Negative = higher value → lower fuel rate (fuel-saving)  
            - Positive = higher value → higher fuel rate (fuel-raising)  
            (Standardized OLS β shown for relative effect size; all significant with *p* < 0.05)

            ---

            ## 1️⃣ Strongest Levers During Apr–Jun 2024 (All Models Agree)

            ### **Hot Blast Pressure (NEGATIVE)**
            - Spearman ρ ≈ −0.49; RF importance notable; OLS significant negative.  
            - Apr–Jun higher by **+0.033 bar** (*p* ~ 3.0e-73).  
            ➡️ Higher wind pressure strongly linked to lower fuel rate.

            ---

            ### **Hot Blast Volume (NEGATIVE)**
            - Negative direction in OLS and correlations; non-trivial RF importance.  
            - Apr–Jun higher by **+6,543 Nm³/hr** (*p* ≪ 1e-10).  
            ➡️ More wind volume aligned with the low-fuel window.

            ---

            ### **Flux Addition Rate (FLUX_MT) (NEGATIVE)**
            - Spearman ρ ≈ −0.27; RF ranked high; OLS coefficient negative.  
            - Apr–Jun higher by **+0.57 t/h** (*p* ≈ 1.4e-5).  
            ➡️ More flux correlated with lower fuel at current burden chemistry.

            ---

            ### **Sinter Basicity (NEGATIVE)**
            - RF and OLS show lower basicity aligns with lower fuel.  
            - Apr–Jun slightly lower (−0.0019; *p* ≈ 0.20).  
            ➡️ Small but consistent with fuel-saving effect.

            ---

            ### **Coke Ash % (POSITIVE)**
            - Higher ash increases fuel demand.  
            - Apr–Jun lower by **−0.38 %** (*p* ≪ 1e-10).  
            ➡️ Lower coke ash supported the fuel reduction.

            ---

            ### **Na₂O in Sinter (POSITIVE)**
            - Higher Na₂O raises fuel demand.  
            - Apr–Jun lower by **−0.0011 %** (*p* ≪ 1e-20).  
            ➡️ Keeping Na₂O low is beneficial.

            ---

            📌 **Net Effect (Apr–Jun 2024):**  
            - **Fuel-saving ↑:** Wind pressure, wind volume, flux rate  
            - **Fuel-saving ↓:** Coke ash, Sinter Na₂O, Sinter basicity  

            ---

            ## 2️⃣ Levers That Moved *Against* the Fuel Decrease

            ### **Injected Fuel (ActualKg/Thm., PCI) (POSITIVE, Strongest)**
            - RF importance ~0.73; β ≈ +4.93.  
            - Apr–Jun higher by **+15.8 kg/thm** (*p* ≪ 1e-10).  
            ➡️ PCI raised Act. Fuel Rate, but was offset by wind/material gains.

            ---

            ### **Hot Blast Temperature °C (POSITIVE)**
            - Positive correlation with fuel rate.  
            - Apr–Jun higher by **+13 °C** (*p* ≪ 1e-200).  
            ➡️ Likely operational coupling, not causal.

            ---

            ### **Sinter Al₂O₃ % (POSITIVE)**
            - Higher alumina raised fuel.  
            - Apr–Jun slightly higher (+0.076 %; *p* ≪ 1e-60).  
            ➡️ Reducing Al₂O₃ helps.

            ---

            ## ✅ Which Variables Drove the Low Fuel Rate?

            **Fuel-reducing (↑ increased):**
            - Hot Blast Pressure  
            - Hot Blast Volume  
            - Flux Rate  

            **Fuel-reducing (↓ decreased):**
            - Coke Ash %  
            - Sinter Na₂O %  
            - Sinter Basicity  

            **Counter-direction (increased fuel, but offset):**
            - PCI (Actual Fuel Rate)  
            - Hot Blast Temp °C  
            - Sinter Al₂O₃ %  

            ---

            ## 📌 Actionable Operating Guidance (Data-Driven)

            - **Maintain higher HB pressure & wind volume** within safe limits.  
            - **Sustain or increase flux rate** while balancing slag chemistry.  
            - **Procure lower coke ash** (≤ Apr–Jun median).  
            - **Keep Na₂O in sinter low** via blending & fines control.  
            - **Maintain slightly lower basicity** (avoid over-basic burdens).  
            - **Be cautious with HBT** – correlation is operational, not causal.  
            - **PCI**: If the goal is absolute min. fuel, consider steady or reduced PCI at high wind/low-ash conditions.

            ---

            # ⚖️ Optimization for Unit Fuel Cost

            ### Definition
            FuelCostEq (kgCokeEq/thm)} = Coke Rate + 0.53 * PCI

            ---

            ## 🔑 Key Results
            - **Apr–Jun 2024 Avg:** **483.0 kgCokeEq/thm**  
            - **Best Historic 7-Day Run (B7D):** **473.4 kgCokeEq/thm**  
            (≈ **−9.6** vs Apr–Jun)  
            - **Production:** A–J **89.96 t/h** vs B7D **82.51 t/h**  
            ➡️ B7D was cheaper but ran slower.

            ---

            ## 📉 What Drove Low Cost in Apr–Jun 2024

            **Operational Drivers**
            - HB Pressure ↑ → Cost ↓ (β ≈ −1.78, *p*≪0.001)  
            - Wind Volume ↑ → Cost ↓  
            - Flux Rate ↑ → Cost ↓ (β ≈ −1.08, *p*≪0.001)  
            - PCI ↑ → Cost ↓ (−0.35 kgCokeEq saved per +1 kg PCI)  
            - HB Temp ↑ → Cost ↑ (+0.15 per 1 °C)

            **Burden Chemistry**
            - Lower K₂O & Na₂O → Cost ↓  
            - Lower FeO, SiO₂, TiO₂ → Cost ↓  
            - Higher Al₂O₃ → Cost ↑

            **Sensitivities**
            - O₂ Enrichment: −2.58 kg/thm per +1 %  
            - HB Pressure: −4.4 per +0.05 bar  
            - Flux: −0.86 per +1 t/h  
            - HB Temp: +1.53 per +10 °C  
            - Wind Volume: −2.5 per +5,000 Nm³/h  

            ---

            ## 🔁 How the Best 7-Day Run Got Cheaper

            **Cost-Reducing Changes (B7D vs Apr–Jun):**
            - PCI +5.95 kg/thm (saved cost)  
            - O₂ Enrichment +0.93 % abs.  
            - Cleaner burden: Ore Al₂O₃ −1.80 %, Ore K₂O −0.019 %, Sinter K₂O −0.042 %  

            **Offsets (fuel-raising moves but outweighed):**
            - HB Pressure −0.18 bar, Wind −16,400 Nm³/h  
            - Flux −1.36 t/h  
            - HB Temp −12.4 °C (helpful)

            **Trade-Off:** Lower throughput (−7.45 t/h).

            ---

            # 📊 Operating Envelope for Low Cost

            **Based on lowest-cost decile (not strict limits):**

            - PCI: **135–143 kg/thm**  
            - O₂ Enrichment: **2.55–3.56 %**  
            - HB Pressure: **2.65–2.70 bar**  
            - HB Temp: **1149–1164 °C**  
            - Flux: **18.6–22.9 t/h**  
            - Wind: **≥115k Nm³/h**  
            - Sinter K₂O: **≤0.11 %**  
            - Sinter Na₂O: **0.045–0.053 %**  
            - Ore K₂O: **≤0.031 %**  
            - Ore Al₂O₃: **≥3.3 % (maintain slag balance)**

            📌 Practical: To minimize cost, **lean on PCI + O₂**, **keep HB pressure up**, **moderate HB temp**, **add flux**, and **suppress alkalis**.

            ---

            # ✅ Action Checklist

            1. **Target PCI** where substitution ratio {d Coke}/{d PCI} <= -0.53).  
            (You are at −0.88 to −0.91 → PCI is cost-saving).  
            2. **Hold HB pressure high** (2.65–2.70 bar).  
            3. **Increase O₂ enrichment** (3.0–3.5 % if feasible).  
            4. **Maintain flux ≥ 19 t/h**.  
            5. **Manage chemistry**: Keep K₂O/Na₂O low, blend ore to reduce alkalis.  
            6. **Throughput trade-off**: B7D was cheaper but slower → optimize cost *and* t/h jointly.
            """

    def build_report_prompt(self, df: pd.DataFrame, label: str) -> str:
        """
        Create a structured report prompt for a provided DataFrame.

        Args:
            df: DataFrame containing the data to be summarised (columns will be cleaned).
            label: Short title/label for the report section.

        Returns:
            A prompt string asking the model to produce a structured operations/thermal/burden report.
        """
        packetter = DFPacket()
        pkt = packetter.packet(df.dropna(axis=1, how="all"))
        return f"""
                You are a blast furnace reviewer. Create a structured report for **{label}**.

                # Data
                {pkt}

                # Deliverables
                - Operations snapshot (throughput, Total Fuel, ETA CO, stability).
                - Thermal profile (top/bosh temps, skin temps by level, heatloads, ΔT).
                - Burden quality & quantity (coke/nutcoke, sinter, CLO; PCI quality).
                - Outputs (HM/slag key analysis—e.g., silicon).
                - Deviations vs typical.
                - Recommendations (concrete levers + expected effect).
                """

    def build_bunker_unitcost_prompt(self) -> str:
        """
        Return a bunker/burden-distribution focused Unit Cost prompt.

        Returns:
            A multi-line string instructing the model to analyze how burden distribution
            and related process parameters influence Unit Cost, and to provide actionable steps.
        """
        return """
            You are a blast furnace burden advisor. Analyze the impact of mainly the burden distribution on **Unit Cost** and to some extent
            combined with the rawmaterial and process parameters.
            I have priorly done this analysis and found key drivers and best practices. Now generate the findings
            without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
            Report everything as Markdown only. 

            Findings:

            # What the data say (linked to your Unit Cost)

            ### Reminder of Unit Cost
            `Unit_Cost = Coke Rate Kg/Thm + 0.53 × ActualKg/Thm.`

            ---

            ## Modeling approach
            **Model:** Transparent linear model (OLS) to reduce confounding.

            **Controls:**  
            - Hot Blast Temp  
            - TopPressureBar  
            - O₂ Enrichment %  
            - ETA CO *(FurnaceTopGasAnalysisCO2ETACO)*  
            - PCI ActualKg/Thm

            **Burden features:**  
            - `portions_total_COKE`, `portions_total_NON COKE`  
            - `angle_wmean_COKE`, `angle_wmean_NON COKE`  
            - `outer_share_COKE`, `outer_share_NON COKE`  
            - `lmg_angle`

            **Model quality:**  
            - **R² ≈ 0.43** on ~5,900 valid rows *(burden + controls explain ~43% of Unit Cost variance)*.  
            - *(Charts: “Standardized Effects…” visualize these effects.)*

            ---

            ## Most influential & directionally consistent effects *(holding controls constant)*
            - **More NON-COKE portions → lower Unit Cost** *(strong, significant).*  
            *Intuition:* better ore/sinter coverage enabling efficient gas flow.
            - **More COKE portions → higher Unit Cost** *(strong, significant).*  
            *Intuition:* coke rings consume more and drive cost up.
            - **Pushing NON-COKE outward (higher `angle_wmean_NON COKE`) → lower Unit Cost** *(significant).*  
            *Intuition:* spreading ore toward the periphery improves permeability/ETA and saves fuel.
            - **Higher LMG angle → slight reduction in Unit Cost** *(small but significant negative coefficient in a sparse, de-collinear model).*
            - **Outer share of COKE:** small negative coefficient *(cost ↓)*, not statistically strong once other features enter.

            ---

            ## Process controls behaved as expected
            - **Higher PCI rate** and **higher ETA CO** both **reduce Unit Cost**.
            - **Higher Hot Blast Temp** **increases** cost *(likely proxying for periods requiring more heat input).*

            ---

            ## “Best burden distributions” found in your history
            Each distribution change interval was evaluated by the **realized mean Unit Cost** until the next change (and we also tracked Coke rate, PCI, and ETA CO).

            **Top performing change windows (≥300 data rows; long enough to trust):**  
            *(See full table in “Top 25 Best Burden Events…”)*

            1. **2024-03-28 17:00 — mean Unit Cost ≈ 487.0**  
            - COKE portions: **11**, NON-COKE portions: **8**  
            - `angle_wmean_COKE` ≈ **26.0°**, `angle_wmean_NON-COKE` ≈ **28.0°**  
            - `outer_share_NON-COKE`: **0.25**  
            - **LMG angle:** **42.5** *(pattern “P TO C”)*  
            - **Purpose:** “TO IMPROVE THE CENTER GAS FLOW”.

            2. **2024-11-08 12:47 — mean Unit Cost ≈ 489.5**  
            - COKE portions: **37**, NON-COKE portions: **24** *(multiple ring sets logged at the same timestamp; summed)*  
            - `angle_wmean_COKE` ≈ **27.4°**, `angle_wmean_NON-COKE` ≈ **28.5°**  
            - `outer_share_NON_COKE`: **0.25**  
            - **Purpose:** “TO INCREASE THE UTILISATION”.

            3. **2025-08-02 10:00 — mean Unit Cost ≈ 492.0**  
            - COKE: **11**, NON-COKE: **8**  
            - `angle_wmean_COKE` ≈ **26.7°**, `angle_wmean_NON-COKE` ≈ **28.0°**  
            - `outer_share_NON_COKE`: **0.25**

            4. **2024-08-20 20:00 — mean Unit Cost ≈ 493.9**  
            - COKE: **11**, NON-COKE: **8**  
            - `angle_wmean_COKE` ≈ **26.8°**, `angle_wmean_NON-COKE` ≈ **28.3°**  
            - `outer_share_NON_COKE`: **0.33**  
            - **Purpose:** “TO CONTROL PRE”.

            ---

            ## Common pattern across best windows
            - **Moderate COKE portions (~10–11)** & **adequate NON-COKE portions (~8)**.  
            - **NON-COKE weighted angle near ~28°** and **≥25% in the outer ring (≥32°)**.  
            - **LMG angle ~40–43** with **P→C charging pattern** frequently noted.  
            - These windows also show **good ETA CO** and **healthy PCI**.

            > **Rule of thumb (from your data):**  
            > Keep **coke portions lean**, keep **non-coke portions ample**, and **bias the non-coke outward** *(center-of-mass ~28° with ≥25% outer share).*


            **Key drivers summary:**  
            - **NON-COKE `portions_total`** is the strongest cost **reducer**.  
            - **COKE `portions_total`** is the strongest cost **increaser**.  
            - **NON-COKE weighted angle** reduces cost *(more outer non-coke).*


            ---


            ## Why this is faithful to your physical process
            - Respects the **6-row block design**; carries **date/time forward** so every material record is stamped correctly.
            - Pairs each **“RINGS”** row with its **following “Angle”** row **only** to obtain degrees *(avoids double-counting portions)*.
            - Separates **`metric=portions`** vs **`metric=percent`** so Extra-Coke “IN %” entries don’t pollute portions counts.
            - Models **change windows** *(pattern holds until next change)*, not isolated points.

            ---

            ## Actionable next steps (recommended)
            - **Lock a candidate best pattern from history:**
            - **~10–11 COKE portions**, **~8 NON-COKE portions**
            - **Target `angle_wmean_NON-COKE` ≈ 28°**, **≥25% outer share**
            - **LMG angle ~40–43**, **charging pattern P→C**
            """

    def build_anomaly_prompt(self, recent_df: pd.DataFrame, notes: str = "") -> str:
        """
        Build an anomaly-spotting prompt using recent timeseries.

        Args:
            recent_df: Recent timeseries DataFrame (averaged packet or raw) to include in the prompt.
            notes: Optional operator notes to include.

        Returns:
            A prompt string instructing the model to report key anomalies (Z-score style),
            one line per issue, with brief observations, alerts and likely causes mapped to controllables.
        """
        # if not recent_df.empty:
        # # Convert the describe DataFrame to markdown string for readable prompt embedding
        #     pkt = recent_df.describe().to_markdown()
        # else:
        #     pkt = "_No timeseries to show_"
        
        # return f"""
        #         You are an expert blast furnace anomaly analyst.

        #         Your task is to examine operational data from the **last 8-hour shift** and identify significant anomalies or deviations using **Z-score-based reasoning**.  
        #         Each anomaly must be reported **concisely in one line per issue**, avoiding repetition or speculation.

        #         ---

        #         ### Context of Data Provided
        #         You have access to time-series process data (recent_df) averaged over 15-minute intervals.

        #         1. **Temperature Profile (Furnace Body)**
        #         - Variable: `"Temperature Profile - BF2_BFBF Furnace Body [furnace_level]mm Temp [circumferential_position]"`
        #         - Represents circumferential temperature readings at multiple furnace levels.
        #         - Level descriptions and proxy names:
        #                 - 4373mm → 7 sensors  
        #                 - 5411mm → 13 sensors  
        #                 - 5757mm → 13 sensors (**Hearth**)  
        #                 - 6103mm → 13 sensors  
        #                 - 6795mm → 12 sensors  
        #                 - 7565mm → 14 sensors  
        #                 - 8335mm → 14 sensors  
        #                 - 9105mm → 12 sensors  
        #                 - 12975mm → 4 sensors (**Bosh**)  
        #                 - 15162mm → 4 sensors (**Belly**)  
        #                 - 18660mm → 4 sensors (**Stack**)  
        #                 - **Tuyeres located at 10500mm**

        #         2. **Heatload Data**
        #         - Format examples:
        #                 - `"Heatload Delta T - Heat load R8 Q3 (Stave No 17-24)"`: average heatload for **Quadrant 3**, staves 17–24.  
        #                 - `"Heatload Delta T - Heat load Row6-10 Q1 (Stave No 17-24)"`: average heatload for **Rows 6–10**, Quadrant 1.  
        #                 - `"Heatload Delta T - Heat load Row6"`: average heatload in **Row 6** across all quadrants.
        #         - Look for abnormal heatload spikes, ΔT excursions, or quadrant-wise imbalances.

        #         3. **Process Parameters**
        #         - Includes: hot blast pressure, blast volume, hot blast temperature, O₂ enrichment, steam rate, PCI (coal rate), etc.
        #         - These influence the stability of the furnace and must be referenced when explaining anomalies.

        #         ---

        #         ### Review Task
        #         Analyze the **most recent 8-hour packet** (15-min averaged):

        #         {pkt}

        #         Use Z-score or deviation reasoning to identify:
        #         - Furnace profile **temperature spikes**
        #         - **Heatload excursions** or ΔT jumps
        #         - **Gas pressure / flow instability**
        #         - Indications of **startup**, **shutdown**, or **blowdown** transitions

        #         Also refer to any operator comments if available:

        #         **Operator Notes:**  
        #         {notes}

        #         ---

        #         ### Response Format (Concise and Structured)
        #         **Output must be under 200 words. Do not repeat points. Avoid unnecessary elaboration.**

        #         1. **Key Observations (2–3 lines)**  
        #         - Summarize the operational trend for the last 8 hours.  
        #         - Mention if the furnace appears stable, heating up, cooling down, or in transition (startup/shutdown).

        #         2. **Alerts (Issue + Severity)**  
        #         - List key anomalies with their severity level (Low / Medium / High).  
        #         - Example: “Hearth temperature increased by >3σ (High severity)” or “Blast volume oscillation ±8% (Medium severity).”

        #         3. **Likely Causes (Mapped to Controls)**  
        #         - Suggest probable causes linked to controllable parameters such as:  
        #             - HB temperature / volume / pressure  
        #             - O₂ enrichment / Steam rate / PCI rate  
        #             - Burden or sinter quality shifts  

        #         ---

        #         ###  Critical Instructions
        #         - **No hallucinations** — use only evidence from the provided data.  
        #         - **No further questions or assumptions** — operator cannot provide feedback now.  
        #         - **Stay factual and domain-grounded** — focus on measurable anomalies only.  
        #         - **One line per issue**, prioritize clarity over verbosity.

        #         """

        pkt = recent_df

        return f"""
                You are an anomaly spotter. Report the key anomalies in last 8hours shift (provided data) using Z-score in one line per issue.

                You also received raw data (recent_df) for:
                a. blast furnace temperature profile denoted "Temperature Profile - BF2_BFBF Furnace Body [furnace_level]mm Temp [circumferential_position]"
                    Description for Number of sensors at different levels and any proxy name (furnace profile) is given below
                    Desc:
                        "4373":
                        n_sensors: 7
                        "5411":
                        n_sensors: 13
                        "5757":
                        proxy_name: "Hearth"      
                        n_sensors: 13
                        "6103":
                        n_sensors: 13
                        "6795":
                        n_sensors: 12
                        "7565":
                        n_sensors: 14
                        "8335":
                        n_sensors: 14
                        "9105":
                        n_sensors: 12
                        "12975":
                        proxy_name: "Bosh"   
                        n_sensors: 4
                        "15162":
                        proxy_name: "Belly"  
                        n_sensors: 4
                        "18660":
                        proxy_name: "Stack"
                        n_sensors: 4

                        "Tuyeres" are located at 10500mm
                b. heatload at different levels denoted by row number. 
                    Ex: "Heatload Delta T - Heat load R8 Q3 (Stave No 17-24)" - Average heatload for Quadrant 3 (for staves 17 to 24)
                        "Heatload Delta T - Heat load Row6-10 Q1 (Stave No 17-24) - Average for Rows 6 to 10 but only Quadrant 3
                        "Heatload Delta T - Heat load Row6" - for average heatload in row 6 across all quadrants
                c. process_params:
                    - blast furnace top pressure, volume, temperature, O₂, steam, PCI (coal rate) etc

                Review the **last 8 hours** for furnace profile temperature spikes, heatload spikes, ΔT excursions,
                gas/pressure instabilities.

                # Recent 8hours packet (averaged to 15mins)
                {pkt}

                # Operator notes
                {notes}

                # Output in brief upto 200 words only
                - Key observations (2–3 lines) for operator of what happened in previous shift. 
                Like Blowdowns, StartUps, Shutdowns.
                Are heatloads increasing?
                Is fuel rate increasing?
                Is blast furnace stable?

                - Alerts (issue + severity).
                - Likely causes mapped to controllables (HB temp/volume/pressure, O₂, steam, PCI) and burden quality.

                NOTE: 1. Avoid any hallucincations and only stick to provided data. Don't be verbose and mention each point only once. 
                2. Currently the operator does not have access to provide prompt feedback. So dont ask questions/expect further input.
                """
    
    def build_summarization_AD_prompt(self, anomaly_report):
        return f""" You are a senior process engineer responsible for blast furnace performance monitoring.
        Your task is to summarize detailed anomaly reports generated from operational data.

        Each anomaly report may contain:
        - Parameter deviations (pressure, temperature, flow, speed, composition, etc.)
        - Heat-load trends and ΔT excursions across furnace zones
        - Time of occurrence and recovery
        - Actions taken by operators
        - Impacts on furnace health, burden distribution, or gas flow
        - Any associated equipment or zone references (staves, tuyeres, bustle main, etc.)

        Summarize the given anomaly report **factually and concisely** with a focus on:
        1. **Event Summary** — a 2–3 line description of what anomaly occurred and when.
        2. **Key Deviations** — clearly list critical **process parameters and heat-loads** that deviated, showing their **numerical changes or Z-scores** (e.g., “HB pressure +18 kPa”, “ΔT +3.2σ at Row8 Q3”).
        3. **Severity & Trend** — highlight any **severe changes** or sharp variations in values rather than long descriptions.
        4. **Probable Cause / Observation** — infer engineering reasoning only if explicitly mentioned (avoid speculation).
        5. **Impact Assessment** — describe how it affected furnace operation (e.g., top pressure, gas flow, fuel rate, slag ratio, heat-load balance).
        6. **Corrective Action / Outcome** — summarize operator responses and final status.

        Guidelines:
        - Prioritize **numbers over text**; include key figures, units, and direction of change.
        - Use an objective tone; avoid opinions or redundant phrasing.
        - Do not fabricate missing data or causes.
        - Keep the final summary within **100 words**.
        - The summary should help an AI system build accurate long-term memory of furnace behavior and parameter responses.

        Input report:
        {anomaly_report}

        Return only the **summarized text** (no labels or JSON).
        """
    
    def build_df_summary_prompt(self, df_markdown):
        """
        Creates a contextual LLM prompt to summarize blast furnace readings
        for operators and process engineers.
        """
        return f"""
            You are a metallurgical process analyst specializing in **blast furnace operations**.
            You are assisting a plant control engineer by summarizing recent **blast furnace readings**
            and highlighting key operational insights.

            ---

            ### 🔧 Context
            The blast furnace produces hot metal through the reduction of iron ore using coke and fluxes.
            The process health is determined by thermal, gas, and pressure balance across various zones
            (top, belly, bosh, hearth).

            The input data below represents **hourly or shift-level readings** of the furnace,
            including gas, pressure, temperature, and flow parameters.

            Each column may represent variables like:
            - **Blast parameters:** hot blast temperature, blast volume, O₂%, pressure.
            - **Thermal state:** tuyere temperature, stave temperature, hearth temperature, gas temperature.
            - **Gas balance:** CO%, CO₂%, N₂%, top gas temperature, top pressure.
            - **Burden and coke data:** burden descent rate, coke rate, sinter %, pellet %, flux basicity.
            - **Operational indicators:** productivity, fuel rate, permeability index, furnace pressure drop.

            ---

            ### 🎯 Your Objective
            From the following dataframe summary, generate a **human-readable, structured summary** that:

            1. **Highlights the overall furnace condition**  
            (e.g., stable, heating up, showing gas imbalance, thermal overload, low permeability, etc.)
            2. **Identifies deviations or anomalies**  
            compared to expected or historical norms (if visible from the data ranges).
            3. **Comments on thermal behavior**  
            (Is the furnace getting hotter, cooling, or showing uneven temperature distribution?)
            4. **Analyzes gas efficiency**  
            (CO utilization, top gas temperature, pressure balance — indicate efficiency or energy loss.)
            5. **Mentions burden and fuel patterns**  
            (Any visible trend in sinter %, coke rate, productivity, etc.)
            6. **Concludes with actionable remarks**  
            (e.g., “Maintain current blast pattern”, “Watch for potential channeling in upper zones”, 
            “Possible cooling imbalance near belly staves”, etc.)

            ---

            ### ⚙️ Output Requirements
            - Write **in concise, technical English** suitable for process engineers.
            - Use bullet points or small paragraphs.
            - Avoid restating raw numbers — interpret them.
            - End with a **1–2 line summary verdict** on furnace health.

            ---

            ### 📊 DataFrame Summary (Markdown)
            Below is the statistical summary of the recent readings:

            {df_markdown}

            ---

            ### 🚀 Generate your summary now.
            """
    
    def build_comparison_prompt(self, current_report, reference_report, condition_name):
        return f"""
            You are an expert metallurgical process analyst specializing in blast furnace operation patterns.
            Your task is to determine whether the CURRENT anomaly report indicates the specific operational condition: **{condition_name}**.

            ### Context
            Blast furnaces have distinct behavioral signatures during different phases:
            - **Startup:** Gradual increase in blast volume, top pressure, and temperatures; irregular readings stabilize progressively.
            - **Shutting Down:** Gradual decline in blast pressure, air volume, and burden descent rate; temperatures fall systematically.
            - **Shutdown:** Near-zero blast parameters, halted burden movement, minimal thermal activity, and sensor flatlines.

            ### Your Task
            Compare the CURRENT anomaly report to the REFERENCE report (which represents the ideal case for the condition: **{condition_name}**).

            You must analyze **contextual, semantic, and numerical alignment** — not just keyword similarity.  
            Focus on:
            - Trend similarities (e.g., rising, stabilizing, or falling parameters)
            - Process stages and sequence of changes
            - Thermal and pressure consistency patterns
            - Evidence of steady-state vs transient instability

            ### Reports
            **CURRENT REPORT:**
            {current_report}

            **REFERENCE REPORT ({condition_name}):**
            {reference_report}

            ### Response Rules
            1. Respond **only** with either `True` or `False`.  
            2. Respond `True` **only if** the CURRENT report strongly aligns with the REFERENCE report for the condition `{condition_name}`.  
            3. Respond `False` if the report does not align or matches another condition.
            4. Do not include any explanation, text, or additional reasoning in your output.
            """



# convenience default instance + module-level wrappers so other modules can import functions
_default_prompts = Prompts()

def recommendation_system_prompt(target_output: str, optimal_solution: str, new_df: pd.DataFrame) -> str:
    return _default_prompts.recommendation_system_prompt(target_output, optimal_solution, new_df)

def build_review_prompt(target_label: str, df1: pd.DataFrame, df2: pd.DataFrame | None, best_snap: dict) -> str:
    return _default_prompts.build_review_prompt(target_label, df1, df2, best_snap)

def build_unitcost_prompt() -> str:
    return _default_prompts.build_unitcost_prompt()

def build_report_prompt(df: pd.DataFrame, label: str) -> str:
    return _default_prompts.build_report_prompt(df, label)

def build_bunker_unitcost_prompt() -> str:
    return _default_prompts.build_bunker_unitcost_prompt()

def build_anomaly_prompt(recent_df: pd.DataFrame, notes: str = "") -> str:
    return _default_prompts.build_anomaly_prompt(recent_df, notes)

def build_summarization_AD_prompt(anomaly_report) -> str:
    return _default_prompts.build_summarization_AD_prompt(anomaly_report)

def build_df_summary_prompt(df_markdown: str) -> str:
    return _default_prompts.build_df_summary_prompt(df_markdown)   

def build_comparison_prompt(current_report, reference_report, codition_name) -> str:
    return _default_prompts.build_comparison_prompt(current_report, reference_report, codition_name) 