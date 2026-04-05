import pandas as pd


def prompt_recommendation_system(
    df_hist: pd.DataFrame, optimal_solution: pd.DataFrame, target_output: str
) -> str:
    """
    Generate a prompt for the recommendation system based on historical data and target parameter.

    Args:
        df_hist (pd.DataFrame): Historical data containing feature vectors.
        optimal_solution (pd.DataFrame): DataFrame containing the optimal solution parameters.
        target_output (str): The output metric to be optimized (e.g., "Unit Cost").
        target_param (str): The parameter to be optimized.

    Returns:
        str: A formatted prompt string for the recommendation system.
    """
    return f"""
        You are a blast furnace burden advisor. Analyze the impact of process parameters and raw material composition on **Unit Cost**.
        I have priorly done this analysis and found key drivers and best practices. Now generate the findings
        without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
        Report everything as Markdown only. 

        Based on the optimisation results, provide specific recommendations to improve {target_output}.
        - Optimal solution computed using current methodology (as predicted by model) {optimal_solution[target_output + '_current']} 
        which compares to previous value (as predicted by model) {optimal_solution[target_output + '_previous']}.
        - Current operating point {df_hist.iloc[-1].to_dict()}.
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
    
    You are an expert data scientist working on optimizing the parameter '{target_output}'.
    Based on the historical data provided, suggest actionable recommendations to improve '{target_output}'.
    
    Historical Data:
    {df_hist.tail(5).to_string(index=False)}
    
    Please provide your recommendations in a clear and concise manner.
    """
