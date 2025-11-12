import pandas as pd
from utils.prompts import build_summarization_AD_prompt
from utils.LLMs import openRouter_llm


class DataSummaryHelper:

    def __init__(self):
        pass


    def summarize_report(self, report: str) -> str:
        prompt = build_summarization_AD_prompt(report)
        system = "You are a helpful assistant that summarizes anomaly detection reports."
        response = openRouter_llm(system, prompt)
        return response
        
    
    def df_summary(self, recent_df):
        try:
            return pd.DataFrame(recent_df).describe().to_markdown()
        except Exception:
            return "Invalid DataFrame provided."
        

default_summarizer = DataSummaryHelper()

def summarize_report(report: str) -> str:
    return default_summarizer.summarize_report(report)

def df_summary(recent_df):
    return default_summarizer.df_summary(recent_df)