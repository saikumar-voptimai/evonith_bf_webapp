import pandas as pd


class DFPacket:
    """
    Convert pandas DataFrames to a markdown "packet" suitable for display.
    Usage:
        pkt = DFPacket(max_rows=160, round_digits=4)
        s = pkt.packet(df)         # or pkt(df)
    """

    def __init__(self, max_rows: int = 160, round_digits: int = 4) -> None:
        """
        Args:
            max_rows: maximum number of rows to include in the displayed sample.
            round_digits: number of decimal places to round numeric columns for display.
        """
        self.max_rows = max_rows
        self.round_digits = round_digits

    def df_packet(self, df: pd.DataFrame) -> str:
        """
        Build a markdown packet from a DataFrame.

        Args:
            df: pandas DataFrame to convert.

        Returns:
            A markdown string including a sampled table and summary statistics.
        """
        if df is None or df.empty:
            return "_No data in the selected window._"

        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].astype(float).round(self.round_digits)

        if len(d) > self.max_rows:
            step = max(1, len(d) // self.max_rows)
            d = d.iloc[::step]

        parts = []
        parts.append(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        parts.append(d.reset_index(names="timestamp").to_markdown(index=False))
        parts.append("\n**Summary Stats:**")
        parts.append(df.describe().round(3).to_markdown())

        return "\n\n".join(parts)

    def __call__(self, df: pd.DataFrame) -> str:
        """Allow instance to be called like a function."""
        return self.packet(df)
