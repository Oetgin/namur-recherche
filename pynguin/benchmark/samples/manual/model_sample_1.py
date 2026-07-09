import pandas as pd  # noqa: D100


def process_df(df: pd.DataFrame) -> bool:
    """Applies different operations to a DataFrame, in place."""
    df.drop(columns=["col1"], inplace=True, errors="ignore")
    df.dropna(inplace=True)
    return not df.empty
