import pandas as pd


def process_df(df: pd.DataFrame) -> bool | None:
    """Applies different operations to a DataFrame, in place."""
    df.drop(columns=["col1"], inplace=True)  # Requires col1 to exist
    df.dropna(inplace=True)
    df = df.iloc[:20]  # Does not do anything as df is not the same as df1 anymore
    if not df.empty:
        return True


# if __name__ == "__main__":
#     col1 = pd.Series([1, 2, 3, 4])
#     col2 = pd.Series([1, 2, 3])
#     df1 = pd.DataFrame({"col1": col1, "col2": col2})

#     print(df1)
#     process_df(df1)
#     print(df1)
