from pynguin.large_language_model.parsing.rewriter import *
import pytest


def test_rewrite_import():
    code = """```python
import pandas as pd
import pytest

def test_process_df_missing_col():
    # Test case where 'col1' does not exist in the DataFrame
    df = pd.DataFrame({"col2": [1, 2, 3]})
    with pytest.raises(KeyError):
        process_df(df)
```"""
    rewrite = rewrite_tests(code)
    assert rewrite


if __name__ == "__main__":
    test_rewrite_import()
