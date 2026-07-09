def simple_func_to_test(s: str = "") -> bool:
    if s.startswith("a") and s.endswith("B"):
        return True
    else:
        return False