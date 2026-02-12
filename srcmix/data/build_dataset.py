import pandas as pd

def load_excel(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    return df