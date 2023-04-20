
import pandas as pd

def collectAllData_v2(path):
    df = pd.read_csv(path)
    #print(df)
    df = df.set_index('Filename')
    return df