import pandas as pd
import numpy as np

def removeDup(df):
    """
    Remove duplicated rows
    Args:
        df = data frame
    """
    dup_records = df.duplicated().value_counts()
    print("Number of duplicated rows: \n")
    print(dup_records)
    df_redup = df.drop_duplicates()
    nrows_duplicates = df_redup.duplicated().sum()
    print("After removing duplicated rows: ", nrows_duplicates, " of duplicated rows")
    
    return df_redup


def findMiss(df):
    """
    Return the missing rate of each variable
    Args:
        df = data frame
    """
    return round(df.isnull().sum()/df.shape[0]*100, 2)