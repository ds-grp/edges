import numpy as np
import pandas as pd

from myhelpers import misc


def clean_fig1(df):
    # Quickly rename the columns for the Figure 1 data
    df.rename(columns={
        'Frequency [MHz]': 'freq',
        ' Weight': 'weight',
        ' a: Tsky [K]': 'tsky',
        ' b: Tres1 [K]': 'tres1',
        ' c: Tres2 [K]': 'tres2',
        ' d: Tmodel [K]': 'tmodel',
        ' e: T21 [K]': 't21'}, inplace=True)

    # Drop the first and last few rows
    # For some reason, they had been set to zero
    df.drop(df.head(6).index, inplace=True)
    df.drop(df.tail(6).index, inplace=True)

    return df


def fetch_fig1():
    df = pd.read_csv(misc.bpjoin('edges/data/raw/figure1_plotdata.csv'))
    df = clean_fig1(df)

    return df
