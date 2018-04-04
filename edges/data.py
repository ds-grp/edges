import numpy as np
import pandas as pd


def clean_fig1(df):
    # Quickly rename the columns for the Figure 1 data to
    # ['freq', 'weight', 'tsky', 'tres1', 'tres2', 'tmodel', 't21']

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


def fetch_fig2():
    try:
        df = pd.read_csv('http://loco.lab.asu.edu/download/792/')
    except URLError:
        df = pd.read_csv('data/raw/figure2_plotdata.csv')

    return df


def fetch_fig1():
    try:
        df = pd.read_csv('http://loco.lab.asu.edu/download/790/')
    except URLError:
        df = pd.read_csv('data/raw/figure1_plotdata.csv')

    df = clean_fig1(df)

    return df

def fetch_processed():
    df = pd.read_csv('data/processed/bowman_2018_fg_and_t21.csv')

    return df
