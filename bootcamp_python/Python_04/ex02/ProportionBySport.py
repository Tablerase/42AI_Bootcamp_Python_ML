import pandas as pd
import matplotlib.pyplot as plt
from Python_04.ex00.FileLoader import FileLoader

# https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html
# https://pandas.pydata.org/docs/user_guide/indexing.html#duplicate-data


def proportion_by_sport(df: pd.DataFrame, olympic_year: int, sport: str, gender: str):
    filtered_athletes = df[(df['Year'] == olympic_year)
                           & (df['Sex'] == gender)].drop_duplicates('Name')
    filtered_by_sport = filtered_athletes.loc[filtered_athletes['Sport'] == sport]
    total_athletes = filtered_athletes.shape[0]
    total_by_sport = filtered_by_sport.shape[0]
    return total_by_sport / total_athletes


if __name__ == "__main__":
    loader = FileLoader()
    df = loader.load('../athlete_events.csv')
    proportion_for_tennis = proportion_by_sport(df, 2004, 'Tennis', 'F')
    print(proportion_for_tennis)
