import pandas as pd
import matplotlib.pyplot as plt
from Python_04.ex00.FileLoader import FileLoader

# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html#how-do-i-filter-specific-rows-from-a-dataframe


def youngest_fellah(df: pd.DataFrame, olympic_year: int):
    male_athletes = df[(df['Sex'] == 'M') & (df['Year'] == olympic_year)]
    male_min = male_athletes['Age'].min()
    male_names = male_athletes.loc[male_athletes['Age']
                                   == male_min, 'Name'].tolist()
    male = {
        'Age': male_min,
        'Names': male_names
    }
    female_athletes = df[(df['Sex'] == 'F') & (df['Year'] == olympic_year)]
    female_min = female_athletes['Age'].min()
    female_names = female_athletes.loc[female_athletes['Age']
                                       == female_min, 'Name'].tolist()
    female = {
        'Age': female_min,
        'Names': female_names
    }
    youngest = {'F': female, 'M': male}
    return youngest


if __name__ == "__main__":
    loader = FileLoader()
    df = loader.load("../athlete_events.csv")
    youngest_athletes = youngest_fellah(df, 2004)
    print(youngest_athletes)
