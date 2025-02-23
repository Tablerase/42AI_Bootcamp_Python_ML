import sys
import pandas as pd
import matplotlib.pyplot as plt


class FileLoader:
    def __init__(self):
        pass

    def load(self, path: str) -> pd.DataFrame:
        try:
            data_frame = pd.read_csv(filepath_or_buffer=path)
            shape = data_frame.shape
            print(f"Loading dataset of dimensions {shape[0]} x {shape[1]}")
        except Exception as e:
            print(e, file=sys.stderr)
            return None
        return data_frame

    def display(self, df: pd.DataFrame, n: int):
        print(df.head(n))


if __name__ == "__main__":
    loader = FileLoader()
    df = loader.load("../athlete_events.csv")
    loader.display(df, 10)

    # ______________________________ Test ______________________________ #
    # print(f"{'Default display':_^60}")
    # print(df)
    # print(f"{'Head display':_^60}")
    # print(df.head(3))
    # print(f"{'Tail display':_^60}")
    # print(df.tail(3))
    # print(f"{'Details display':_^60}")
    # print(df.info())
    # print(f"{'Columns by key':_^60}")
    # print(df[['Name', 'Age']])
    # print(f"{'Filter rows':_^60}")
    # age_35plus = df["Age"] > 35
    # print(df[age_35plus])
    # print(f"{'Dataframe shape':_^60}")
    # print(df.shape)
    # print(f"{'Selection with loc':_^60}")
    # atheles_names = df.loc[df["Age"] > 75, ["Name", "Age"]]
    # print(atheles_names)
    # print(f"{'Selection with iloc':_^60}")
    # future_johns = df.iloc[0:5, 0:12]
    # print(future_johns)
    # future_johns.iloc[0:3, 1] = 'John Doe'
    # print(future_johns)
    # print(f"{'ploting with pandas':_^60}")
    # senior42_athelete = df.loc[lambda df: df['Age']
    #                            == 42, ['Age', 'Height', 'Weight']]
    # print(senior42_athelete)
    # senior42_athelete.plot.scatter(x='Height', y='Weight', alpha=0.5)
    # plt.show()
