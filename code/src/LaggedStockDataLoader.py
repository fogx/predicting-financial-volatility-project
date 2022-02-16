
#Create the LaggedStockDataLaoder

import pandas as pd
from pathlib import Path

path_root = Path().absolute().parent.resolve()

def lag_var(df: pd.DataFrame, columns: list, lag: int = 1) -> pd.DataFrame:
    """
    lag the Series `lag` times
    df: dataframe to create lags in
    columns: list of columns, where this should be applied to
    !warning! drops na columns, which shortens df
    """
    for col in columns:
        df = pd.concat([df] + [df[col].shift(lag).rename(f'{df[col].name}_{lag + 1}') for lag in range(lag)],
                       axis=1)
    return df.dropna()


def filter_log_vals_from_df(df):
    # filter log values so they aren't duplicated in the data
    drop_list = [col for col in df.loc[:, df.columns.str.contains("log")]]
    df.drop(drop_list, axis=1, inplace=True)
    return df

def simple_split(df):
    """
    manual split: train: 2010-2017, validation: 2018, test: 2019-2020 (because of corona)
    (73/9/18 or 8y/1y/2y)
    """
    splits = {
        "train": slice("2010", "2016"),
        "val": slice("2017", "2018"),
        "test": slice("2019", None)
    }
    return splits

class LaggedStockDataLoader:
    """
    simplified version of the Data Loader for jupyter notebooks -> function calls got flattened
    This data_loader converts the raw dataset into a train-val-test split, by first cleaning the data
    dropping irrelevant columns and empty rows and then lagging necessary parameters.
    """

    def __init__(self, path_data=None, drop_na=True, split_fn=simple_split, load=False):
        self.df = None
        self.y_columns = ["outcome_rv", "outcome_rv_log"]
        try:
            self.splits = split_fn(self.df)
        except Exception as e:
            raise Exception("Split not implemented") from e

        if path_data:
            self.path_data = path_data
        else:
            self.path_data = path_root / "Abgabeordner" / "BAC.csv"
        if load:
            # load the data from a hdf5 instead of initializing it from a csv
            self.load()
        else:
            # try to load a csv and process it
            self.from_csv(drop_na, split_fn)

    def from_csv(self, drop_na, split_fn):
        self.df = pd.read_csv(self.path_data, parse_dates=["DT"])
        # data cleanup - regularize lag_rv_day to lag_rv_day_1
        self.df.rename(columns={"lag_rv_day": "lag_rv_day_1"}, inplace=True)
        # data cleanup - drop rf because the data is broken and 3month, because it has too many Na values
        self.df.drop(["rf", "lag_rv_3month"], axis=1, inplace=True)
        # set index to DT to get datetimeindexes (required for simple splits) and remove the variable from the table
        self.df.set_index("DT", inplace=True)
        # drop na values
        self.df = filter_log_vals_from_df(self.df)
        if drop_na:
            self.df = self.df.dropna()
        # lag relevant parameters
        self.df = lag_var(self.df, ["lag_rv_week", "lag_rv_month"], 21)

    def _get_x(self, df):
        """
        return passed df with only X columns
        """
        return df[df.columns.difference(self.y_columns)]

    def _get_y(self, df):
        """
        return passed df with only Y columns
        """
        return df[[x for x in self.y_columns if x in df]]

    def _get_data(self, data_name):
        """
        :param data_name: name of split to get "train", "val" or "test"
        :return Tuple: the dataset as a X,Y tuple
        """
        return self._get_x(self.df[self.splits[data_name]]), self._get_y(self.df[self.splits[data_name]])

    def get_train_data(self):
        return self._get_data("train")

    def get_validation_data(self):
        return self._get_data("val")

    def get_test_data(self):
        return self._get_data("test")

    def get_labels(self):
        return [x for x in self.df.columns]

    def get_df(self):
        return self.df

    def save(self, path_save):
        """
        save the data split as hdf5
        :param Path path_save: where to save the hdf5 (along with the file name
        :return Bool success
        """
        hdf = path_save
        # since we will be using lots of jupyter notebooks, it will simplify our work to redundantly save the datasplits, instead of importing the dataloader (which would be the better solution)
        x_train, y_train = self.get_train_data()
        x_val, y_val = self.get_validation_data()
        x_test, y_test = self.get_test_data()
        self.df.to_hdf(hdf, key="df")
        x_train.to_hdf(hdf, key="x_train")
        x_val.to_hdf(hdf, key="x_val")
        x_test.to_hdf(hdf, key="x_test")
        y_train.to_hdf(hdf, key="y_train")
        y_val.to_hdf(hdf, key="y_val")
        y_test.to_hdf(hdf, key="y_test")

    def load(self):
        """
        load the datasets from a hdf5 file
        :return Bool: success
        """
        with pd.HDFStore(self.path_data.as_posix()) as store:
            self.df = store["df"]