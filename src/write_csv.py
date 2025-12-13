
import os
import pandas as pd

def write_csv(dataframe: pd.DataFrame, directory: str, filename: str, index: bool = False):
    """
    Save a Pandas DataFrame to a CSV file in the specified directory.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame to save.
    directory : str
        The directory where the file will be saved.
    filename : str
        The name of the file (must include the '.csv' extension).
    index : bool, optional
        Whether to include the DataFrame's index in the CSV file. Default is False.

    Raises
    ------
    ValueError
        If the filename does not end with '.csv', or the DataFrame is empty.
    FileNotFoundError
        If the specified directory does not exist.
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not filename.endswith(".csv"):
        raise ValueError("Filename must end with '.csv'")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if dataframe.empty:
        raise ValueError("DataFrame must contain observations.")

    filepath = os.path.join(directory, filename)
    dataframe.to_csv(filepath, index=index)
