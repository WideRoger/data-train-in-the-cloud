
from google.cloud import bigquery

import pandas as pd

from colorama import Fore, Style

from taxifare.ml_logic.params import PROJECT, DATASET


def get_bq_chunk(table: str,
                 index: int,
                 chunk_size: int,
                 dtypes: dict = None,
                 verbose=True) -> pd.DataFrame:
    """
    return a chunk of a bigquery dataset table
    format the output dataframe according to the provided data types
    """
    if verbose:
        print(Fore.MAGENTA + f"Source data from bigquery {table}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})" + Style.RESET_ALL)

    table = f"{PROJECT}.{DATASET}.{table}"

    client = bigquery.Client()

    rows = client.list_rows(table, start_index=index, max_results=chunk_size)

    big_query_df = rows.to_dataframe()

    if big_query_df.shape[0] == 0:
        return None  # end of data

    big_query_df = big_query_df.astype(dtypes)

    return big_query_df


def save_bq_chunk(table: str, data: pd.DataFrame, is_first: bool):
    """
    save a chunk of the raw dataset to bigquery
    empty the table beforehands if `is_first` is True
    """

    print(Fore.BLUE + f"\nSave data to bigquery {table}:" + Style.RESET_ALL)

    table = f"{PROJECT}.{DATASET}.{table}"

    # bq requires str columns starting with a letter or underscore
    data.columns = [
        f"_{column}" if type(column) != str else column for column in data.columns
    ]

    client = bigquery.Client()

    # define write mode and schema
    write_mode = "WRITE_TRUNCATE" if is_first else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(
        Fore.YELLOW
        + f"\n{'Write' if is_first else 'Append'} {table} ({data.shape[0]} rows)"
        + Style.RESET_ALL
    )

    # load data
    job = client.load_table_from_dataframe(data, table, job_config=job_config)

    job.result()  # wait for the job to complete
