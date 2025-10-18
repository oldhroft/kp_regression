import logging
import typing as T

from numpy import ndarray
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split  # type: ignore

from kp_regression.data_pipe import Dataset


def check_data_and_get_train_val_plain_input(
    ds: Dataset,
    ds_val: T.Optional[Dataset],
    val_frac: T.Optional[float],
    error_if_both_absent: bool = False,
) -> T.Tuple[NDArray, NDArray, T.Optional[NDArray], T.Optional[NDArray]]:
    """
    Extracts training and validation data arrays from Dataset objects, handling validation splits and type checks.

    Notes:
        - The function asserts that ds.X and ds.y are numpy arrays.
        - If ds_val is provided, it also asserts that ds_val.X and ds_val.y are numpy arrays.
        - If both ds_val and val_frac are None and error_if_both_absent is True, a ValueError is raised.
    """

    assert isinstance(ds.X, ndarray), "For MLP dataset X should be Numpy"
    assert isinstance(ds.y, ndarray), "For MLP dataset y should be Numpy"

    X, y = ds.X, ds.y

    if ds_val is None and val_frac is not None:
        logging.info("Received val frac, creating val split")

        split_result = tuple(
            train_test_split(
                X,
                y,
                test_size=val_frac,
                random_state=17,
                shuffle=True,
            )
        )

        assert (
            len(split_result) == 4
        ), "Result of train-test split should contain exactly 4 items"

        split_result = T.cast(T.Tuple[NDArray, ...], split_result)

        X, X_val, y, y_val = split_result

    elif ds_val is not None:
        assert isinstance(ds_val.X, ndarray), "Dataset X should be Numpy"
        assert isinstance(ds_val.y, ndarray), "Dataset y should be Numpy"

        X_val, y_val = ds_val.X, ds_val.y

    else:
        if error_if_both_absent:
            raise ValueError(
                "Both ds_val and val_frac are None, but validation data is required."
            )
        X_val, y_val = None, None

    return X, y, X_val, y_val
