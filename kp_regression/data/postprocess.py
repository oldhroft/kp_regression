from numpy.typing import NDArray
from numpy import array, abs, take_along_axis, clip

INDICES = array(
    [
        0,
        3,
        7,
        10,
        13,
        17,
        20,
        23,
        27,
        30,
        33,
        37,
        40,
        43,
        47,
        50,
        53,
        57,
        60,
        63,
        67,
        70,
        73,
        77,
        80,
        83,
        87,
        90,
    ],
)


def attach_kp_index_to_grid(y: NDArray) -> NDArray:

    abs_diff_amin = (abs(y[..., None] - INDICES.reshape(1, 1, -1))).argmin(axis=2)

    return take_along_axis(
        INDICES.reshape(1, 1, -1), abs_diff_amin[..., None], axis=2
    ).squeeze()


def clip_kp(y: NDArray) -> NDArray:

    return clip(y, 0, 90)
