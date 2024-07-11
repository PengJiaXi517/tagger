import os

from registry import TAG_FUNCTIONS


@TAG_FUNCTIONS.register()
def demo(data, params, result):
    # print(1)
    return {"1": 1}
