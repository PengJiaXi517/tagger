def get_dict_data(data, keys):
    """Fetch value in `data` through `keys` recursively."""
    assert isinstance(data, dict)
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    assert len(keys) > 0

    if len(keys) == 1:
        return data[keys[0]]
    else:
        return get_dict_data(data[keys[0]], keys[1:])


def write_dict_data(data, keys, value):
    """Write value into `data` through `keys` recursively."""
    assert isinstance(data, dict)
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    assert len(keys) > 0

    if len(keys) == 1:
        data[keys[0]] = value
    else:
        if keys[0] not in data:
            data[keys[0]] = dict()
        write_dict_data(data[keys[0]], keys[1:], value)
