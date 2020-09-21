
def str2int(value, default):
    try:
        return int(value)
    except Exception as e:
        return default


def str2float(value, default):
    try:
        return float(value)
    except Exception as e:
        return default


def str2bool(value, default):
    try:
        return bool(value)
    except Exception as e:
        return default

