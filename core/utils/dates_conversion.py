

def ms_to_timeframe(ms):
    """
    Converts milliseconds to a human-readable timeframe string (e.g., "1d", "2h", "30m", "15s").

    Parameters
    ----------
    ms : int
        Time in milliseconds.

    Returns
    -------
    str
        A string representing the timeframe.
    """
    conversions = {
        "d": 60000 * 60 * 24,
        "h": 60000 * 60,
        "m": 60000
    }

    # Find the largest unit that fits into the given milliseconds
    for unit, duration in conversions.items():
        count = ms // duration
        if count > 0:
            return f"{count}{unit}"

    # If ms is less than a minute, return in seconds
    return f"{ms // 1000}s"

def timeframe_to_ms(t):
    """
    Converts a timeframe string (e.g., "1d", "2h", "30m") to milliseconds.

    Parameters
    ----------
    t : str
        A string representing the timeframe (e.g., "1d", "2h", "30m").

    Returns
    -------
    int
        The time in milliseconds.
    """
    conversions = {
        "m" : 60000,
        "h" : 60000 * 60,
        "d" : 60000 * 60 * 24
    }
    return int(t[:(-1)]) * conversions[t[-1]]