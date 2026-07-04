import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

_ZONED_RE = re.compile(r'^(?P<iso>[^[]+)\[(?P<zone>[^\]]+)]$')
log = logging.getLogger(__name__)


def convert_values_2d(values):
    val_2d = []
    for val in values:
        tmp_array = [val]
        val_2d.append(tmp_array)
    return val_2d

def filter_data(time_stamps, train_values, features, resolution_min: int):
    ind = 0
    resolution_sec = resolution_min * 60
    for i in range(1, len(time_stamps)):
        time = parse_zoned_dt(time_stamps[i])
        time_prev = parse_zoned_dt(time_stamps[i - 1])
        diff_sec = (time - time_prev).total_seconds()
        if diff_sec != resolution_sec:
            ind = i
    log.info('Start of continuous data chunk: ' + time_stamps[ind])
    log.info('Data set size is ' + str(len(train_values) - ind))
    time_stamps_cont = []
    train_values_cont = []
    features_cont = []
    for i in range(ind, len(time_stamps)):
        time_stamps_cont.append(time_stamps[i])
        train_values_cont.append(train_values[i])
        features_cont.append(features[i])
    return time_stamps_cont, train_values_cont, features_cont

def is_valid(time_stamps: list, resolution_min: int)->bool:
    resolution_sec = resolution_min * 60
    for i in range(1, len(time_stamps)):
        time = parse_zoned_dt(time_stamps[i])
        time_prev = parse_zoned_dt(time_stamps[i - 1])
        diff_sec = (time - time_prev).total_seconds()
        if diff_sec != resolution_sec:
            return False
    return True

def parse_zoned_dt(s: str) -> datetime:
    """
    Parse a Java‑style ZonedDateTime string into a timezone‑aware datetime.
    Example input: '2024-01-01T00:00+02:00[Asia/Jerusalem]'
    """
    m = _ZONED_RE.match(s)
    if not m:
        raise ValueError(f"Not a ZonedDateTime string: {s!r}")

    iso_part = m.group('iso')     # '2024-01-01T00:00+02:00'
    zone_id  = m.group('zone')    # 'Asia/Jerusalem'

    # 1. Parse the ISO part – this already contains the fixed +02:00 offset
    dt = datetime.fromisoformat(iso_part)

    # 2. Re‑anchor the datetime to the real zone ID
    #    This keeps the *instant* the same but swaps the tzinfo so
    #    later arithmetic uses correct DST rules.
    return dt.astimezone(ZoneInfo(zone_id))