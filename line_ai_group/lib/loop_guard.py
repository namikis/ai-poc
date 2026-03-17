from collections import OrderedDict
from threading import Lock

MAX_EVENT_IDS = 512

_seen_event_ids: OrderedDict[str, None] = OrderedDict()
_seen_lock = Lock()


def is_duplicate_event(event_id: str | None) -> bool:
    if not event_id:
        return False

    with _seen_lock:
        if event_id in _seen_event_ids:
            return True

        _seen_event_ids[event_id] = None
        _seen_event_ids.move_to_end(event_id)

        while len(_seen_event_ids) > MAX_EVENT_IDS:
            _seen_event_ids.popitem(last=False)

    return False
