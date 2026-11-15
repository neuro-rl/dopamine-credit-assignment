import traceback

import nest


_tracked_recorders = []
_listeners_enabled = False
_capture_stack = True


def plugin_recorder_listeners(capture_stack: bool = True):
    global _listeners_enabled, _capture_stack
    _listeners_enabled = True
    _capture_stack = capture_stack


def unplugin_recorder_listeners():
    global _listeners_enabled
    _listeners_enabled = False


def create(*args, **kwargs):
    nodes = nest.Create("spike_recorder", *args, **kwargs)
    if _listeners_enabled:
        _tracked_recorders.append({
            "nodes": nodes,
            "stack": traceback.format_stack(limit=8) if _capture_stack else None,
        })
    return nodes


def clear_tracked_recorders():
    _tracked_recorders.clear()


def get_tracked_recorders():
    return [item["nodes"] for item in _tracked_recorders]


def print_max_events_among_recorders(prefix: str = "[Recorder tracking]"):
    if not _tracked_recorders:
        print(f"{prefix} recorders=0 max_events=0")
        return 0

    max_events = 0
    max_index = -1
    live_count = 0

    for index, item in enumerate(_tracked_recorders):
        try:
            n_events = int(nest.GetStatus(item["nodes"], "n_events")[0])
        except Exception:
            continue

        live_count += 1
        if n_events > max_events:
            max_events = n_events
            max_index = index

    print(
        f"{prefix} recorders={len(_tracked_recorders)} "
        f"live_recorders={live_count} max_events={max_events} max_index={max_index}"
    )
    return max_events
