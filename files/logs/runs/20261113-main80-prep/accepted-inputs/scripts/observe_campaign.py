"""Bounded launch observation. Ends within180s; never stops background workers."""
import argparse
import time

from main80.common import inside, now, read_json, write_json
from main80.monitor import current_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    if __import__('pathlib').Path(args.campaign).name != args.campaign:
        parser.error('campaign ID required')
    directory = inside('output/runs/' + args.campaign)
    started = time.monotonic()
    end = started + 179
    updates = []
    while time.monotonic() < end:
        if (directory / 'status.json').exists():
            status = current_status(directory)
            update = {k: status.get(k) for k in ('updated_at', 'state', 'stage', 'completed', 'planned', 'failed', 'eta_range', 'monitor_url')}
            if not updates or update != updates[-1]:
                updates.append(update)
                print(update, flush=True)
            if status['state'] in ('BLOCKED', 'SUBMITTED', 'STALE'):
                break
        time.sleep(min(5, max(0, end - time.monotonic())))
    write_json(directory / 'launch-observation.json', {'started_monotonic': started, 'ended_at': now(),
        'elapsed_seconds': time.monotonic() - started, 'hard_limit_seconds': 180, 'updates': updates,
        'background_workers_stopped_by_observer': False, 'ETA_policy': 'unknown unless all pending method/length/cache strata have adequate formal observations'})


if __name__ == '__main__':
    main()
