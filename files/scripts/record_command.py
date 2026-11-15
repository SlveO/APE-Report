"""Record an authorized preparation command verbatim, including failures."""
import argparse
import subprocess
import sys

from main80.common import ROOT, inside, now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required after --')
    directory = inside(args.directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = {'command': command, 'cwd': str(ROOT), 'started_at': now(), 'status': 'RUNNING'}
    write_json(directory / 'command.json', record)
    with (directory / 'stdout').open('xb') as out, (directory / 'stderr').open('xb') as err:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, timeout=args.timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    record.update(ended_at=now(), exit_code=code, status='SUBMITTED')
    write_json(directory / 'command.json', record)
    print(f'{directory.relative_to(ROOT)}: exit {code}')
    if code:
        print((directory / 'stderr').read_text()[-5000:])
    else:
        print((directory / 'stdout').read_text()[-1500:])
    raise SystemExit(code)


if __name__ == '__main__':
    main()
