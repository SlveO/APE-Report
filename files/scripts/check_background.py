"""Finite detached-process, local HTTP, real-browser and refresh verification."""
import base64
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket
from main80.common import ROOT, inside, now, read_json, read_records, write_json
from main80.campaign import launch_monitor, stop_own_worker


class DevTools:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=8, suppress_origin=True,
            http_no_proxy=['127.0.0.1', 'localhost'])
        self.serial = 0

    def call(self, method, params=None):
        self.serial += 1
        self.socket.send(json.dumps({'id': self.serial, 'method': method, 'params': params or {}}))
        while True:
            result = json.loads(self.socket.recv())
            if result.get('id') == self.serial:
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result.get('result')

    def evaluate(self, expression):
        return self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True})['result']['value']


def json_url(url):
    with urllib.request.urlopen(url, timeout=4) as response:
        return json.load(response)


def main():
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    directory = inside(f'logs/runs/20261113-main80-prep/background-{stamp}')
    directory.mkdir(parents=True, exist_ok=False)
    campaign = directory.name
    initial = {'campaign': campaign, 'state': 'PREP_TEST', 'stage': 'detached_fixture', 'updated_at': now(),
        'completed': 0, 'planned': 12, 'failed': 0, 'elapsed_seconds': 0, 'remaining_seconds': 288000,
        'eta_range': None, 'rates': [{'method': 'fixture', 'cache_mode': 'cold', 'length_bucket': 4096, 'count': 3, 'mean': 1.0, 'low': 1.0, 'high': 1.0}],
        'current_job': {'scope': 'synthetic UI fixture; no model'}, 'notices': []}
    write_json(directory / 'status.json', initial)
    url = launch_monitor(directory)
    monitor_pid = read_json(directory / 'monitor.json')['pid']
    def stop_monitor():
        try:
            os.kill(monitor_pid, __import__('signal').SIGTERM)
        except ProcessLookupError:
            pass
    atexit.register(stop_monitor)
    # The short-lived launcher exits while its detached child keeps updating status.
    child = 'import time; from main80.common import read_json,write_json,now; from pathlib import Path; p=Path(' + repr(str(directory / 'status.json')) + ');\nfor i in range(1,13):\n d=read_json(p);d.update(completed=i,updated_at=now());write_json(p,d);time.sleep(1)\n'
    launcher = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-B","-c",' + repr(child) + '],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(p.pid)'
    process = subprocess.run([sys.executable, '-B', '-c', launcher], capture_output=True, text=True, check=True)
    worker_pid = int(process.stdout.strip())
    write_json(directory / 'launch-proof.json', {'launcher_exit_code': process.returncode, 'detached_child_pid': worker_pid, 'monitor_url': url, 'scope': 'PREP fixture only'})
    chrome = shutil.which('google-chrome')
    if chrome is None:
        raise RuntimeError('installed Chrome required for DevTools visual verification')
    profile = inside(f'.runtime/browser/{campaign}')
    profile.mkdir(parents=True, exist_ok=True)
    command = [chrome, f'--user-data-dir={profile}', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
        '--disable-gpu', '--disable-dev-shm-usage', '--disable-background-networking', '--disable-component-update',
        '--disable-sync', '--metrics-recording-only', '--no-first-run', '--no-default-browser-check', '--new-window', url]
    with (directory / 'browser.stdout').open('wb') as out, (directory / 'browser.stderr').open('wb') as err:
        browser = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
    atexit.register(lambda: stop_own_worker(browser))
    write_json(directory / 'browser-command.json', {'command': command, 'pid': browser.pid, 'at': now()})
    end = time.monotonic() + 20
    target = None
    while time.monotonic() < end:
        active = profile / 'DevToolsActivePort'
        if active.exists():
            port = int(active.read_text().splitlines()[0])
            try:
                target = next((t for t in json_url(f'http://127.0.0.1:{port}/json/list') if t.get('url') == url), None)
                if target:
                    break
            except OSError:
                pass
        if browser.poll() is not None:
            raise RuntimeError(f'isolated browser exited {browser.returncode}; inspect saved stderr')
        time.sleep(0.2)
    if not target:
        raise RuntimeError('browser did not expose its own campaign page')
    client = DevTools(target['webSocketDebuggerUrl'])
    end = time.monotonic() + 10
    while time.monotonic() < end:
        loaded = client.evaluate('JSON.stringify({campaign:document.querySelector("#campaign")?.textContent,counts:document.querySelector("#counts")?.textContent,width:innerWidth})')
        loaded = json.loads(loaded)
        if loaded.get('campaign') == campaign:
            break
        time.sleep(0.2)
    assert loaded.get('campaign') == campaign
    first = loaded['counts']
    time.sleep(5.5)
    later = json.loads(client.evaluate('JSON.stringify({campaign:document.querySelector("#campaign").textContent,counts:document.querySelector("#counts").textContent})'))
    assert first != later['counts'], 'HTML did not refresh after detached launcher exit'
    for width in (1100, 420):
        client.call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': width < 600})
        shot = client.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
        (directory / f'monitor-{width}.png').write_bytes(base64.b64decode(shot['data']))
    receipts, _ = read_records(directory / 'browser-seen.jsonl')
    assert any(r.get('rendered') and 'Chrome/' in r.get('user_agent', '') for r in receipts)
    client.call('Browser.close')
    browser.wait(timeout=10)
    # Real watchdog signal to an owned, otherwise idle process. No model/GPU.
    import threading
    sleeper = subprocess.Popen([sys.executable, '-B', '-c', 'import time;time.sleep(20)'], start_new_session=True)
    timer = threading.Timer(0.4, lambda: stop_own_worker(sleeper))
    timer.start()
    time.sleep(0.8)  # Simulate the controller being busy outside its polling loop.
    timer.join(timeout=5)
    assert sleeper.poll() is not None
    result = {'role': 'PLANNER', 'status': 'PASS', 'at': now(), 'monitor_url': url, 'directory': str(directory),
        'launcher_exited_while_child_continued': True, 'actual_browser_rendered': True,
        'five_second_HTML_refresh': True, 'counts_before': first, 'counts_after': later['counts'],
        'independent_deadline_watchdog': True, 'screenshots': ['monitor-1100.png', 'monitor-420.png'],
        'formal_campaign_started': False, 'no_model_or_GPU_execution': True}
    write_json(directory / 'result.json', result)
    write_json('logs/runs/20261113-main80-prep/background-test-latest.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
