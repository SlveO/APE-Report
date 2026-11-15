"""Local-only campaign UI with five-second updates and browser render receipt."""
import argparse
import json
import mimetypes
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .common import append_json, inside, now, read_json, write_json

HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>APE · MAIN80</title>
<style>
:root{color-scheme:light;--ink:#171a20;--muted:#535b66;--line:#dce0e5;--accent:#155db3;--bad:#a31926;--good:#146340}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#fff;font:14px/1.5 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:28px 24px}header{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid var(--ink);gap:24px}
h1{font-size:24px;line-height:1.2;font-weight:600;text-wrap:balance}h2{font-size:17px;font-weight:600;margin:30px 0 12px}p{margin:12px 0}
.meta,small{color:var(--muted)}.strip{display:flex;flex-wrap:wrap;gap:28px;padding:20px 0;border-bottom:1px solid var(--line)}.number{display:block;font-size:23px;line-height:1.3;font-variant-numeric:tabular-nums}
progress{width:100%;height:14px;accent-color:var(--accent)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;text-align:left}
th{font-weight:600;color:var(--muted)}td,th{padding:10px 12px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{padding-left:0}
.scroll{overflow-x:auto}.bad{color:var(--bad)}.good{color:var(--good)}a{color:var(--accent);text-underline-offset:3px}a:focus-visible{outline:2px solid var(--accent);outline-offset:4px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f8;padding:16px;font-size:12px}.job{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}
@media(max-width:600px){main{padding:18px 16px}header{display:block}.strip{gap:16px}.number{font-size:20px}td,th{white-space:nowrap}}
</style><main><header><h1>APE / <span id="campaign">加载中</span></h1><p class="meta" id="updated">每5秒更新</p></header>
<p id="state" role="status" aria-live="polite">正在读取实际运行状态…</p>
<div class="strip"><div>成功 / 计划<strong class="number" id="counts">—</strong></div><div>失败 attempts<strong class="number" id="failed">—</strong></div><div>已用 / 剩余预算<strong class="number" id="budget">—</strong></div><div>预计完成<strong class="number" id="eta">暂不可可靠估计</strong></div></div>
<p class="meta">连续80小时预算</p><progress id="time" max="288000" value="0" aria-label="已用墙钟预算"></progress>
<h2>当前任务</h2><p class="job" id="job">—</p>
<h2>实际速度</h2><div class="scroll"><table><thead><tr><th>方法</th><th>缓存</th><th>文档长度</th><th>有效观测</th><th>秒 / 回答</th><th>观测范围</th></tr></thead><tbody id="rates"></tbody></table></div>
<h2>记录与交付</h2><p><a href="artifacts/attempts.jsonl">逐题原始记录</a> · <a href="artifacts/events.jsonl">事件与变更</a> · <a href="artifacts/sampling.json">当前样本ID</a> · <a href="artifacts/summary.json">汇总</a> · <a href="artifacts/incomplete.json">未完成清单</a></p>
<h2>抽样变更与错误</h2><pre id="events">暂无</pre><p class="meta">GPU0 · BF16 · seed42 · greedy。模型加载单列；搜索缓存计时不作冷启动加速结论。</p></main>
<script>
const $=id=>document.getElementById(id);const hours=s=>(s/3600).toFixed(2)+'h';
async function update(){try{const r=await fetch('status.json',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();
$('campaign').textContent=d.campaign;$('updated').textContent='更新 '+new Date(d.updated_at).toLocaleTimeString();$('state').textContent=d.state+' · '+(d.stage||'准备');$('state').className=d.state==='RUNNING'?'good':d.state==='BLOCKED'?'bad':'';
$('counts').textContent=d.completed+' / '+d.planned;$('failed').textContent=d.failed;$('budget').textContent=hours(d.elapsed_seconds)+' / '+hours(d.remaining_seconds);$('eta').textContent=d.eta_range?d.eta_range.map(x=>new Date(x).toLocaleString()).join(' — '):'暂不可可靠估计';$('time').value=d.elapsed_seconds;
$('job').textContent=JSON.stringify(d.current_job||{},null,2);$('rates').replaceChildren();for(const x of d.rates||[]){let tr=document.createElement('tr');for(const v of [x.method,x.cache_mode,x.length_bucket,x.count,x.mean.toFixed(2),x.low.toFixed(2)+'–'+x.high.toFixed(2)]){let td=document.createElement('td');td.textContent=v;tr.append(td)}$('rates').append(tr)}$('events').textContent=JSON.stringify(d.notices||[],null,2)
}catch(e){$('state').textContent='连接失败：'+e.message+'，5秒后重试';$('state').className='bad'}}
update();setInterval(update,5000);fetch('seen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rendered:true,title:document.title,width:innerWidth})});
</script></html>'''


def current_status(directory):
    status = read_json(directory / 'status.json')
    if status.get('state') == 'RUNNING':
        age = time.time() - datetime.fromisoformat(status['updated_at']).timestamp()
        if age > 60:
            status['state'] = 'STALE'
            status['notices'] = status.get('notices', []) + [{'event': 'controller_heartbeat_stale', 'seconds': age}]
    if (directory / 'state.json').exists():
        state = read_json(directory / 'state.json')
        if 'started_epoch' in state:
            status['elapsed_seconds'] = max(0, time.time() - state['started_epoch'])
            status['remaining_seconds'] = max(0, state['deadline_epoch'] - time.time())
    return status


def serve(directory, campaign, port=0):
    directory = inside(directory)
    prefix = f'/campaign/{campaign}/'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, content, kind, status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == prefix:
                self.send(HTML.encode(), 'text/html; charset=utf-8')
            elif route == prefix + 'status.json':
                self.send(json.dumps(current_status(directory)).encode(), 'application/json')
            elif route.startswith(prefix + 'artifacts/'):
                name = route.removeprefix(prefix + 'artifacts/')
                allowed = {'attempts.jsonl', 'events.jsonl', 'sampling.json', 'summary.json', 'incomplete.json'}
                if name in allowed and (directory / name).exists():
                    self.send((directory / name).read_bytes(), 'application/json')
                else:
                    self.send(b'Artifact not yet available', 'text/plain', 404)
            else:
                self.send(b'Not found', 'text/plain', 404)

        def do_POST(self):
            if urlparse(self.path).path != prefix + 'seen':
                self.send(b'Not found', 'text/plain', 404)
                return
            n = int(self.headers.get('Content-Length', '0'))
            if not 0 < n < 4096:
                self.send(b'Invalid receipt', 'text/plain', 400)
                return
            receipt = json.loads(self.rfile.read(n))
            append_json(directory / 'browser-seen.jsonl', {'at': now(), 'campaign': campaign,
                'user_agent': self.headers.get('User-Agent'), **receipt})
            self.send(b'ok', 'text/plain')
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    write_json(directory / 'monitor.json', {'pid': __import__('os').getpid(), 'url': f'http://127.0.0.1:{server.server_port}{prefix}', 'started_at': now()})
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    serve(args.directory, args.campaign)


if __name__ == '__main__':
    main()
