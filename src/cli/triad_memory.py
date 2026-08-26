import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from runtime.memory.triad_memory import TriadMemory, memory_dir


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('start')
    p.add_argument('--T', type=float, default=10.0)
    p.add_argument('--N', type=int, default=64)
    p = sub.add_parser('record')
    p.add_argument('text')
    p.add_argument('--tags', default='')
    p = sub.add_parser('recall')
    p.add_argument('query')
    p.add_argument('-n', type=int, default=None)
    p = sub.add_parser('resonate')
    p.add_argument('query')
    p.add_argument('-n', type=int, default=None)
    p.add_argument('--T', type=float, default=None)
    p.add_argument('--force', type=float, default=None)
    p = sub.add_parser('sleep')
    p.add_argument('--T', type=float, default=None)
    sub.add_parser('status')
    a = ap.parse_args()

    m = TriadMemory(memory_dir(), N=getattr(a, 'N', 64))
    if a.cmd == 'start':
        print(json.dumps(m.start(a.T), ensure_ascii=False))
    elif a.cmd == 'record':
        tags = [t for t in a.tags.split(',') if t]
        print(json.dumps({'id': m.record(a.text, tags)},
                         ensure_ascii=False))
    elif a.cmd == 'recall':
        for item in m.recall(a.query, a.n):
            print(json.dumps(item, ensure_ascii=False))
    elif a.cmd == 'resonate':
        for item in m.resonate(a.query, a.n, T=a.T, force=a.force):
            print(json.dumps(item, ensure_ascii=False))
    elif a.cmd == 'sleep':
        print(json.dumps(m.sleep(a.T), ensure_ascii=False))
    elif a.cmd == 'status':
        print(json.dumps(m.status(), ensure_ascii=False))

if __name__ == '__main__':
    main()
