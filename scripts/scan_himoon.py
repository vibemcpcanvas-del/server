# -*- coding: utf-8 -*-
"""himoon(ip24 스트리밍 클라이언트) 창 핸들 스캐너."""
import sys
sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")
import win32gui

def scan():
    wins = []
    def cb(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h)
            c = win32gui.GetClassName(h)
            wins.append((h, c, t))
    win32gui.EnumWindows(cb, None)
    # himoon 관련 우선
    hits = [w for w in wins if any(k in (w[1] + w[2]).lower()
            for k in ("himoon", "moonlight", "hiip", "stream"))]
    print("=== himoon/stream 관련 창 ===")
    for h, c, t in hits:
        print(f"  {h} | {c} | {t[:50]}")
    if not hits:
        print("  (없음 — 전체 목록)")
        for h, c, t in wins:
            if t.strip():
                print(f"  {h} | {c} | {t[:50]}")
    return hits

if __name__ == "__main__":
    scan()
