"""파이 HUD 상태 판정 측정 — 젯슨 담당 질문 ①②③ 대응.

같은 패킷을 실제 파이 StateTracker 에 넣어
  ① 패킷 state 가 바뀐 횟수 vs 디바운스 뒤 화면 색이 바뀐 횟수
  ② 노란색 상승 확인 프레임(rise_frames) 3 / 5 / 7 을 동시에 비교
  ③ 좌·우 선 번호가 바뀐 횟수 (선이 사라지거나 생긴 것과 번호만 바뀐 것을 분리)
를 잰다. HUD(receive) 는 끄고 돌릴 것 — 같은 5005 포트를 쓴다.

사용: python3 pi_state.py 40
"""
import json
import socket
import sys
import time
from collections import Counter
from pathlib import Path

from hud_system import PacketError, _decode_packet
from hud_state import StateTracker
from hud_theme import STATE_LOST, STATE_NORMAL, normalize_state

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
RISES = [3, 5, 7]          # 3 이 지금 설정
NAMES = {0: "녹", 1: "노", 2: "빨"}

base = json.loads(Path("hud_config.json").read_text(encoding="utf-8"))
fall = int(base.get("state", {}).get("fall_frames", 5))
trackers = {}
for r in RISES:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("state", {})["rise_frames"] = r
    trackers[r] = StateTracker(cfg)
timeout = float(base["network"]["packet_timeout_seconds"])

shown = {r: None for r in RISES}          # 직전 화면 state
shown_since = {r: None for r in RISES}
shown_changes = {r: Counter() for r in RISES}
short_shown = {r: Counter() for r in RISES}   # 0.5 초 안에 끝난 표시 구간

raw_prev = None
raw_run = 0
raw_runs = {0: Counter(), 1: Counter(), 2: Counter()}
raw_changes = Counter()

def bucket(n):
    return "1" if n == 1 else "2" if n == 2 else "3~4" if n <= 4 else "5+"

ids_prev = {"left": None, "right": None}
id_renum = Counter()       # 번호만 바뀜 (앞뒤 둘 다 있음)
id_appear = Counter()      # 없다가 생김 / 있다가 사라짐
id_renum_in_state = Counter()

n = bad = 0
last_rx = None

def feed(now, raw_state, conf):
    for r, tr in trackers.items():
        st = tr.update(now, raw_state=raw_state, confidence=conf, last_received=last_rx)
        if shown[r] is None or n <= 10:   # 켜자마자 빨→녹 한 번은 빼고 센다
            shown[r], shown_since[r] = st.state, now
        elif st.state != shown[r]:
            if now - shown_since[r] < 0.5:
                short_shown[r][shown[r]] += 1
            shown_changes[r][(shown[r], st.state)] += 1
            shown[r], shown_since[r] = st.state, now

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5005))
s.settimeout(0.1)
print(f"{SECONDS:.0f}초 동안 받습니다. (HUD 는 꺼져 있어야 함)")
t0 = time.monotonic()
while time.monotonic() - t0 < SECONDS:
    try:
        data = s.recvfrom(65535)[0]
    except socket.timeout:
        now = time.monotonic()
        if last_rx is not None and now - last_rx > timeout:
            feed(now, STATE_LOST, 0.0)   # 무수신 구간도 실제와 같이 반영
        continue
    try:
        pk = _decode_packet(data)
    except (PacketError, OverflowError, ValueError):
        bad += 1
        continue
    now = time.monotonic()
    last_rx = now
    n += 1

    raw = normalize_state(pk.get("state", STATE_NORMAL))
    if pk.get("status") == "lost":
        raw = STATE_LOST
    conf = float(pk.get("confidence", 1.0))

    if raw_prev is None or raw == raw_prev:
        raw_run += 1
    else:
        raw_runs[raw_prev][bucket(raw_run)] += 1
        raw_changes[(raw_prev, raw)] += 1
        raw_run = 1
    raw_prev = raw

    for side in ("left", "right"):
        cur = pk.get(f"id_{side}")
        cur = None if cur in (None, -1) else cur
        old = ids_prev[side]
        if old is not None and cur is not None and cur != old:
            id_renum[side] += 1
            id_renum_in_state[raw] += 1
        elif (old is None) != (cur is None) and n > 1:
            id_appear[side] += 1
        ids_prev[side] = cur

    feed(now, raw, conf)

if raw_prev is not None:
    raw_runs[raw_prev][bucket(raw_run)] += 1

dur = time.monotonic() - t0
per40 = 40.0 / dur
def fmt(c):
    return "  ".join(f"{NAMES[a]}→{NAMES[b]} {v}" for (a, b), v in sorted(c.items())) or "없음"

print(f"\n패킷 {n} ({n/dur:.1f}/초)  거부 {bad}  측정 {dur:.0f}초  (아래 '/40초' 는 40초 환산)")
print("\n① 패킷 state 그대로")
print(f"  바뀜 {sum(raw_changes.values())}회  (/40초 {sum(raw_changes.values())*per40:.0f})   {fmt(raw_changes)}")
print("  같은 state 가 이어진 길이(패킷 수)별 횟수")
for k in (0, 1, 2):
    c = raw_runs[k]
    print(f"    {NAMES[k]}  1개 {c['1']:3d}   2개 {c['2']:3d}   3~4개 {c['3~4']:3d}   5개+ {c['5+']:3d}")

print(f"\n①② 디바운스 뒤 화면 색 (fall {fall} 고정, 노란색 올라가는 확인 프레임만 바꿔 봄)")
for r in RISES:
    tot = sum(shown_changes[r].values())
    tag = "  ← 지금 설정" if r == 3 else ""
    print(f"  rise {r}: 바뀜 {tot}회 (/40초 {tot*per40:.0f})   "
          f"0.5초 안에 끝난 노랑 {short_shown[r][1]} · 빨강 {short_shown[r][2]}{tag}")
    print(f"          {fmt(shown_changes[r])}")

print("\n③ 선 번호")
print(f"  번호만 바뀜  왼 {id_renum['left']}  오 {id_renum['right']}  "
      f"(합 {sum(id_renum.values())}, /40초 {sum(id_renum.values())*per40:.0f})")
print(f"    그때 state: 녹 {id_renum_in_state[0]}  노 {id_renum_in_state[1]}  빨 {id_renum_in_state[2]}")
print(f"  선이 생기거나 사라짐  왼 {id_appear['left']}  오 {id_appear['right']}")
