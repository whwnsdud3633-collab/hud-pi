#!/usr/bin/env python3
"""젯슨 없이 레이더 거리 표시를 확인한다. 파이에서 receive 를 띄워 둔 채 실행.

    python3 tools/fake_radar_jetson.py

레이더 담당이 준 fake_jetson.py 를 파이 실제 수신 경로에 맞게 고친 것이다.
  - "lanes": [] 를 빼야 계수(left/right)가 그려진다. lanes 키가 있으면
    파이는 계수를 무시하고 lanes 를 그리므로, 빈 lanes 면 차선이 안 나온다
  - left/right 는 실측 패킷(2026-09-12) 계수. 원본 값은 두 선이 X 자로 교차했다
  - state 0 + dist_on (띄우면 안 됨), 잘못된 값 (무시해야 함) 단계를 더했다
"""

import json
import socket
import time

PI_IP, PORT = "127.0.0.1", 5005
LEFT = [-0.003151, 0.505674, 423.193818]
RIGHT = [-0.008059, 5.896502, -500.737265]
Y_RANGE = [216.0, 410.4]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
seq = 0


def send(state, **extra):
    global seq
    lanes = state != 2
    message = {
        "v": 1, "seq": seq, "ts": time.time(), "img_w": 960, "img_h": 540,
        "state": state, "conf": 1.0 if state == 0 else (0.7 if state == 1 else 0.0),
        "left": LEFT if lanes else None, "right": RIGHT if lanes else None,
        "y_range": Y_RANGE if lanes else None,
        "fps": 28.0, "id_left": 1, "id_right": 3,
        "event": "", "intent": "", "extra": None,
    }
    message.update(extra)
    sock.sendto(json.dumps(message).encode(), (PI_IP, PORT))
    seq += 1


def phase(text, frames, state, make=None):
    print(text, flush=True)
    for i in range(frames):
        send(state, **(make(i) if make else {}))
        time.sleep(1 / 28)


while True:
    phase("① state 0 녹색 차선 — 거리 없음", 60, 0)
    phase("② state 0 + dist_on — 차선만, 숫자 나오면 안 됨", 60, 0,
          lambda i: {"dist_on": True, "dist_m": 25.0})
    phase("③ state 1 노란 차선", 60, 1)
    phase("④ state 2 + 거리 20 m → 4 m 로 줄어듦", 112, 2,
          lambda i: {"dist_on": True, "dist_m": round((20.0 - i * 0.15) * 2) / 2})
    phase("⑤ state 2 + 잘못된 거리 — 숫자 사라지고 화면은 정상", 40, 2,
          lambda i: {"dist_on": True, "dist_m": ["x", None, -1, float("nan")][i % 4]})
    phase("⑥ state 2 + dist_on false — 숫자 없음", 40, 2,
          lambda i: {"dist_on": False, "dist_m": 12.0})
    phase("⑦ state 2 + 거리 10 m", 40, 2, lambda i: {"dist_on": True, "dist_m": 10.0})
    print("⑧ 송신 3초 중단 — 0.35 초 안에 숫자·차선 사라지고 상태 바 빨강", flush=True)
    time.sleep(3)
