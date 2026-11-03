import socket, sys, time
import numpy as np
from hud_system import LaneSmoother, PacketError, _decode_packet

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
ALPHAS = [1.0, 0.7, 0.5]
YS = [0.55, 0.65, 0.75]
PX = 960.0

sm = {a: LaneSmoother(a, 0.18, 0.0, 3) for a in ALPHAS}
prev = {a: None for a in ALPHAS}
dx = {a: [] for a in ALPHAS}
dlen = {a: [] for a in ALPHAS}
n = bad = id_change = st_change = 0
p_ids = p_state = None

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5005))
s.settimeout(2.0)
print(f"{SECONDS:.0f}초 동안 받습니다. 커브·직선을 한 번씩 지나가 주세요.")
t0 = time.time()
while time.time() - t0 < SECONDS:
    try:
        data = s.recvfrom(65535)[0]
    except socket.timeout:
        print("  (2초간 패킷 없음)")
        continue
    try:
        pk = _decode_packet(data)
    except (PacketError, OverflowError):
        bad += 1
        continue
    n += 1
    ids = tuple(pk.get("lane_ids") or [])
    if p_ids is not None and ids != p_ids:
        id_change += 1
    p_ids = ids
    stt = pk.get("state")
    if p_state is not None and stt != p_state:
        st_change += 1
    p_state = stt
    for a in ALPHAS:
        lanes = sm[a].update(pk["lanes"], pk.get("lane_ids"))
        cur = None
        if len(lanes) == 2 and stt != 2:
            cur = []
            for ln in lanes:
                p = np.asarray(ln, float)
                p = p[np.argsort(p[:, 1])]
                xs = {y: float(np.interp(y, p[:, 1], p[:, 0])) for y in YS if p[0, 1] <= y <= p[-1, 1]}
                cur.append((xs, float(p[-1, 1] - p[0, 1])))
        if cur is not None and prev[a] is not None:
            for (xs, ln), (pxs, pln) in zip(cur, prev[a]):
                dx[a] += [abs(xs[y] - pxs[y]) * PX for y in xs if y in pxs]
                dlen[a].append(abs(ln - pln) * PX)
        prev[a] = cur

def line(v):
    v = np.asarray(v)
    if not len(v):
        return "데이터 없음"
    return f"평균 {v.mean():5.2f}  중앙 {np.median(v):5.2f}  상위10% {np.percentile(v, 90):6.2f}  최대 {v.max():6.1f}"

print(f"\n패킷 {n} ({n / SECONDS:.1f}/초)  거부 {bad}  선 번호 바뀜 {id_change}회  state 바뀜 {st_change}회")
print("\n[가로 흔들림] 같은 높이에서 선이 옆으로 움직인 양 (카메라 px)")
for a in ALPHAS:
    print(f"  alpha {a:.1f}  {line(dx[a])}")
print("\n[길이 흔들림] 선 세로 길이 변화 (카메라 px) — 평활이 안 건드리는 부분")
for a in ALPHAS:
    print(f"  alpha {a:.1f}  {line(dlen[a])}")
