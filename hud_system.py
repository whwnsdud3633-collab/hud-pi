#!/usr/bin/env python3
"""차선 좌표를 UDP로 받아 HUD에 흰색 연속선으로 표시하는 독립 실행 코드.

이 파일에는 YOLO 추론이나 학습 코드가 포함되지 않는다. 기존 차선 추론
파이프라인에서 HudSender를 import하여 최종 차선 폴리라인 좌표만 보내면 된다.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import time
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = 1
MAX_DATAGRAM_BYTES = 60_000
# 패킷이 실어 나르는 경고 어휘. 렌더러가 아니라 프로토콜 쪽 값이다.
WARNING_STATES = (
    "none",
    "departure_left",
    "departure_right",
    "change_left",
    "change_right",
)
DEFAULT_CONFIG: dict[str, Any] = {
    "network": {
        "bind_host": "0.0.0.0",
        "port": 5005,
        "packet_timeout_seconds": 0.35,
        # 커널 수신 버퍼 요청값(바이트). 리눅스는 이 값의 두 배를 잡는다.
        "recv_buffer_bytes": 16_384,
    },
    "state": {
        "lane_fade_seconds": 0.4,
        "caution_timeout_seconds": 1.2,
        "rise_frames": 3,
        "fall_frames": 5,
    },
    "display": {
        "width": 1920,
        "height": 1200,
        "fullscreen": True,
        "window_name": "Rain Lane HUD",
        "line_thickness": 8,
        "line_color_bgr": [255, 255, 255],
        "flip_horizontal": False,
        "flip_vertical": False,
        "show_status": False,
        "smoothing_alpha": 0.55,
        "association_distance": 0.18,
        # 한 프레임에 허용할 차선 x 이동량(정규화). 실측에서 같은 트랙의
        # 프레임 간 이동은 p99 가 0.018 이라 0.05 면 정상 주행을 막지 않는다.
        "jump_gate_distance": 0.05,
        # 이만큼 연속으로 임계값을 넘으면 실제 이동으로 보고 받아들인다.
        "jump_gate_frames": 3,
    },
    "destination_quad_normalized": [
        [0.08, 0.08],
        [0.92, 0.08],
        [0.98, 0.98],
        [0.02, 0.98],
    ],
}


class PacketError(ValueError):
    """잘못된 HUD 좌표 패킷."""


def write_default_config(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        print(f"config already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"config created: {path}")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        write_default_config(path)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_config(path: Path, config: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _segment_span_in_box(
    first: list[float], second: list[float], bounds: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    """선분이 상자 안에 있는 매개변수 구간 [t0, t1] 을 구한다 (Liang-Barsky).

    상자를 전혀 지나지 않으면 None. t 는 first -> second 를 0 -> 1 로 본 값이다.
    """
    x_min, y_min, x_max, y_max = bounds
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    t0, t1 = 0.0, 1.0
    for slope, offset in (
        (-dx, first[0] - x_min),
        (dx, x_max - first[0]),
        (-dy, first[1] - y_min),
        (dy, y_max - first[1]),
    ):
        if slope == 0.0:
            # 이 축으로 움직이지 않는다. 시작부터 밖이면 통째로 버린다.
            if offset < 0.0:
                return None
            continue
        ratio = offset / slope
        if slope < 0.0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)
    return (t0, t1) if t0 <= t1 else None


def _lerp_point(first: list[float], second: list[float], t: float) -> list[float]:
    return [
        first[0] + (second[0] - first[0]) * t,
        first[1] + (second[1] - first[1]) * t,
    ]


def _clip_polyline(
    points: list[list[float]],
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> list[list[float]]:
    """상자 밖으로 나간 구간을 버리고, 경계를 넘는 자리에 교점을 끼운다.

    화면 밖 x 를 0 이나 1 로 뭉개면(예전 clamp) 여러 점이 같은 경계값에 몰려
    테두리를 따라 달리는 수직선이 생긴다. 다항식이 화면 밖을 크게 도는 구간이
    정확히 그렇다. 그래서 버리는 쪽으로 가되, 그냥 걷어 내면 선이 경계에서
    뚝 끊기므로 경계와의 교점을 계산해 끝점으로 넣는다.

    상자를 들락날락하면 조각이 여러 개 나온다. 차선은 폴리라인 하나로
    다뤄지므로 그중 가장 긴 조각만 남긴다.
    """
    if len(points) < 2:
        return []

    pieces: list[list[list[float]]] = []
    current: list[list[float]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            pieces.append(current)
        current = []

    for index in range(len(points) - 1):
        first = points[index]
        second = points[index + 1]
        span = _segment_span_in_box(first, second, bounds)
        if span is None:
            flush()
            continue
        t0, t1 = span
        if t0 > 0.0:
            # 이 선분에서 상자로 들어왔다. 앞 조각과는 이어지지 않는다.
            flush()
        if not current:
            current = [_lerp_point(first, second, t0)]
        current.append(_lerp_point(first, second, t1))
        if t1 < 1.0:
            # 이 선분에서 상자를 빠져나갔다.
            flush()
    flush()

    if not pieces:
        return []
    return max(pieces, key=_path_length)


def _path_length(points: list[list[float]]) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        dx = points[index + 1][0] - points[index][0]
        dy = points[index + 1][1] - points[index][1]
        total += math.hypot(dx, dy)
    return total


def _sanitize_lane(lane: Iterable[Iterable[float]], max_points: int = 48) -> list[list[float]]:
    points: list[list[float]] = []
    for raw_point in lane:
        point = list(raw_point)
        if len(point) != 2 or not _finite(point[0]) or not _finite(point[1]):
            continue
        points.append([float(point[0]), float(point[1])])
    points.sort(key=lambda item: item[1])
    # 화면 밖은 clamp 하지 않고 잘라 낸다. clamp 하면 경계에 점이 몰려
    # 차선이 테두리를 따라 꺾여 올라간다.
    points = _clip_polyline(points)
    if len(points) > max_points:
        indices = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
        points = [points[index] for index in indices]
    if len(points) < 2:
        return []
    return [[round(x, 5), round(y, 5)] for x, y in points]


def _select_hud_lanes_with_ids(
    lanes: Iterable[Iterable[Iterable[float]]],
    ids: Iterable[Any] | None = None,
    *,
    lane_change: bool,
) -> tuple[list[list[list[float]]], list[Any]]:
    """평상시 2개, 차선 변경 시 최대 3개 차선을 선택한다.

    젯슨이 보내는 차선 트랙 id 를 같이 받아 정렬·선택 뒤에도 차선과 짝이
    유지되게 돌려준다. id 는 LaneSmoother 가 차선 동일성 판정에 쓴다.
    id 를 모르는 차선 자리에는 None 이 들어간다.
    """
    id_list = list(ids) if ids is not None else []
    valid: list[tuple[list[list[float]], Any]] = []
    for index, lane in enumerate(lanes):
        cleaned = _sanitize_lane(lane)
        if not cleaned:
            continue
        lane_id = id_list[index] if index < len(id_list) else None
        valid.append((cleaned, lane_id))
    valid.sort(key=lambda item: item[0][-1][0])
    limit = 3 if lane_change else 2
    if len(valid) > limit:
        bottom_x = [lane[-1][0] for lane, _ in valid]
        if lane_change:
            selected = sorted(
                range(len(valid)),
                key=lambda index: abs(bottom_x[index] - 0.5),
            )[:3]
        else:
            left = [index for index, x in enumerate(bottom_x) if x <= 0.5]
            right = [index for index, x in enumerate(bottom_x) if x > 0.5]
            if left and right:
                selected = [left[-1], right[0]]
            else:
                selected = sorted(
                    range(len(valid)),
                    key=lambda index: abs(bottom_x[index] - 0.5),
                )[:2]
        valid = [valid[index] for index in sorted(selected)]
    return [lane for lane, _ in valid], [lane_id for _, lane_id in valid]


def _select_hud_lanes(
    lanes: Iterable[Iterable[Iterable[float]]],
    *,
    lane_change: bool,
) -> list[list[list[float]]]:
    """평상시 2개, 차선 변경 시 최대 3개 차선을 선택한다."""
    selected, _ = _select_hud_lanes_with_ids(lanes, None, lane_change=lane_change)
    return selected


# ---------------------------------------------------------------------------
# lanes 폴리라인 양끝 외삽
#
# 젯슨이 lanes 로 보내는 선은 커브에서 짧고 본넷 쪽이 비어 있어 HUD 에
# 토막처럼 뜬다. 정규화 이미지 공간에서(정렬·trim 적용 전) 양끝을 늘린다.
#   맞춤 차수(fit_degree): 기본 1차(직선). 이미지 공간에서 가까운 쪽 차선은
#     원래 거의 직선이고 휨은 소실점 근처에 몰려 있다. 끝 몇 점이 차지하는
#     y 범위가 좁아 2차항은 거의 노이즈로 정해지고, 그걸 멀리 늘리면 오차가
#     제곱으로 커져 끝이 갈고리처럼 말린다. 2 로 두면 예전 2차 외삽.
#   가까운 쪽(y 큰 쪽): near_to 까지.
#   먼 쪽(y 작은 쪽): 관측 길이의 far_ratio 만큼만. 0 이면 늘리지 않는다
#     (소실점을 넘어 두 선이 교차하는 경우가 있었다).
#   한쪽 끝에서 늘리는 y 거리는 min(관측 길이 × 0.6, max_extend) 로 막는다.
#     검출이 짧을 때 화면 절반을 지어내지 않게.
#   near_anchor (0 이면 끔): 가까운 쪽 끝을 항상 이 y 에 맞춘다. 켜면 near_to 는
#     무시한다. 가속할 때 본넷이 올라와 검출이 위에서 끝나면 선 끝이 따라
#     올라가 "차선이 들린" 것처럼 보였다. 끝점이 한 자리에 있어야 한다.
#       끝이 위에 있으면  near_anchor 까지 늘린다 (위 1차 외삽, max_extend 상한
#                         그대로. 상한에 걸리면 닿는 데까지만)
#       끝이 아래 있으면  near_anchor 아래 점을 버리고 그 y 의 점을 보간해 끝으로 둔다
#     2차식 경로도 y_range 가 본넷에 잘리는 건 같으므로 이 단계만 적용한다.
# 2차식(left/right 계수) 폴백 경로는 이미 y_range 로 펴서 오므로 대상이 아니다.
# ---------------------------------------------------------------------------

LANE_EXTEND_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "near_to": 1.0,
    "far_ratio": 0.0,
    "min_points": 6,
    "fit_points": 8,
    "fit_degree": 1,
    "max_rms": 0.02,
    "max_extend": 0.30,
    "near_anchor": 0.0,
}
# 생성한 점의 x 가 이 범위를 벗어나면 거기서 멈춘다
LANE_EXTEND_X_LIMITS = (-0.5, 1.5)
# 맞춤에 쓰는 최소 점 수. 이보다 적으면 외삽하지 않는다
LANE_EXTEND_MIN_FIT = 4
# 한쪽 끝에서 늘리는 거리의 상한 (관측 y 길이 대비)
LANE_EXTEND_OBSERVED_RATIO = 0.6


def lane_extend_settings(config: dict[str, Any]) -> dict[str, Any]:
    """display.lane_extend 를 기본값 위에 얹는다. 키가 없어도 동작한다."""
    settings = dict(LANE_EXTEND_DEFAULTS)
    raw = config.get("display", {}).get("lane_extend")
    if isinstance(raw, dict):
        settings.update(raw)
    settings["enabled"] = bool(settings["enabled"])
    for key in ("near_to", "far_ratio", "max_rms", "max_extend", "near_anchor"):
        settings[key] = float(settings[key])
    settings["fit_degree"] = 2 if int(settings["fit_degree"]) >= 2 else 1
    # 점이 fit_points 보다 적으면 있는 만큼 쓴다. 다만 LANE_EXTEND_MIN_FIT 미만은 안 늘린다
    settings["fit_points"] = max(LANE_EXTEND_MIN_FIT, int(settings["fit_points"]))
    settings["min_points"] = max(LANE_EXTEND_MIN_FIT, int(settings["min_points"]))
    return settings


def _extend_end(
    fit_y: Any,
    fit_x: Any,
    target_y: float,
    max_rms: float,
    degree: int,
    *,
    exact: bool = False,
) -> list[list[float]]:
    """끝 점들(fit_y 의 마지막이 끝 점)로 x = f(y) 를 degree 차로 맞춰 target_y 까지 편다.

    잔차 RMS 는 맞춘 차수 기준이다(1차면 직선 잔차).
    exact 면 남은 거리가 점 간격의 절반보다 짧아도 target_y 에 점 하나를 찍는다
    (near_anchor 는 끝이 정확히 그 y 에 와야 한다).

    돌려주는 점은 끝 점에서 바깥쪽 순서다. 끝 점 자체는 넣지 않는다.
    """
    import numpy as np

    end_y = float(fit_y[-1])
    span = target_y - end_y
    gaps = np.abs(np.diff(fit_y))
    step = float(np.median(gaps)) if gaps.size else 0.0
    if step <= 1e-6 or abs(span) <= 1e-6:
        return []
    if not exact and abs(span) < step * 0.5:
        return []
    # 끝 점 기준으로 y 를 옮겨 맞춘다. 0~1 원좌표 그대로보다 조건수가 좋다.
    rel_y = fit_y - end_y
    if float(np.ptp(rel_y)) <= 1e-6:
        return []
    coefficients = np.polyfit(rel_y, fit_x, degree)
    residual = fit_x - np.polyval(coefficients, rel_y)
    # 노이즈에 곡선을 맞추면 엉뚱한 방향으로 뻗는다. 잘 안 맞으면 늘리지 않는다.
    if float(np.sqrt(np.mean(residual * residual))) > max_rms:
        return []

    count = max(1, int(math.ceil(abs(span) / step - 1e-9)))
    offsets = np.linspace(span / count, span, count)
    xs = np.polyval(coefficients, offsets)
    low, high = LANE_EXTEND_X_LIMITS
    output: list[list[float]] = []
    for offset, x in zip(offsets, xs):
        if not (low <= x <= high):
            break
        output.append([round(float(x), 5), round(end_y + float(offset), 5)])
    return output


def _cut_near(points: Any, anchor: float) -> list[list[float]]:
    """y 오름차순 점들에서 anchor 보다 아래(y 큰 쪽) 점을 버리고 anchor 위치 점을 끝에 둔다."""
    keep = int((points[:, 1] <= anchor).sum())
    output = [[float(x), float(y)] for x, y in points[:keep]]
    if keep == 0:
        # 선 전체가 anchor 아래에 있다. 그릴 구간이 없다.
        return []
    if output[-1][1] < anchor:
        x0, y0 = output[-1]
        x1, y1 = (float(v) for v in points[keep])
        ratio = (anchor - y0) / (y1 - y0)
        output.append([round(x0 + (x1 - x0) * ratio, 5), round(anchor, 5)])
    return output


def extend_lane(
    lane: list[list[float]],
    settings: dict[str, Any],
    *,
    anchor_only: bool = False,
) -> list[list[float]]:
    """정규화 폴리라인 하나의 양끝을 늘린다. y 는 먼 쪽 -> 가까운 쪽 오름차순.

    anchor_only 면 near_anchor 맞추기만 한다 (2차식 경로).
    """
    import numpy as np

    anchor = settings["near_anchor"]
    if (anchor_only and anchor <= 0.0) or len(lane) < 2:
        return lane
    points = np.asarray(lane, dtype=np.float64)
    order = np.argsort(points[:, 1], kind="stable")
    points = points[order]
    ys = points[:, 1]
    xs = points[:, 0]
    fit = min(settings["fit_points"], len(points))
    can_fit = len(points) >= settings["min_points"] and fit >= LANE_EXTEND_MIN_FIT
    degree = settings["fit_degree"]
    max_rms = settings["max_rms"]
    min_y = float(ys[0])
    max_y = float(ys[-1])
    observed = max_y - min_y
    limit = min(observed * LANE_EXTEND_OBSERVED_RATIO, settings["max_extend"])

    body: list[list[float]] | None = None  # None 이면 원래 점 그대로
    near: list[list[float]] = []
    if anchor > 0.0:
        # near_anchor 가 near_to 보다 우선한다.
        if max_y > anchor:
            # 아래로 넘친 점은 버린다. 점 개수와 상관없이 한다.
            body = _cut_near(points, anchor)
        elif max_y < anchor and can_fit:
            # limit 안에서 anchor 까지. 끝 점이 마지막이 되도록 정순 그대로 넘긴다.
            near = _extend_end(
                ys[-fit:], xs[-fit:], min(anchor, max_y + limit), max_rms, degree,
                exact=True,
            )
    elif can_fit:
        # 가까운 쪽: near_to 까지, 단 limit 이내.
        near_to = min(settings["near_to"], max_y + limit)
        near = _extend_end(ys[-fit:], xs[-fit:], near_to, max_rms, degree)

    # 먼 쪽: 관측 길이의 far_ratio 만큼(limit 이내). 0 이면 건너뛴다.
    # 이미지 위(y<0)로는 넘기지 않는다. 2차식 경로는 대상이 아니다.
    far: list[list[float]] = []
    if not anchor_only and can_fit and settings["far_ratio"] > 0.0:
        far_to = max(0.0, min_y - min(settings["far_ratio"] * observed, limit))
        far = _extend_end(ys[:fit][::-1], xs[:fit][::-1], far_to, max_rms, degree)

    if body is None:
        if not near and not far:
            return lane
        body = [[float(x), float(y)] for x, y in points]
    return far[::-1] + body + near


def extend_lanes(
    lanes: list[list[list[float]]],
    settings: dict[str, Any],
    *,
    state: int = 0,
    anchor_only: bool = False,
) -> list[list[list[float]]]:
    """lanes 패킷의 차선들을 늘린다. 꺼져 있거나 state 2 면 그대로 돌려준다.

    anchor_only 는 2차식으로 편 선용이다. near_anchor 맞추기만 한다.
    near_anchor 로 잘라서 점이 2개 미만이 된 선은 뺀다.
    """
    if not settings["enabled"] or not lanes or state == 2:
        return lanes
    if anchor_only and settings["near_anchor"] <= 0.0:
        return lanes
    extended = [extend_lane(lane, settings, anchor_only=anchor_only) for lane in lanes]
    if settings["near_anchor"] > 0.0:
        extended = [lane for lane in extended if len(lane) >= 2]
    return extended


class HudSender:
    """추론 코드에서 사용하는 HUD UDP 송신 어댑터."""

    def __init__(self, host: str, port: int = 5005) -> None:
        self.destination = (host, int(port))
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.socket.close()

    def send_normalized(
        self,
        lanes: Iterable[Iterable[Iterable[float]]],
        *,
        lane_change: bool = False,
        status: str | None = None,
        fps: float = 0.0,
        inference_ms: float = 0.0,
    ) -> None:
        selected = _select_hud_lanes(lanes, lane_change=lane_change)
        if status is None:
            if not selected:
                status = "lost"
            elif len(selected) < 2:
                status = "degraded"
            else:
                status = "ok"
        packet = {
            "v": PROTOCOL_VERSION,
            "seq": self.sequence,
            "sent_at": time.time(),
            "status": status,
            "fps": round(float(fps), 2),
            "inference_ms": round(float(inference_ms), 2),
            "lane_change": bool(lane_change),
            "lanes": selected,
        }
        payload = json.dumps(packet, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_DATAGRAM_BYTES:
            raise PacketError(f"packet too large: {len(payload)} bytes")
        self.socket.sendto(payload, self.destination)
        self.sequence += 1

    def send_pixels(
        self,
        lanes: Iterable[Iterable[Iterable[float]]],
        *,
        frame_width: int,
        frame_height: int,
        lane_change: bool = False,
        status: str | None = None,
        fps: float = 0.0,
        inference_ms: float = 0.0,
    ) -> None:
        width = max(1, int(frame_width) - 1)
        height = max(1, int(frame_height) - 1)
        normalized = [
            [[float(x) / width, float(y) / height] for x, y in lane]
            for lane in lanes
        ]
        self.send_normalized(
            normalized,
            lane_change=lane_change,
            status=status,
            fps=fps,
            inference_ms=inference_ms,
        )

    def __enter__(self) -> "HudSender":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# 젯슨 실제 송신 포맷 어댑터 -------------------------------------------------
#
# 젯슨은 lanes 배열이 아니라 left/right 를 따로, 그것도 점 배열이 아니라
# 2차 다항식 계수로 보낸다. 실측 패킷(2026-09-12, 192.168.50.10)에서 확인:
#
#   {"v":1,"seq":292,"ts":1789149939.25,"img_w":960,"img_h":540,
#    "state":0,"conf":1.0,
#    "left":[-0.003151,0.505674,423.193818],
#    "right":[-0.008059,5.896502,-500.737265],
#    "fps":0.0,"id_left":1,"id_right":3,
#    "event":"","intent":"","extra":null,"y_range":[216.0,410.4]}
#
# left/right = [a, b, c] 이고 x = a*y^2 + b*y + c, 좌표 단위는 픽셀,
# y 의 유효 구간이 y_range = [y_top, y_bottom] 이다. 위 패킷을 풀면
# left 는 y=216 에서 x=385, y=410 에서 x=100, right 는 같은 구간에서
# x=397 -> x=562 로 소실점에서 수렴하고 아래에서 벌어진다.

# 계수를 몇 점으로 펴서 보낼지. CLAUDE.md 의 "점 개수 12개 고정" 을 따른다.
JETSON_LANE_SAMPLES = 12

# status 필드가 없어서 state 로 유추한다.
JETSON_STATE_STATUS = {0: "ok", 1: "degraded", 2: "lost"}


def _lane_track_id(value: Any) -> Any:
    """차선 트랙 id 를 비교 가능한 값으로 정리한다. 모르면 None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value:
        return value
    return None


def _is_jetson_packet(packet: dict[str, Any]) -> bool:
    """lanes 가 없고 left/right 가 있으면 젯슨 포맷으로 본다."""
    return "lanes" not in packet and ("left" in packet or "right" in packet)


def _poly_lane_to_normalized(
    coefficients: Any,
    y_range: Any,
    *,
    img_w: float,
    img_h: float,
    samples: int = JETSON_LANE_SAMPLES,
) -> list[list[float]]:
    """2차 계수 + y 구간을 0~1 정규화 폴리라인으로 편다."""
    if not isinstance(coefficients, (list, tuple)) or len(coefficients) != 3:
        raise PacketError("lane must be 3 polynomial coefficients")
    if not all(_finite(value) for value in coefficients):
        raise PacketError("lane coefficients must be finite")
    if not isinstance(y_range, (list, tuple)) or len(y_range) != 2:
        raise PacketError("y_range must be a pair")
    if not all(_finite(value) for value in y_range):
        raise PacketError("y_range must be finite")

    a, b, c = (float(value) for value in coefficients)
    y_top, y_bottom = (float(value) for value in y_range)
    if y_bottom < y_top:
        y_top, y_bottom = y_bottom, y_top
    if y_bottom - y_top < 1.0:
        raise PacketError("y_range span too small")

    # send_pixels 와 같은 규칙으로 나눈다 (마지막 픽셀이 1.0 이 되도록)
    width = max(1.0, float(img_w) - 1.0)
    height = max(1.0, float(img_h) - 1.0)

    points: list[list[float]] = []
    for index in range(samples):
        progress = index / (samples - 1)
        y = y_top + (y_bottom - y_top) * progress
        x = (a * y + b) * y + c
        points.append([x / width, y / height])
    return points


def _adapt_jetson_packet(packet: dict[str, Any]) -> None:
    """젯슨 포맷을 기존 lanes 포맷으로 옮긴다. 원본 키는 지우지 않는다."""
    img_w = packet.get("img_w")
    img_h = packet.get("img_h")
    if not _finite(img_w) or not _finite(img_h):
        raise PacketError("img_w/img_h must be numbers")
    if float(img_w) < 2.0 or float(img_h) < 2.0:
        raise PacketError("img_w/img_h must be at least 2")

    y_range = packet.get("y_range")
    lanes: list[list[list[float]]] = []
    lane_ids: list[Any] = []
    for side in ("left", "right"):
        coefficients = packet.get(side)
        if coefficients is None:
            continue
        # 한쪽이라도 계수가 있으면 y_range 없이는 펼 수 없다
        lanes.append(
            _poly_lane_to_normalized(
                coefficients,
                y_range,
                img_w=float(img_w),
                img_h=float(img_h),
            )
        )
        # 트랙 id 를 차선과 같은 순서로 실어 보낸다. 스무딩이 이걸로 차선
        # 동일성을 판정한다. 없으면 None 자리를 남겨 근접도 폴백으로 간다.
        lane_ids.append(_lane_track_id(packet.get(f"id_{side}")))
    packet["lanes"] = lanes
    packet["lane_ids"] = lane_ids

    # ts -> sent_at
    if "sent_at" not in packet and _finite(packet.get("ts")):
        packet["sent_at"] = float(packet["ts"])

    # status 가 없으면 state 로 유추한다
    if "status" not in packet:
        state = packet.get("state")
        status = None
        if isinstance(state, bool):
            state = None
        if isinstance(state, (int, float)) and _finite(state):
            status = JETSON_STATE_STATUS.get(int(state))
        if status is None:
            # state 도 못 믿으면 HudSender 와 같은 규칙으로 개수를 본다
            if not lanes:
                status = "lost"
            elif len(lanes) < 2:
                status = "degraded"
            else:
                status = "ok"
        packet["status"] = status

    # conf -> confidence (상태 계층이 읽는 이름)
    if "confidence" not in packet and _finite(packet.get("conf")):
        packet["confidence"] = float(packet["conf"])

    # event / intent 는 그대로 살려 두고, 이름이 정확히 일치할 때만
    # warning 으로 올린다. 젯슨 쪽 어휘를 아직 못 받아서 추측 매핑은 안 한다.
    if "warning" not in packet:
        for key in ("event", "intent"):
            value = packet.get(key)
            if isinstance(value, str) and value in WARNING_STATES:
                packet["warning"] = value
                break


def _drop_lanes_for_poly(packet: dict[str, Any]) -> None:
    """display.ignore_lanes 용. lanes / lane_ids 를 떨어뜨려 2차식 경로로 보낸다.

    젯슨이 lanes 를 같이 보내도 예전처럼 left/right 계수를 펴서 그리게 한다.
    A/B 비교와 발표 중 즉시 되돌리기용. left/right 가 아예 없는 패킷(mock,
    예전 송신기)은 떨어뜨리면 그릴 게 없어지므로 그대로 둔다.
    """
    if "left" not in packet and "right" not in packet:
        return
    packet.pop("lanes", None)
    packet.pop("lane_ids", None)


def _decode_packet(payload: bytes, *, ignore_lanes: bool = False) -> dict[str, Any]:
    if len(payload) > MAX_DATAGRAM_BYTES:
        raise PacketError("packet exceeds size limit")
    try:
        packet = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PacketError("invalid JSON") from exc
    if not isinstance(packet, dict):
        raise PacketError("packet must be an object")
    if packet.get("v", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise PacketError("unsupported protocol version")
    # 어댑터보다 먼저 해야 _is_jetson_packet 이 2차식 경로로 판정한다.
    if ignore_lanes:
        _drop_lanes_for_poly(packet)
    # 파이 내부 표시. 2차식 계수를 펴서 만든 lanes 는 외삽 대상이 아니다.
    packet["lanes_from_poly"] = _is_jetson_packet(packet)
    if packet["lanes_from_poly"]:
        _adapt_jetson_packet(packet)
    sequence = packet.get("seq")
    if not isinstance(sequence, int) or sequence < 0:
        raise PacketError("seq must be a non-negative integer")
    status = packet.get("status", "ok")
    if status not in {"ok", "degraded", "lost", "mock"}:
        raise PacketError("unknown status")
    lane_change = bool(packet.get("lane_change", False))
    raw_lanes = packet.get("lanes", [])
    if not isinstance(raw_lanes, list):
        raise PacketError("lanes must be a list")
    raw_ids = packet.get("lane_ids")
    if not isinstance(raw_ids, list):
        raw_ids = None
    # 선택·정렬로 차선 순서가 바뀌므로 id 도 같이 따라가야 짝이 안 어긋난다.
    packet["lanes"], packet["lane_ids"] = _select_hud_lanes_with_ids(
        raw_lanes, raw_ids, lane_change=lane_change
    )
    packet["status"] = status
    packet["lane_change"] = lane_change
    packet["fps"] = float(packet.get("fps", 0.0))
    packet["inference_ms"] = float(packet.get("inference_ms", 0.0))
    return packet


# 기본값 212992B 면 젯슨 패킷(~260B, 커널 계산으로 ~1KB)이 221개, 28fps 로
# 7.9초치가 쌓인다. 매 프레임 전부 비우므로 평소에는 1~2개만 들어 있고,
# 버퍼는 렌더가 멈췄을 때만 의미가 있다. 가득 차면 커널은 새 패킷을 버리고
# 옛 패킷을 남기므로 크게 잡을수록 멈춘 뒤 첫 화면이 더 옛날 것이 된다.
# 16384 요청 -> 실제 32768, 젯슨 패킷 34개(1.2초)다.
DEFAULT_RECV_BUFFER_BYTES = 16_384


def open_receiver(
    bind_host: str, port: int, recv_buffer_bytes: int = DEFAULT_RECV_BUFFER_BYTES
) -> socket.socket:
    """수신용 논블로킹 UDP 소켓."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(recv_buffer_bytes))
    receiver.bind((bind_host, port))
    receiver.setblocking(False)
    return receiver


def receive_latest(
    receiver: socket.socket,
    timeout: float,
    last_sequence: int,
    *,
    ignore_lanes: bool = False,
) -> tuple[dict[str, Any] | None, int, int, Any]:
    """쌓인 패킷을 전부 꺼내 가장 최근 것 하나만 돌려준다.

    한 번에 하나씩 꺼내면 렌더가 송신보다 느린 순간마다 커널 버퍼에 패킷이
    쌓이고, 화면은 그만큼 옛날 차선을 계속 그린다.
    seq 가 last_sequence 보다 작은 패킷(역전)은 쓰지 않는다.
    ignore_lanes 가 참이면 lanes 를 버리고 2차식 계수로 그린다 (_decode_packet).

    반환: (패킷 또는 None, 안 쓰고 버린 정상 패킷 수, 깨진 패킷 수,
    마지막 깨진 패킷의 송신 주소)
    """
    readable, _, _ = select.select([receiver], [], [], timeout)
    if not readable:
        return None, 0, 0, None
    best: dict[str, Any] | None = None
    valid = 0
    invalid = 0
    invalid_address = None
    while True:
        try:
            payload, address = receiver.recvfrom(65_535)
        except BlockingIOError:
            break
        try:
            packet = _decode_packet(payload, ignore_lanes=ignore_lanes)
        except (PacketError, ValueError, TypeError):
            invalid += 1
            invalid_address = address
            continue
        valid += 1
        if packet["seq"] < last_sequence:
            continue
        if best is None or packet["seq"] >= best["seq"]:
            best = packet
    skipped = valid - (1 if best is not None else 0)
    return best, skipped, invalid, invalid_address


class LaneSmoother:
    """차선을 프레임 사이에 대응시켜 HUD 흔들림을 줄인다.

    대응은 트랙 id 를 먼저 본다. 젯슨이 id_left / id_right 로 차선 트랙 id 를
    보내므로, 운전자가 차선을 바꿔 id 가 달라지면 화면상 같은 자리에 있어도
    다른 차선이다. 근접도로만 이으면 EMA 가 두 차선을 하나로 이어 붙여
    차선 변경 순간에 없던 급커브가 생긴다. id 가 바뀌면 그 차선의 히스토리를
    버리고 새 차선으로 시작한다.

    id 가 없는 패킷(mock, 예전 송신기)은 예전처럼 아래쪽 x 근접도로 잇는다.

    급변 게이트: 한 프레임에 jump_gate_distance 보다 크게 움직이면 그 변화를
    임계값까지만 반영한다. 급정지처럼 차체가 순간적으로 흔들릴 때 선이 튀는
    것을 막는다. 다만 jump_gate_frames 연속으로 계속 크게 움직이면 실제로
    이동한 것이므로 그대로 받아들인다. 아니면 선이 영영 따라오지 못한다.
    """

    def __init__(
        self,
        alpha: float,
        association_distance: float,
        jump_gate_distance: float = 0.0,
        jump_gate_frames: int = 3,
    ) -> None:
        self.alpha = float(alpha)
        self.association_distance = float(association_distance)
        self.jump_gate_distance = float(jump_gate_distance)
        self.jump_gate_frames = int(jump_gate_frames)
        # 트랙 하나 = {"points": ndarray, "id": Any, "hold": int}
        self.tracks: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.tracks = []

    def _associate(self, current: list[tuple[Any, Any]]) -> list[int | None]:
        """이번 프레임 차선마다 이어 붙일 직전 트랙 번호를 정한다."""
        unused = set(range(len(self.tracks)))
        matches: list[int | None] = [None] * len(current)

        # 1) 트랙 id 가 같으면 거리와 무관하게 같은 차선이다.
        for index, (_, lane_id) in enumerate(current):
            if lane_id is None:
                continue
            for track_index in sorted(unused):
                if self.tracks[track_index]["id"] == lane_id:
                    matches[index] = track_index
                    unused.discard(track_index)
                    break

        # 2) 남은 것만 근접도로 잇는다. 양쪽 다 id 가 있는데 1) 에서 안 붙었다면
        #    id 가 바뀐 것이므로 이어서는 안 된다. 한쪽이라도 id 를 모를 때만
        #    근접도 폴백을 허용한다.
        for index, (lane, lane_id) in enumerate(current):
            if matches[index] is not None:
                continue
            best_index = None
            best_distance = float("inf")
            for track_index in unused:
                if lane_id is not None and self.tracks[track_index]["id"] is not None:
                    continue
                distance = abs(
                    float(lane[-1, 0] - self.tracks[track_index]["points"][-1, 0])
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = track_index
            if best_index is not None and best_distance <= self.association_distance:
                matches[index] = best_index
                unused.discard(best_index)
        return matches

    def update(
        self,
        lanes: list[list[list[float]]],
        ids: list[Any] | None = None,
    ) -> list[list[list[float]]]:
        import numpy as np

        id_list = list(ids) if ids is not None else []
        current = [
            (
                np.asarray(lane, dtype=np.float32),
                id_list[index] if index < len(id_list) else None,
            )
            for index, lane in enumerate(lanes)
        ]
        current.sort(key=lambda item: float(item[0][-1, 0]))
        matches = self._associate(current)

        tracks: list[dict[str, Any]] = []
        for index, (lane, lane_id) in enumerate(current):
            hold = 0
            track_index = matches[index]
            if track_index is not None:
                previous = self.tracks[track_index]["points"]
                previous_x = np.interp(lane[:, 1], previous[:, 1], previous[:, 0])
                blended = self.alpha * lane[:, 0] + (1.0 - self.alpha) * previous_x
                if self.jump_gate_distance > 0.0:
                    delta = blended - previous_x
                    if float(np.max(np.abs(delta))) > self.jump_gate_distance:
                        hold = int(self.tracks[track_index]["hold"]) + 1
                        if hold <= self.jump_gate_frames:
                            blended = previous_x + np.clip(
                                delta,
                                -self.jump_gate_distance,
                                self.jump_gate_distance,
                            )
                        else:
                            # 계속 크게 움직인다면 흔들림이 아니라 실제 이동이다.
                            hold = 0
                lane[:, 0] = blended
            tracks.append({"points": lane, "id": lane_id, "hold": hold})

        self.tracks = tracks
        return [np.round(track["points"], 5).tolist() for track in tracks]


def _build_homography(quad: list[list[float]], width: int, height: int) -> Any:
    import cv2
    import numpy as np

    source = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    destination = np.asarray(quad, dtype=np.float32)
    destination[:, 0] *= width - 1
    destination[:, 1] *= height - 1
    return cv2.getPerspectiveTransform(source, destination)


def _transform_lane(
    lane: list[list[float]],
    matrix: Any,
    *,
    flip_horizontal: bool,
    flip_vertical: bool,
) -> Any:
    import cv2
    import numpy as np

    points = np.asarray(lane, dtype=np.float32)
    if flip_horizontal:
        points[:, 0] = 1.0 - points[:, 0]
    if flip_vertical:
        points[:, 1] = 1.0 - points[:, 1]
    transformed = cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix)[0]
    return np.rint(transformed).astype(np.int32)


def run_receiver(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    config = load_config(args.config)
    network = config["network"]
    display = config["display"]
    width = int(display["width"])
    height = int(display["height"])
    bind_host = args.bind_host or str(network["bind_host"])
    port = args.port or int(network["port"])
    timeout = float(network["packet_timeout_seconds"])
    color = tuple(int(value) for value in display["line_color_bgr"])
    thickness = int(display["line_thickness"])
    matrix = _build_homography(config["destination_quad_normalized"], width, height)
    smoother = LaneSmoother(
        alpha=float(display.get("smoothing_alpha", 0.55)),
        association_distance=float(display.get("association_distance", 0.18)),
        jump_gate_distance=float(display.get("jump_gate_distance", 0.05)),
        jump_gate_frames=int(display.get("jump_gate_frames", 3)),
    )

    receiver = open_receiver(
        bind_host,
        port,
        int(network.get("recv_buffer_bytes", DEFAULT_RECV_BUFFER_BYTES)),
    )
    ignore_lanes = bool(display.get("ignore_lanes", False))

    window_name = str(display["window_name"])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if bool(display["fullscreen"]) and not args.windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, width, height)

    latest: dict[str, Any] | None = None
    latest_received = 0.0
    last_sequence = -1
    invalid_packets = 0
    print(f"HUD listening on {bind_host}:{port} / {width}x{height}; Q or Esc to stop")

    try:
        while True:
            packet, _, invalid, address = receive_latest(
                receiver, 0.01, last_sequence, ignore_lanes=ignore_lanes
            )
            for _ in range(invalid):
                invalid_packets += 1
                if invalid_packets % 30 == 1:
                    print(f"ignored invalid packet from {address}")
            if packet is not None:
                packet["lanes"] = smoother.update(
                    packet["lanes"], packet.get("lane_ids")
                )
                latest = packet
                latest_received = time.monotonic()
                last_sequence = packet["seq"]

            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            fresh = latest is not None and time.monotonic() - latest_received <= timeout
            if fresh and latest["status"] != "lost":
                for lane in latest["lanes"]:
                    points = _transform_lane(
                        lane,
                        matrix,
                        flip_horizontal=bool(display["flip_horizontal"]),
                        flip_vertical=bool(display["flip_vertical"]),
                    )
                    cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
                if bool(display.get("show_status", False)):
                    cv2.putText(
                        canvas,
                        f"{latest['status']} | {len(latest['lanes'])} lanes | {latest['fps']:.1f} FPS",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            else:
                smoother.reset()

            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        cv2.destroyAllWindows()


def run_calibration(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    config = load_config(args.config)
    display = config["display"]
    width = int(display["width"])
    height = int(display["height"])
    initial = np.asarray(config["destination_quad_normalized"], dtype=np.float32)
    points = initial.copy()
    selected = 0
    name = "HUD calibration | 1-4 select | arrows move | S save | Q quit"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if bool(display["fullscreen"]) and not args.windowed:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(name, width, height)

    while True:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        pixels = np.rint(points * np.asarray([width - 1, height - 1])).astype(np.int32)
        cv2.polylines(canvas, [pixels], True, (255, 255, 255), 3, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            color = (0, 255, 255) if index == selected else (255, 255, 255)
            cv2.circle(canvas, tuple(point), 14, color, 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index + 1),
                tuple(point + [18, -10]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        cv2.putText(
            canvas,
            "1-4 SELECT | ARROWS MOVE | S SAVE | R RESET | Q QUIT",
            (24, height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 180),
            2,
        )
        cv2.imshow(name, canvas)
        key = cv2.waitKeyEx(20)
        if ord("1") <= key <= ord("4"):
            selected = key - ord("1")
        elif key in (81, 2424832):
            points[selected, 0] -= 0.0025
        elif key in (83, 2555904):
            points[selected, 0] += 0.0025
        elif key in (82, 2490368):
            points[selected, 1] -= 0.0025
        elif key in (84, 2621440):
            points[selected, 1] += 0.0025
        elif key in (ord("r"), ord("R")):
            points = initial.copy()
        elif key in (ord("s"), ord("S")):
            config["destination_quad_normalized"] = np.round(np.clip(points, 0, 1), 5).tolist()
            save_config(args.config, config)
            print(f"calibration saved: {args.config}")
        elif key in (27, ord("q"), ord("Q")):
            break
        points = np.clip(points, 0.0, 1.0)
    cv2.destroyAllWindows()


def _mock_lane(
    bottom_x: float, top_x: float, phase: float, count: int = 24, curve: float = 0.0
) -> list[list[float]]:
    """curve 는 원거리 끝의 가로 휨(정규화 x). 근거리 0 에서 원거리로 갈수록
    제곱으로 커진다. + 면 오른쪽 커브, - 면 왼쪽 커브."""
    output = []
    for index in range(count):
        y = 0.38 + 0.60 * index / (count - 1)
        progress = (y - 0.38) / 0.60
        x = top_x + (bottom_x - top_x) * progress**1.45
        x += 0.012 * math.sin(phase + progress * 2.2)
        x += curve * (1.0 - progress) ** 2
        output.append([min(1.0, max(0.0, x)), y])
    return output


def run_mock_sender(args: argparse.Namespace) -> None:
    host, port_text = args.destination.rsplit(":", 1)
    sender = HudSender(host, int(port_text))
    started = time.monotonic()
    print(f"mock sender -> {host}:{port_text}; Ctrl+C to stop")
    try:
        while True:
            elapsed = time.monotonic() - started
            lane_change = bool(args.cycle and int(elapsed / 6) % 2)
            phase = elapsed * 0.8
            curve = args.curve
            if args.curve_period > 0:
                # 좌우 커브를 번갈아. 페이드 비교 중에 양쪽 휨을 다 보려는 용도
                curve *= math.sin(2.0 * math.pi * elapsed / args.curve_period)
            if lane_change:
                lanes = [
                    _mock_lane(0.14, 0.42, phase, curve=curve),
                    _mock_lane(0.50, 0.54, phase + 0.2, curve=curve),
                    _mock_lane(0.86, 0.68, phase + 0.4, curve=curve),
                ]
            else:
                lanes = [
                    _mock_lane(0.18, 0.43, phase, curve=curve),
                    _mock_lane(0.82, 0.57, phase + 0.3, curve=curve),
                ]
            sender.send_normalized(
                lanes,
                lane_change=lane_change,
                status="mock",
                fps=args.fps,
            )
            time.sleep(max(0.001, 1.0 / args.fps))
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="차선 AR-HUD 표시 및 좌표 송신 어댑터")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="기본 hud_config.json 생성")
    init_parser.add_argument("--config", type=Path, default=Path("hud_config.json"))
    init_parser.add_argument("--overwrite", action="store_true")

    receive_parser = subparsers.add_parser("receive", help="UDP 차선 좌표를 전체화면 HUD로 표시")
    receive_parser.add_argument("--config", type=Path, default=Path("hud_config.json"))
    receive_parser.add_argument("--bind-host")
    receive_parser.add_argument("--port", type=int)
    receive_parser.add_argument("--windowed", action="store_true")

    calibration_parser = subparsers.add_parser("calibrate", help="HUD 사다리꼴 투영 영역 보정")
    calibration_parser.add_argument("--config", type=Path, default=Path("hud_config.json"))
    calibration_parser.add_argument("--windowed", action="store_true")

    mock_parser = subparsers.add_parser("mock", help="모델 없이 2/3개 차선 좌표 송신")
    mock_parser.add_argument("--destination", default="127.0.0.1:5005")
    mock_parser.add_argument("--fps", type=float, default=22.0)
    mock_parser.add_argument("--cycle", action="store_true", help="6초마다 일반/차선 변경 전환")
    mock_parser.add_argument("--curve", type=float, default=0.0,
                             help="원거리 끝 가로 휨(정규화 x). 0.15 정도면 뚜렷. 음수는 왼쪽")
    mock_parser.add_argument("--curve-period", type=float, default=0.0,
                             help="0 보다 크면 이 주기(초)로 좌우 커브를 번갈아")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init-config":
        write_default_config(args.config, overwrite=args.overwrite)
    elif args.command == "receive":
        run_receiver(args)
    elif args.command == "calibrate":
        run_calibration(args)
    elif args.command == "mock":
        run_mock_sender(args)


if __name__ == "__main__":
    main()
