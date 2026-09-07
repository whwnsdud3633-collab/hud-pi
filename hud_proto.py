#!/usr/bin/env python3
"""젯슨 ↔ 파이 패킷 스키마. 양쪽이 공유하는 단일 출처.

    ┌─────────────┐   UDP 5005    ┌──────────────┐
    │ Jetson      │ ── encode ──▶ │ Raspberry Pi │
    │ (추론)      │               │ (렌더)       │
    └─────────────┘   ◀ decode ── └──────────────┘

**이 파일은 젯슨 쪽 담당자와 바이트 단위로 동일해야 한다.**
필드를 추가·삭제·재배치하면 젯슨 쪽 코드가 즉시 깨진다. 변경이 필요하면
코드를 먼저 고치지 말고 사람에게 알릴 것. 바꿀 때는 VERSION 을 올리고
양쪽을 동시에 교체한다.

여기에는 스키마와 인코딩/디코딩만 둔다. 스무딩·차선 선택·설정 로드 같은
동작은 hud_system 이 담당하고, 이 파일을 호출해서 쓴다.
"""

from __future__ import annotations

import json
import math
from typing import Any

# 스키마 버전. 필드가 하나라도 바뀌면 올린다.
VERSION = 1

# 젯슨이 고정 종방향 거리로 리샘플링해서 보내는 점 개수.
# 가변 길이면 슬롯별 시간축 필터를 걸 수 없어 오버레이가 떨린다.
POINT_COUNT = 12

# 각 슬롯의 종방향 거리(m). **인덱스 0 이 가장 가깝다 (근거리 → 원거리).**
# hud_align.DEFAULT_DEPTHS 와 같은 값이어야 하고, 깊이 가중 눈높이 보정이
# 이 순서를 그대로 믿는다. 뒤집히면 근거리와 원거리 보정이 서로 바뀐다.
DEPTHS: tuple[float, ...] = (
    5.00, 8.18, 11.36, 14.55, 17.73, 20.91,
    24.09, 27.27, 30.45, 33.64, 36.82, 40.00,
)

STATUS_VALUES = ("ok", "degraded", "lost")

WARNING_VALUES = (
    "none",
    "departure_left",
    "departure_right",
    "change_left",
    "change_right",
)

# 방어적 상한. 망가진 패킷 하나가 렌더 루프를 통째로 멈추게 두지 않는다.
MAX_LANES = 6
MAX_POINTS = 64
PACKET_MAX_BYTES = 65_535


class PacketError(Exception):
    """패킷이 스키마에 맞지 않는다. 수신부는 이걸 잡아서 세고 넘어간다."""


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------
#
# v1 페이로드는 UTF-8 JSON 오브젝트 하나다. 구분자는 (",", ":") 로 붙여 쓴다.
#
#   v                   int     스키마 버전. VERSION 과 다르면 거부
#   seq                 int     송신 일련번호. 0 부터. 재시작하면 0 으로 돌아온다
#   sent_at             float   송신 시각 (time.time(), 초)
#   t_capture           int     프레임 캡처 시각 (CLOCK_MONOTONIC, ns) — 선택
#   status              str     STATUS_VALUES 중 하나
#   fps                 float   추론 파이프라인 실측 fps
#   inference_ms        float   추론 1회 소요 시간
#   lane_change         bool    차선 변경 중인지
#   warning             str     WARNING_VALUES 중 하나
#   confidence          float   0..1
#   departure_distance  float   차선까지 남은 거리(m)
#   lkas, acc           bool    상태 표시등
#   lanes               list    차선 폴리라인. [[[x, y], ...], ...] 정규화 0..1
#                               점은 POINT_COUNT 개, 근거리 → 원거리 순
#
# t_capture 는 CLAUDE.md 가 요구하지만 현재 송신부(hud_ui.HudUiSender)가
# 아직 넣지 않는다. 없으면 None 으로 디코드해 지연 보정을 건너뛴다.
# 필수로 만들려면 VERSION 을 2 로 올리고 양쪽을 같이 바꿔야 한다.

REQUIRED_FIELDS = ("v", "seq", "lanes")

DEFAULTS: dict[str, Any] = {
    "sent_at": 0.0,
    "t_capture": None,
    "status": "ok",
    "fps": 0.0,
    "inference_ms": 0.0,
    "lane_change": False,
    "warning": "none",
    "confidence": 1.0,
    "departure_distance": 0.0,
    "lkas": True,
    "acc": True,
}


# ---------------------------------------------------------------------------
# 인코딩
# ---------------------------------------------------------------------------


def encode(
    lanes: Any,
    *,
    seq: int,
    sent_at: float,
    status: str = "ok",
    fps: float = 0.0,
    inference_ms: float = 0.0,
    lane_change: bool = False,
    warning: str = "none",
    confidence: float = 1.0,
    departure_distance: float = 0.0,
    lkas: bool = True,
    acc: bool = True,
    t_capture: int | None = None,
) -> bytes:
    """패킷 하나를 바이트로. 젯슨 쪽에서 이 함수를 쓴다.

    hud_ui.HudUiSender 가 직접 만드는 dict 와 바이트 단위로 같은 결과를 낸다.
    두 곳이 갈라지지 않도록 송신부를 이 함수로 옮기는 편이 안전하다.
    """
    if warning not in WARNING_VALUES:
        raise ValueError(f"unknown warning: {warning}")
    if status not in STATUS_VALUES:
        raise ValueError(f"unknown status: {status}")
    packet = {
        "v": VERSION,
        "seq": int(seq),
        "sent_at": sent_at,
        "status": status,
        "fps": round(float(fps), 2),
        "inference_ms": round(float(inference_ms), 2),
        "lane_change": bool(lane_change),
        "warning": warning,
        "confidence": round(float(confidence), 3),
        "departure_distance": round(float(departure_distance), 2),
        "lkas": bool(lkas),
        "acc": bool(acc),
        "lanes": lanes,
    }
    if t_capture is not None:
        packet["t_capture"] = int(t_capture)
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# 디코딩
# ---------------------------------------------------------------------------


def _as_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PacketError(f"{field}: 숫자가 아니다 ({value!r})") from exc
    if not math.isfinite(number):
        raise PacketError(f"{field}: 유한한 값이 아니다 ({value!r})")
    return number


def _decode_lanes(raw: Any) -> list[list[list[float]]]:
    """차선 목록을 검사해서 정규화 폴리라인 리스트로 만든다."""
    if not isinstance(raw, (list, tuple)):
        raise PacketError(f"lanes: 리스트가 아니다 ({type(raw).__name__})")
    if len(raw) > MAX_LANES:
        raise PacketError(f"lanes: 너무 많다 ({len(raw)} > {MAX_LANES})")

    lanes: list[list[list[float]]] = []
    for index, lane in enumerate(raw):
        if not isinstance(lane, (list, tuple)):
            raise PacketError(f"lanes[{index}]: 리스트가 아니다")
        if len(lane) > MAX_POINTS:
            raise PacketError(f"lanes[{index}]: 점이 너무 많다 ({len(lane)})")
        points: list[list[float]] = []
        for point in lane:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                raise PacketError(f"lanes[{index}]: 점 형식이 [x, y] 가 아니다")
            x = _as_float(point[0], f"lanes[{index}].x")
            y = _as_float(point[1], f"lanes[{index}].y")
            points.append([x, y])
        # 점이 하나뿐인 차선은 그릴 수 없다. 거부가 아니라 조용히 버린다.
        if len(points) >= 2:
            lanes.append(points)
    return lanes


def decode(data: bytes) -> dict[str, Any]:
    """UDP 바이트 → 프레임 dict. 스키마에 안 맞으면 PacketError.

    빠진 선택 필드는 DEFAULTS 로 채워서 돌려주므로, 호출부는 status / fps /
    seq / lanes 를 .get 없이 바로 인덱싱해도 된다.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise PacketError(f"바이트가 아니다 ({type(data).__name__})")
    if len(data) > PACKET_MAX_BYTES:
        raise PacketError(f"패킷이 너무 크다 ({len(data)} bytes)")
    try:
        packet = json.loads(bytes(data).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PacketError("UTF-8 로 디코드할 수 없다") from exc
    except json.JSONDecodeError as exc:
        raise PacketError(f"JSON 이 아니다: {exc.msg}") from exc

    if not isinstance(packet, dict):
        raise PacketError(f"오브젝트가 아니다 ({type(packet).__name__})")
    for field in REQUIRED_FIELDS:
        if field not in packet:
            raise PacketError(f"필수 필드가 없다: {field}")
    if packet["v"] != VERSION:
        raise PacketError(
            f"스키마 버전이 다르다 (받은 값 {packet['v']!r}, 기대 {VERSION}). "
            "젯슨 쪽과 hud_proto 를 같이 교체해야 한다"
        )

    try:
        seq = int(packet["seq"])
    except (TypeError, ValueError) as exc:
        raise PacketError(f"seq: 정수가 아니다 ({packet['seq']!r})") from exc
    if seq < 0:
        raise PacketError(f"seq: 음수 ({seq})")

    status = packet.get("status", DEFAULTS["status"])
    if status not in STATUS_VALUES:
        raise PacketError(f"status: 모르는 값 ({status!r})")
    warning = packet.get("warning", DEFAULTS["warning"])
    if warning not in WARNING_VALUES:
        # 모르는 경고는 패킷을 버릴 이유가 못 된다. 안전한 쪽으로 떨어뜨린다.
        warning = "none"

    t_capture = packet.get("t_capture")
    if t_capture is not None:
        try:
            t_capture = int(t_capture)
        except (TypeError, ValueError) as exc:
            raise PacketError(f"t_capture: 정수가 아니다 ({t_capture!r})") from exc

    return {
        "v": VERSION,
        "seq": seq,
        "sent_at": _as_float(packet.get("sent_at", DEFAULTS["sent_at"]), "sent_at"),
        "t_capture": t_capture,
        "status": status,
        "fps": _as_float(packet.get("fps", DEFAULTS["fps"]), "fps"),
        "inference_ms": _as_float(
            packet.get("inference_ms", DEFAULTS["inference_ms"]), "inference_ms"),
        "lane_change": bool(packet.get("lane_change", DEFAULTS["lane_change"])),
        "warning": warning,
        "confidence": _as_float(
            packet.get("confidence", DEFAULTS["confidence"]), "confidence"),
        "departure_distance": _as_float(
            packet.get("departure_distance", DEFAULTS["departure_distance"]),
            "departure_distance"),
        "lkas": bool(packet.get("lkas", DEFAULTS["lkas"])),
        "acc": bool(packet.get("acc", DEFAULTS["acc"])),
        "lanes": _decode_lanes(packet["lanes"]),
    }
