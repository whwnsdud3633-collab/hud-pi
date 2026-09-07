#!/usr/bin/env python3
"""HUD 상태 계층. 설정, 차선 선택, 시간축 스무딩, 패킷 입출력.

렌더러(hud_theme / hud_ui)와 배선(hud_proto) 사이에 있다. 화면에 무엇을
그릴지는 모르고, 바이트를 어떻게 눕히는지도 모른다. 그 사이의 판단만 한다.

패킷 스키마는 전부 hud_proto 에 있다. 이 파일은 그걸 호출하기만 한다.
스키마를 여기서 다시 해석하지 말 것. 두 곳이 갈라지면 젯슨 쪽과 조용히
어긋난다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hud_align import ensure_alignment_config
from hud_proto import DEPTHS, POINT_COUNT, PacketError, decode

__all__ = [
    "PacketError",
    "LaneSmoother",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG",
]


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "display": {
        "width": 1280,
        "height": 720,
        "fullscreen": True,
        "window_name": "AR HUD",
        # 패널 → 반사판 → 눈 경로에서 상이 뒤집힌다. 실차 장착 후 켠다.
        # 적용은 hud_align.AlignmentMap.project 가 한다.
        "flip_horizontal": False,
        # 시간축 스무딩. 1.0 이면 스무딩 없음, 낮을수록 무겁게 따라간다.
        "smoothing_alpha": 0.55,
        # 프레임 간 같은 차선으로 볼 근거리 x 차이(정규화).
        "association_distance": 0.18,
    },
    "network": {
        # 젯슨 직결 이더넷과 개발용 핫스팟 양쪽에서 받는다.
        "bind_host": "0.0.0.0",
        "port": 5005,
        # 이 시간 동안 새 패킷이 없으면 차선을 지운다.
        "packet_timeout_seconds": 0.5,
    },
}


def _merge_defaults(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    """빠진 키만 채운다. 사용자가 적어 둔 값은 건드리지 않는다."""
    for key, value in defaults.items():
        if isinstance(value, dict):
            _merge_defaults(target.setdefault(key, {}), value)
        else:
            target.setdefault(key, value)


def load_config(path: str | Path) -> dict[str, Any]:
    """설정을 읽고 빠진 항목을 기본값으로 채운다.

    파일이 없어도 기본값으로 동작한다. 파일을 만들지는 않는다. 저장은
    preview 의 S 키나 hud_align 의 보정 도구가 명시적으로 한다.
    """
    config: dict[str, Any] = {}
    file_path = Path(path)
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file_path}: JSON 을 읽을 수 없다 — {exc.msg}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{file_path}: 최상위가 오브젝트가 아니다")
        config = loaded
    _merge_defaults(config, DEFAULT_CONFIG)
    # alignment 기본값까지 채워 둬야 저장했을 때 보정 항목이 파일에 남는다.
    ensure_alignment_config(config)
    return config


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    """설정을 저장한다. 쓰다 죽어도 원본이 남도록 임시 파일에 쓰고 바꿔친다."""
    file_path = Path(path)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    temporary.replace(file_path)


# ---------------------------------------------------------------------------
# 패킷
# ---------------------------------------------------------------------------


def _decode_packet(data: bytes) -> dict[str, Any]:
    """UDP 바이트 → 프레임 dict.

    파싱은 전부 hud_proto.decode 가 한다. 여기서 필드를 다시 해석하지 말 것.
    """
    return decode(data)


# ---------------------------------------------------------------------------
# 차선 선택
# ---------------------------------------------------------------------------


def _near_x(lane: Any) -> float:
    """차선의 근거리 쪽 x. 점 순서에 상관없이 y 가 가장 큰 점을 쓴다.

    규약은 근거리 → 원거리지만, 추론 쪽이 뒤집어 보내도 선택이 흔들리지
    않도록 방향에 의존하지 않는 방식으로 고른다.
    """
    array = np.asarray(lane, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        return float("nan")
    return float(array[int(np.argmax(array[:, 1])), 0])


def _select_hud_lanes(lanes: Any, *, lane_change: bool = False) -> list[Any]:
    """자차가 달리고 있는 차로의 좌/우 차선을 고른다.

    화면 중앙(x=0.5)을 자차 위치로 보고, 왼쪽에서 가장 가까운 하나와
    오른쪽에서 가장 가까운 하나를 집는다. 3차선 도로에서 바깥 차선까지
    다 그리면 반사식 HUD 에서는 밝은 선이 늘어나 시야만 어지럽다.

    차선 변경 중에는 넘어가는 쪽 차선이 중앙을 가로지르므로 좌/우 구분이
    무너진다. 이때는 중앙에서 가까운 순으로 3개까지 남겨 전환을 보여 준다.
    """
    if not lanes:
        return []

    scored = []
    for lane in lanes:
        if lane is None or len(lane) < 2:
            continue
        x = _near_x(lane)
        if math.isnan(x):
            continue
        scored.append((x, lane))
    if not scored:
        return []

    if lane_change:
        scored.sort(key=lambda item: abs(item[0] - 0.5))
        return [lane for _, lane in scored[:3]]

    left = [item for item in scored if item[0] <= 0.5]
    right = [item for item in scored if item[0] > 0.5]
    selected = []
    if left:
        selected.append(max(left, key=lambda item: item[0]))
    if right:
        selected.append(min(right, key=lambda item: item[0]))
    if not selected:
        return []
    # 한쪽에만 차선이 잡혔다면 중앙에서 가까운 두 개로 채운다.
    if len(selected) == 1 and len(scored) >= 2:
        scored.sort(key=lambda item: abs(item[0] - 0.5))
        selected = scored[:2]
    selected.sort(key=lambda item: item[0])
    return [lane for _, lane in selected]


# ---------------------------------------------------------------------------
# 시간축 스무딩
# ---------------------------------------------------------------------------


class LaneSmoother:
    """차선을 프레임 사이에 이어 붙여 슬롯별로 지수 평활한다.

    점 개수가 고정(POINT_COUNT)이라 i 번째 점은 항상 같은 거리를 뜻한다.
    그래서 슬롯끼리 짝지어 필터를 걸 수 있다. 개수가 흔들리면 짝이 어긋나
    오히려 떨림이 커지므로, 그런 프레임은 평활하지 않고 그대로 통과시킨다.
    """

    def __init__(self, alpha: float = 0.55, association_distance: float = 0.18) -> None:
        # alpha 는 새 값을 얼마나 믿을지. 1.0 이면 스무딩 없음.
        self.alpha = float(min(1.0, max(0.0, alpha)))
        self.association_distance = float(association_distance)
        self._previous: list[np.ndarray] = []

    def reset(self) -> None:
        """송신기가 끊겼거나 재시작했다. 이어 붙일 과거를 버린다."""
        self._previous = []

    def update(self, lanes: Any) -> list[list[list[float]]]:
        """이번 프레임 차선을 받아 평활된 차선을 돌려준다."""
        if not lanes:
            self._previous = []
            return []

        current: list[np.ndarray] = []
        for lane in lanes:
            array = np.asarray(lane, dtype=np.float32)
            if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
                current.append(np.ascontiguousarray(array[:, :2]))

        smoothed: list[np.ndarray] = []
        taken: set[int] = set()
        for array in current:
            index = self._associate(array, taken)
            if index is None or self.alpha >= 1.0:
                smoothed.append(array)
                continue
            taken.add(index)
            previous = self._previous[index]
            smoothed.append(previous + (array - previous) * self.alpha)

        self._previous = smoothed
        return [lane.tolist() for lane in smoothed]

    def _associate(self, lane: np.ndarray, taken: set[int]) -> int | None:
        """직전 프레임에서 같은 차선을 찾는다. 없으면 None (새 트랙)."""
        best_index: int | None = None
        best_distance = self.association_distance
        near_x = _near_x(lane)
        for index, previous in enumerate(self._previous):
            if index in taken or previous.shape != lane.shape:
                continue
            distance = abs(_near_x(previous) - near_x)
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index


# ---------------------------------------------------------------------------
# 목 데이터
# ---------------------------------------------------------------------------


def _mock_lane(
    near_x: float,
    far_x: float,
    phase: float = 0.0,
    *,
    count: int = POINT_COUNT,
    horizon: float = 0.40,
) -> list[list[float]]:
    """네트워크 없이 화면을 볼 때 쓰는 가짜 차선.

    점을 y 축에 균등하게 놓지 않는다. 젯슨은 고정 종방향 거리로 리샘플링해서
    보내고, 원근에서 이미지 y 는 거리에 반비례하므로 (y = 지평선 + c/d) 실제
    패킷은 원거리 쪽이 촘촘하다. 목 데이터도 같은 간격을 써야 스무딩과 깊이
    가중 보정을 실제와 같은 조건에서 확인할 수 있다.

    반환은 [[x, y], ...] 정규화 좌표, **근거리 → 원거리** 순. hud_proto.DEPTHS
    의 슬롯 순서와 같아야 한다.
    """
    depths = np.asarray(DEPTHS, dtype=np.float32)
    if count != len(depths):
        depths = np.linspace(depths[0], depths[-1], count, dtype=np.float32)

    # y(d) = horizon + c/d, c 는 가장 가까운 점이 화면 아래(y=1)에 오도록 잡는다.
    c = (1.0 - horizon) * float(depths[0])
    ys = horizon + c / depths

    # x 는 이미지 위에서 직선이다. 곧은 차선은 투영해도 직선이므로 y 로 잰다.
    span = ys[0] - ys[-1]
    t = (ys[0] - ys) / span if span > 1e-6 else np.zeros_like(ys)
    xs = near_x + (far_x - near_x) * t

    # 완만한 곡선. 원거리로 갈수록 크게 흔들려 실제 커브처럼 보인다.
    xs = xs + 0.035 * t * t * math.sin(phase)

    return [[float(x), float(y)] for x, y in zip(xs, ys)]
