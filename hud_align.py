#!/usr/bin/env python3
"""운전자 시점 정렬. 정규화 노면 좌표를 패널 픽셀로 옮긴다.

이 모듈이 담당하는 것은 하나다. **눈 위치 파라미터를 받아 호모그래피를 만들고
차선 폴리라인을 패널 픽셀로 투영한다.** 그 파라미터를 로터리 인코더가 채우는지
아이트래킹이 채우는지는 여기서 알 바가 아니다. 나중에 소스만 갈아끼우면 된다.

    [로터리 인코더]  →  (트림 dy / dx)   ─┐
                                         ├→  hud_align  →  H  →  렌더
    [아이트래킹]     →  (실측 eye_y/z)   ─┘

핵심은 **점마다 깊이를 끌고 다니는 것**이다. 눈이 e 만큼 움직일 때 거리 d 의
노면 점이 가상상 평면에서 움직이는 양은 e 가 아니라 e × (1 − d_vi/d) 다.
가상상 거리 1m 기준으로 5m 점은 0.8e, 40m 점은 0.975e 만큼 움직인다. 단일
픽셀 오프셋으로는 근거리든 원거리든 한쪽이 반드시 어긋난다.

그래서 project() 는 (points, depths, valid) 세 개를 돌려준다. 호출부가
필터링·정렬을 할 때 두 배열에 똑같이 적용해야 깊이가 살아남는다.
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

# 젯슨이 고정 종방향 거리로 리샘플링해서 보내는 12개 슬롯의 거리(m).
#
#   경고: 이 값은 hud_proto 의 리샘플 거리와 **바이트 단위로 같아야 한다.**
#   한쪽만 바꾸면 깊이 가중 보정이 조용히 틀린 값을 쓴다. 화면은 그럴듯하게
#   나오지만 근거리와 원거리 정렬이 서로 다른 방향으로 어긋난다.
#   hud_proto 가 생기면 그쪽을 단일 출처로 삼고 여기서 import 한다.
DEFAULT_DEPTHS: tuple[float, ...] = (
    5.00, 8.18, 11.36, 14.55, 17.73, 20.91,
    24.09, 27.27, 30.45, 33.64, 36.82, 40.00,
)

# 사다리꼴 꼭짓점 순서. 네 점짜리 설정값은 전부 이 순서를 따른다.
#   0 근거리-좌, 1 근거리-우, 2 원거리-우, 3 원거리-좌  (시계 방향)
QUAD_ORDER = ("near_left", "near_right", "far_right", "far_left")

DEFAULT_ALIGNMENT: dict[str, Any] = {
    # 정규화 입력 좌표에서 이 y 보다 위(값이 작은 쪽)는 지평선 너머로 보고 버린다.
    "horizon": 0.40,
    # 젯슨이 보내는 정규화 좌표계에서 노면 사다리꼴이 차지하는 영역.
    # far 쪽 y 는 horizon 과 맞춰 둔다. 어긋나면 투영이 클리핑 경계에서 튄다.
    "source_quad": [[0.02, 1.00], [0.98, 1.00], [0.60, 0.40], [0.40, 0.40]],
    # 실측 보정 결과. 패널 픽셀 좌표 4점. None 이면 미보정 상태다.
    "panel_quad": None,
    # 미보정 폴백. 패널 크기에 대한 비율. 반사식 HUD 는 도로의 좁은 띠만
    # 덮으므로 화면 전체가 아니라 띠 모양으로 잡는다.
    "fallback_quad": [[0.04, 0.90], [0.96, 0.90], [0.60, 0.30], [0.40, 0.30]],
    # 눈높이 트림. 패널 픽셀 단위. 깊이 가중되어 적용된다.
    "trim_dy": 0.0,
    "trim_dx": 0.0,
    # 가상상 거리(m). 반사 광학계 설계값. panel.yaml 과 맞춰야 한다.
    "d_vi": 1.0,
    "depths": list(DEFAULT_DEPTHS),
    # valid 판정 여유. 패널 크기의 배수. 이 밖으로 나간 점은 유효하지 않다.
    # 지평선 근처 점은 호모그래피에서 수만 픽셀로 튀므로 반드시 걸러야 한다.
    "valid_margin": 3.0,
    # 지평선에서 잘릴 때 교점을 보간해 끼워 넣는다. 리본 윗변이 계단지지 않는다.
    "interpolate_horizon": True,
}


def ensure_alignment_config(config: dict[str, Any]) -> None:
    """설정에 alignment 기본값을 채워 넣는다. 이미 있는 값은 건드리지 않는다."""
    alignment = config.setdefault("alignment", {})
    for key, value in DEFAULT_ALIGNMENT.items():
        if isinstance(value, list):
            alignment.setdefault(key, [list(v) if isinstance(v, list) else v
                                       for v in value])
        else:
            alignment.setdefault(key, value)
    # 좌우 반전은 반사 경로에서 상이 뒤집히기 때문에 필요하다. 투영 단계에서
    # 처리하므로 여기서 존재를 보장해 둔다.
    display = config.setdefault("display", {})
    display.setdefault("flip_horizontal", False)
    display.setdefault("width", 1280)
    display.setdefault("height", 720)


def _as_quad(value: Any) -> np.ndarray | None:
    """4x2 실수 배열로 변환한다. 모양이 안 맞으면 None."""
    if value is None:
        return None
    try:
        quad = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    return quad


def _slot_depths(count: int, depths: np.ndarray) -> np.ndarray:
    """점 개수에 맞는 깊이 배열을 만든다.

    젯슨이 규약대로 12개를 보내면 슬롯 거리를 그대로 쓴다. 개수가 다르면
    (스무딩이 점을 지웠거나 목 데이터거나) 슬롯 위치를 비율로 보간한다.
    """
    if count == 0:
        return np.zeros(0, np.float32)
    if count == len(depths):
        return depths.copy()
    if len(depths) == 1:
        return np.full(count, depths[0], np.float32)
    source = np.linspace(0.0, 1.0, len(depths), dtype=np.float32)
    target = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.interp(target, source, depths).astype(np.float32)


def _as_lane_array(lane: Any, depths: np.ndarray) -> np.ndarray:
    """차선을 (N, 3) [x, y, depth] 배열로 정규화한다.

    받아들이는 형태:
      - [[x, y], ...]          정규화 폴리라인. 깊이는 슬롯 거리에서 채운다
      - [[x, y, d], ...]       깊이가 딸려 온 폴리라인. 그 값을 그대로 쓴다
      - clip_above_horizon() 이 돌려준 (N, 3) 배열
    """
    if lane is None:
        return np.zeros((0, 3), np.float32)
    array = np.asarray(lane, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        return np.zeros((0, 3), np.float32)
    if array.shape[1] >= 3:
        return np.ascontiguousarray(array[:, :3])
    out = np.empty((array.shape[0], 3), np.float32)
    out[:, :2] = array[:, :2]
    out[:, 2] = _slot_depths(array.shape[0], depths)
    return out


class AlignmentMap:
    """정규화 노면 좌표 → 패널 픽셀. 깊이를 같이 끌고 다닌다."""

    def __init__(self, config: dict[str, Any]) -> None:
        ensure_alignment_config(config)
        alignment = config["alignment"]
        display = config["display"]

        self.width = int(display["width"])
        self.height = int(display["height"])
        self.flip_horizontal = bool(display["flip_horizontal"])

        self.horizon = float(alignment["horizon"])
        self.d_vi = max(1e-3, float(alignment["d_vi"]))
        self.trim_dy = float(alignment["trim_dy"])
        self.trim_dx = float(alignment["trim_dx"])
        self.valid_margin = float(alignment["valid_margin"])
        self.interpolate_horizon = bool(alignment["interpolate_horizon"])

        depths = np.asarray(alignment["depths"], dtype=np.float32).ravel()
        if depths.size == 0 or not np.isfinite(depths).all():
            depths = np.asarray(DEFAULT_DEPTHS, dtype=np.float32)
        # 0m 이하는 1 − d_vi/d 를 발산시킨다. 물리적으로도 의미가 없다.
        self.depths = np.maximum(depths, self.d_vi + 1e-3)

        source = _as_quad(alignment["source_quad"])
        if source is None:
            source = _as_quad(DEFAULT_ALIGNMENT["source_quad"])
        self.source_quad = source

        panel = _as_quad(alignment["panel_quad"])
        self._calibrated = panel is not None
        if panel is None:
            fallback = _as_quad(alignment["fallback_quad"])
            if fallback is None:
                fallback = _as_quad(DEFAULT_ALIGNMENT["fallback_quad"])
            panel = fallback * np.asarray(
                [self.width - 1, self.height - 1], dtype=np.float32
            )
        self.panel_quad = panel

        self.homography = self._build_homography()
        # 슬롯별 깊이 가중치. 12개짜리 고정 입력에서는 매 프레임 다시 계산할
        # 필요가 없다. 개수가 다를 때만 즉석 계산으로 떨어진다.
        self._slot_weights = self._depth_weights(self.depths)

    # 호모그래피 ----------------------------------------------------------

    def _build_homography(self) -> np.ndarray:
        """정규화 사다리꼴 → 패널 사다리꼴 대응으로 H 를 만든다.

        Level 3 로 갈 때도 행렬 원소를 보간하지 말 것. 아이박스 4코너에서
        목적지 좌표 4쌍을 이중선형 보간한 뒤 이 함수를 다시 부르는 쪽이
        수치적으로 훨씬 안정적이다.
        """
        try:
            return cv2.getPerspectiveTransform(
                self.source_quad.astype(np.float32),
                self.panel_quad.astype(np.float32),
            )
        except cv2.error:
            # 사다리꼴이 찌부러졌다. 정규화 좌표를 패널에 그대로 펴는 것으로 뒤로 뺀다.
            return np.asarray(
                [[self.width - 1, 0.0, 0.0],
                 [0.0, self.height - 1, 0.0],
                 [0.0, 0.0, 1.0]], dtype=np.float64
            )

    @property
    def calibrated(self) -> bool:
        """실측 보정(panel_quad)이 저장되어 있는지."""
        return self._calibrated

    # 눈 위치 파라미터 ------------------------------------------------------

    def _depth_weights(self, depths: np.ndarray) -> np.ndarray:
        """깊이 d 의 점이 눈 이동 e 에 대해 움직이는 비율 (1 − d_vi/d).

        d → ∞ 이면 1(눈을 따라 그대로 움직인다), d → d_vi 이면 0(가상상
        평면 위의 점이라 움직이지 않는다).
        """
        safe = np.maximum(depths, self.d_vi + 1e-3)
        return np.clip(1.0 - self.d_vi / safe, 0.0, 1.0).astype(np.float32)

    def set_trim(self, dy: float, dx: float | None = None) -> None:
        """눈높이 트림을 갱신한다. 인코더나 아이트래킹이 부르는 진입점."""
        self.trim_dy = float(dy)
        if dx is not None:
            self.trim_dx = float(dx)

    def set_virtual_image_distance(self, d_vi: float) -> None:
        """가상상 거리를 바꾼다. 깊이 가중치를 다시 계산해야 한다."""
        self.d_vi = max(0.05, float(d_vi))
        self.depths = np.maximum(self.depths, self.d_vi + 1e-3)
        self._slot_weights = self._depth_weights(self.depths)

    # 지평선 --------------------------------------------------------------

    def clip_above_horizon(self, lane: Any) -> np.ndarray:
        """지평선 위 구간을 잘라 낸다. 깊이는 같이 살아남는다.

        정규화 좌표에서 y 가 작을수록 멀다. horizon 보다 작은 y 는 노면이
        아니라 하늘이므로 호모그래피에 넣으면 화면 밖으로 발산한다.

        반환값은 (M, 3) [x, y, depth] 배열이고 그대로 project() 에 넣는다.
        """
        array = _as_lane_array(lane, self.depths)
        if array.shape[0] == 0:
            return array

        keep = array[:, 1] >= self.horizon
        if keep.all():
            return array
        if not keep.any():
            return np.zeros((0, 3), np.float32)

        if not self.interpolate_horizon:
            return np.ascontiguousarray(array[keep])

        # 경계를 가로지르는 구간마다 교점을 만들어 끼운다. 그냥 버리면
        # 리본 윗변이 슬롯 간격만큼 뭉텅뭉텅 잘려 프레임마다 튄다.
        pieces: list[np.ndarray] = []
        for index in range(array.shape[0]):
            if keep[index]:
                pieces.append(array[index])
            if index + 1 >= array.shape[0] or keep[index] == keep[index + 1]:
                continue
            first, second = array[index], array[index + 1]
            span = second[1] - first[1]
            if abs(span) < 1e-9:
                continue
            ratio = float((self.horizon - first[1]) / span)
            if not 0.0 <= ratio <= 1.0:
                continue
            pieces.append(first + (second - first) * ratio)
        if len(pieces) < 1:
            return np.zeros((0, 3), np.float32)
        return np.ascontiguousarray(np.vstack(pieces).astype(np.float32))

    # 투영 ----------------------------------------------------------------

    def project(self, lane: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """정규화 노면 좌표를 패널 픽셀로 투영한다.

        반환: (points, depths, valid)
            points  (N, 2) int32   패널 픽셀
            depths  (N,)   float32 각 점의 종방향 거리(m)
            valid   (N,)   bool    쓸 수 있는 점인지

        **세 배열은 길이가 같고 인덱스가 대응한다.** 호출부에서 마스크로
        걸러내거나 정렬할 때는 반드시 세 배열에 똑같이 적용해야 한다.
        points 만 걸러내면 "이 점이 몇 미터짜리였는지"가 사라지고, 깊이
        가중 보정을 나중에 한 번 더 걸 수 없게 된다.
        """
        array = _as_lane_array(lane, self.depths)
        count = array.shape[0]
        if count == 0:
            return (np.zeros((0, 2), np.int32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), dtype=bool))

        depths = np.ascontiguousarray(array[:, 2])
        source = np.ascontiguousarray(array[:, :2]).reshape(-1, 1, 2)
        points = cv2.perspectiveTransform(source, self.homography).reshape(-1, 2)

        finite = np.isfinite(points).all(axis=1)
        if not finite.all():
            points[~finite] = 0.0

        # 깊이 가중 눈높이 보정. 이 한 줄을 위해 깊이를 여기까지 끌고 왔다.
        if self.trim_dy or self.trim_dx:
            if count == self._slot_weights.shape[0]:
                weights = self._slot_weights
            else:
                weights = self._depth_weights(depths)
            if self.trim_dy:
                points[:, 1] += self.trim_dy * weights
            if self.trim_dx:
                points[:, 0] += self.trim_dx * weights

        # 패널 → 반사판 → 눈 경로에서 상이 뒤집히므로 여기서 되돌린다.
        if self.flip_horizontal:
            points[:, 0] = (self.width - 1) - points[:, 0]

        margin_x = self.width * self.valid_margin
        margin_y = self.height * self.valid_margin
        valid = (
            finite
            & (points[:, 0] >= -margin_x) & (points[:, 0] <= self.width + margin_x)
            & (points[:, 1] >= -margin_y) & (points[:, 1] <= self.height + margin_y)
        )

        # int32 캐스팅 전에 잘라 둔다. 지평선 근처 점은 쉽게 1e9 를 넘어
        # 그대로 캐스팅하면 부호가 뒤집힌 좌표가 나온다.
        np.clip(points[:, 0], -margin_x, self.width + margin_x, out=points[:, 0])
        np.clip(points[:, 1], -margin_y, self.height + margin_y, out=points[:, 1])

        return (np.rint(points).astype(np.int32),
                depths.astype(np.float32, copy=False),
                valid)


# ---------------------------------------------------------------------------
# 보정 도구. hud_ui 가 "run hud_align.py pick / aim first" 라고 안내하는 그것.
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        config = {}
    ensure_alignment_config(config)
    return config


def _save_config(path: str, config: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _reference_lanes() -> list[list[list[float]]]:
    """보정용 기준 차선. 정규화 좌표에서 곧게 뻗은 3.5m 폭 차선 한 쌍."""
    rows = np.linspace(1.0, 0.40, len(DEFAULT_DEPTHS), dtype=np.float32)
    # 소실점으로 수렴하는 직선. x 는 y 에 선형으로 좁아진다.
    left = [[float(0.5 - 0.46 * (y - 0.40) / 0.60), float(y)] for y in rows]
    right = [[float(0.5 + 0.46 * (y - 0.40) / 0.60), float(y)] for y in rows]
    return [left, right]


def _draw_reference(mapper: AlignmentMap) -> np.ndarray:
    """기준 차선과 거리 눈금을 그린 화면. 배경은 순수 검정을 유지한다."""
    canvas = np.zeros((mapper.height, mapper.width, 3), np.uint8)
    rails = []
    for lane in _reference_lanes():
        points, depths, valid = mapper.project(mapper.clip_above_horizon(lane))
        points, depths = points[valid], depths[valid]
        if len(points) < 2:
            continue
        rails.append((points, depths))
        cv2.polylines(canvas, [points], False, (232, 214, 79), 2, cv2.LINE_AA)

    if len(rails) == 2:
        (left, left_d), (right, _) = rails
        span = min(len(left), len(right))
        for index in range(span):
            distance = float(left_d[index])
            a = (int(left[index][0]), int(left[index][1]))
            b = (int(right[index][0]), int(right[index][1]))
            cv2.line(canvas, a, b, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{distance:.0f}m", (b[0] + 6, b[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1,
                        cv2.LINE_AA)
    return canvas


def _run_pick(args: Any) -> None:
    """실제 도로 위 차선 사다리꼴이 패널 어디에 보이는지 4점을 찍는다."""
    config = _load_config(args.config)
    mapper = AlignmentMap(config)
    picked: list[list[float]] = []

    def on_mouse(event: int, x: int, y: int, *_: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            picked.append([float(x), float(y)])

    name = "hud_align pick"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, mapper.width, mapper.height)
    cv2.setMouseCallback(name, on_mouse)
    print("클릭 순서: " + " → ".join(QUAD_ORDER))
    print("keys: r 다시  s 저장  q 취소")

    while True:
        canvas = _draw_reference(mapper)
        for index, (x, y) in enumerate(picked):
            cv2.drawMarker(canvas, (int(x), int(y)), (255, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.putText(canvas, QUAD_ORDER[index], (int(x) + 10, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if len(picked) < 4:
            cv2.putText(canvas, f"click {QUAD_ORDER[len(picked)]}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                        cv2.LINE_AA)
        cv2.imshow(name, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            picked.clear()
        elif key == ord("s") and len(picked) == 4:
            config["alignment"]["panel_quad"] = picked
            _save_config(args.config, config)
            print(f"saved panel_quad → {args.config}")
            mapper = AlignmentMap(config)
    cv2.destroyAllWindows()


def _run_aim(args: Any) -> None:
    """눈높이 트림을 실시간으로 맞춘다. 깊이 가중이라 근/원거리가 같이 움직인다."""
    config = _load_config(args.config)
    mapper = AlignmentMap(config)
    name = "hud_align aim"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, mapper.width, mapper.height)
    # 방향키는 백엔드에 따라 코드가 달라서 i/k/j/l 을 정식 키로 둔다.
    print("keys: i/k 상하  j/l 좌우  [ ] 가상상거리  0 리셋  s 저장  q 종료")
    steps = {ord("i"): (-1.0, 0.0), ord("k"): (1.0, 0.0),
             ord("j"): (0.0, -1.0), ord("l"): (0.0, 1.0),
             82: (-1.0, 0.0), 84: (1.0, 0.0), 81: (0.0, -1.0), 83: (0.0, 1.0)}

    while True:
        canvas = _draw_reference(mapper)
        cv2.putText(
            canvas,
            "trim dy {:+.1f}  dx {:+.1f}  d_vi {:.2f}m  {}".format(
                mapper.trim_dy, mapper.trim_dx, mapper.d_vi,
                "cal" if mapper.calibrated else "uncal"),
            (16, mapper.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (160, 160, 160), 1, cv2.LINE_AA)
        cv2.imshow(name, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break
        if key in steps:
            step_dy, step_dx = steps[key]
            mapper.set_trim(mapper.trim_dy + step_dy, mapper.trim_dx + step_dx)
        elif key == ord("0"):
            mapper.set_trim(0.0, 0.0)
        elif key in (ord("["), ord("]")):
            mapper.set_virtual_image_distance(
                mapper.d_vi + (0.05 if key == ord("]") else -0.05))
        elif key == ord("s"):
            config["alignment"]["trim_dy"] = mapper.trim_dy
            config["alignment"]["trim_dx"] = mapper.trim_dx
            config["alignment"]["d_vi"] = mapper.d_vi
            _save_config(args.config, config)
            print(f"saved trim → {args.config}")
    cv2.destroyAllWindows()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="운전자 시점 정렬 보정")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("pick", "노면 사다리꼴 4점 찍기"),
                            ("aim", "눈높이 트림 맞추기")):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", default="hud_config.json")
    args = parser.parse_args()
    if args.command == "pick":
        _run_pick(args)
    else:
        _run_aim(args)


if __name__ == "__main__":
    main()
