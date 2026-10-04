#!/usr/bin/env python3
"""반사식 HUD 화면 구성 모듈.

hud_system.py 가 좌표 수신과 전송을 담당하고, 이 파일은 화면에 무엇을 어떻게
그릴지만 담당한다. 레이아웃 값은 모두 hud_config.json 의 "ui" 항목에 들어 있어
코드를 고치지 않고 숫자만 바꿔서 배치를 조정할 수 있다.

단독 실행:
    python3 hud_ui.py preview            # 네트워크 없이 화면 구성만 확인
    python3 hud_ui.py sample --out ./png # PNG 로 저장해서 노트북에서 확인
    python3 hud_ui.py pattern            # 패널 진단용 테스트 패턴
    python3 hud_ui.py receive            # UDP 수신 + 이 파일의 화면 구성
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hud_align import AlignmentMap, ensure_alignment_config
# 인식 상태 정의는 렌더러 쪽 한 곳(hud_theme)에만 둔다
from hud_theme import (
    DESIGN_H,
    DESIGN_W,
    LANE_STATES,
    STATE_COLORS,
    STATE_LOST,
    STATE_NORMAL,
    ThemeRenderer,
    normalize_state,
)
# 무수신 판정, 페이드, 디바운싱은 그리기 코드가 아니라 상태 계층이 맡는다
from hud_state import StateTracker, ensure_state_config
from hud_system import (
    PacketError,
    WARNING_STATES,
    _decode_packet,
    _mock_lane,
    _select_hud_lanes,
    LaneSmoother,
    load_config,
    save_config,
)


DEFAULT_UI: dict[str, Any] = {
    "safe_area": [0.06, 0.06, 0.94, 0.94],
    "lane": {
        "color_bgr": [255, 255, 255],
        "thickness": 8,
        "emphasis_thickness": 13,
    },
    "warning": {
        "enabled": True,
        "color_bgr": [70, 190, 255],
        "chevron_count": 3,
        "chevron_width": 0.045,
        "chevron_height": 0.070,
        "chevron_gap": 0.028,
        "thickness": 7,
        "side_margin": 0.085,
        "center_y": 0.52,
        "blink_hz": 2.4,
        "flow_hz": 1.6,
        "label": True,
        "label_scale": 0.85,
        "label_y": 0.22,
    },
    "debug": {
        "enabled": False,
        "color_bgr": [140, 140, 140],
        "scale": 0.5,
        "thickness": 1,
        "origin": [0.03, 0.06],
        "line_gap": 0.045,
    },
    "center_tick": {
        "enabled": False,
        "color_bgr": [120, 120, 120],
        "width": 0.05,
        "y": 0.95,
        "thickness": 4,
    },
}


def ensure_ui_config(config: dict[str, Any]) -> dict[str, Any]:
    """설정에 ui 항목이 없으면 기본값을 채워 넣는다."""
    ui = config.setdefault("ui", {})
    for section, defaults in DEFAULT_UI.items():
        if not isinstance(defaults, dict):
            ui.setdefault(section, defaults)
            continue
        target = ui.setdefault(section, {})
        for key, value in defaults.items():
            target.setdefault(key, value)
    return ui


class HudUiSender:
    """추론 코드용 송신기. 기존 HudSender 에 warning 항목을 더한다."""

    def __init__(self, host: str, port: int = 5005) -> None:
        self.destination = (host, int(port))
        self.sequence = 0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self.socket.close()

    def send(
        self,
        lanes: Any,
        *,
        warning: str = "none",
        lane_change: bool = False,
        fps: float = 0.0,
        inference_ms: float = 0.0,
        frame_width: int | None = None,
        frame_height: int | None = None,
        confidence: float = 1.0,
        departure_distance: float = 0.0,
        lkas: bool = True,
        acc: bool = True,
    ) -> None:
        if warning not in WARNING_STATES:
            raise ValueError(f"unknown warning: {warning}")
        if frame_width and frame_height:
            w = max(1, int(frame_width) - 1)
            h = max(1, int(frame_height) - 1)
            lanes = [[[float(x) / w, float(y) / h] for x, y in lane] for lane in lanes]
        selected = _select_hud_lanes(lanes, lane_change=lane_change)
        if not selected:
            status = "lost"
        elif len(selected) < 2:
            status = "degraded"
        else:
            status = "ok"
        packet = {
            "v": 1,
            "seq": self.sequence,
            "sent_at": time.time(),
            "status": status,
            "fps": round(float(fps), 2),
            "inference_ms": round(float(inference_ms), 2),
            "lane_change": bool(lane_change),
            "warning": warning,
            "confidence": round(float(confidence), 3),
            "departure_distance": round(float(departure_distance), 2),
            "lkas": bool(lkas),
            "acc": bool(acc),
            "lanes": selected,
        }
        payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self.socket.sendto(payload, self.destination)
        self.sequence += 1

    def __enter__(self) -> "HudUiSender":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class HudRenderer:
    """정규화 좌표를 받아 HUD 한 장을 그린다."""

    def __init__(self, config: dict[str, Any]) -> None:
        display = config["display"]
        self.width = int(display["width"])
        self.height = int(display["height"])
        ensure_alignment_config(config)
        self.mapper = AlignmentMap(config)
        self.ui = ensure_ui_config(config)
        self.rect = (0, 0, self.width, self.height)
        self.static_layer = self._build_static_layer()
        self.state = 0
        self.lane_state = 0
        self.lane_opacity = 1.0

    @property
    def calibrated(self) -> bool:
        """운전자 시점 정렬이 실측으로 잡혀 있는지."""
        return self.mapper.calibrated

    # 좌표 변환 -------------------------------------------------------------

    def _road_points(self, lane: Any) -> tuple[np.ndarray, np.ndarray]:
        """노면 위 차선 좌표를 패널 픽셀로. 지평선 위쪽은 버린다."""
        clipped = self.mapper.clip_above_horizon(lane)
        return self.mapper.project(clipped)

    def _panel_point(self, x: float, y: float) -> tuple[int, int]:
        """화면 고정 요소용. 안전 영역 안의 비율 좌표를 픽셀로 옮긴다."""
        x0, y0, x1, y1 = (float(v) for v in self.ui["safe_area"])
        px = (x0 + x * (x1 - x0)) * (self.width - 1)
        py = (y0 + y * (y1 - y0)) * (self.height - 1)
        return int(round(px)), int(round(py))

    def _draw_clipped_polyline(
        self,
        canvas: np.ndarray,
        points: np.ndarray,
        valid: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """화면 밖으로 나가는 부분을 잘라 내며 선을 잇는다."""
        joint = max(1, thickness // 2)
        for index in range(len(points) - 1):
            if not (valid[index] and valid[index + 1]):
                continue
            first = (int(points[index][0]), int(points[index][1]))
            second = (int(points[index + 1][0]), int(points[index + 1][1]))
            inside, clipped_a, clipped_b = cv2.clipLine(self.rect, first, second)
            if not inside:
                continue
            cv2.line(canvas, clipped_a, clipped_b, color, thickness, cv2.LINE_AA)
            if 0 < index < len(points) - 1:
                cv2.circle(canvas, clipped_a, joint, color, -1, cv2.LINE_AA)

    # 정적 요소 -------------------------------------------------------------

    def _build_static_layer(self) -> np.ndarray:
        """매 프레임 다시 그릴 필요가 없는 요소를 한 번만 그려 캐시한다."""
        layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        tick = self.ui["center_tick"]
        if bool(tick["enabled"]):
            half = float(tick["width"]) / 2.0
            y = float(tick["y"])
            left = self._panel_point(0.5 - half, y)
            right = self._panel_point(0.5 + half, y)
            cv2.line(
                layer,
                left,
                right,
                tuple(int(v) for v in tick["color_bgr"]),
                int(tick["thickness"]),
                cv2.LINE_AA,
            )
        return layer

    def rebuild(self, config: dict[str, Any]) -> None:
        """보정값이나 레이아웃이 바뀐 뒤 캐시를 다시 만든다."""
        self.__init__(config)

    # 개별 요소 -------------------------------------------------------------

    def _draw_lanes(
        self,
        canvas: np.ndarray,
        lanes: list[list[list[float]]],
        emphasis_side: str | None,
    ) -> None:
        style = self.ui["lane"]
        color = tuple(int(v * self.lane_opacity) for v in style["color_bgr"])
        base = int(style["thickness"])
        strong = int(style["emphasis_thickness"])
        for lane in lanes:
            if len(lane) < 2:
                continue
            bottom_x = float(lane[-1][0])
            thickness = base
            if emphasis_side == "left" and bottom_x <= 0.5:
                thickness = strong
            elif emphasis_side == "right" and bottom_x > 0.5:
                thickness = strong
            points, valid = self._road_points(lane)
            if len(points) < 2:
                continue
            self._draw_clipped_polyline(canvas, points, valid, color, thickness)

    def _chevron_points(
        self, center_x: float, center_y: float, pointing: str
    ) -> np.ndarray:
        """꺾쇠 하나의 세 점을 정규화 좌표로 만든다."""
        style = self.ui["warning"]
        half_w = float(style["chevron_width"]) / 2.0
        half_h = float(style["chevron_height"]) / 2.0
        tip_x = center_x + (half_w if pointing == "right" else -half_w)
        back_x = center_x - (half_w if pointing == "right" else -half_w)
        return np.asarray(
            [
                [back_x, center_y - half_h],
                [tip_x, center_y],
                [back_x, center_y + half_h],
            ],
            dtype=np.float32,
        )

    def _draw_warning(
        self, canvas: np.ndarray, warning: str, elapsed: float
    ) -> None:
        style = self.ui["warning"]
        if not bool(style["enabled"]) or warning == "none":
            return

        side = "left" if warning.endswith("_left") else "right"
        departure = warning.startswith("departure")
        # 이탈은 중앙으로 돌아오라는 뜻이라 안쪽을, 변경은 진행 방향인 바깥쪽을 가리킨다.
        if departure:
            pointing = "right" if side == "left" else "left"
        else:
            pointing = "left" if side == "left" else "right"

        if departure and math.sin(elapsed * 2.0 * math.pi * float(style["blink_hz"])) < 0:
            return

        color = tuple(int(v) for v in style["color_bgr"])
        thickness = int(style["thickness"])
        count = int(style["chevron_count"])
        gap = float(style["chevron_gap"]) + float(style["chevron_width"])
        margin = float(style["side_margin"])
        center_y = float(style["center_y"])
        base_x = margin if side == "left" else 1.0 - margin
        half = (count - 1) / 2.0

        active = -1
        if not departure:
            active = int(elapsed * float(style["flow_hz"]) * count) % count
            if pointing == "left":
                active = count - 1 - active

        for index in range(count):
            center_x = base_x + (index - half) * gap
            points = self._chevron_points(center_x, center_y, pointing)
            weight = thickness
            if active >= 0 and index != active:
                weight = max(2, thickness - 4)
            panel = np.asarray(
                [self._panel_point(px, py) for px, py in points], dtype=np.int32
            )
            cv2.polylines(canvas, [panel], False, color, weight, cv2.LINE_AA)

        if bool(style["label"]):
            text = "DEPARTURE" if departure else "LANE CHANGE"
            self._draw_centered_text(
                canvas,
                text,
                float(style["label_y"]),
                float(style["label_scale"]),
                color,
                2,
            )

    def _draw_centered_text(
        self,
        canvas: np.ndarray,
        text: str,
        y: float,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        anchor_x, anchor_y = self._panel_point(0.5, y)
        origin = (int(anchor_x - text_w / 2), int(anchor_y + text_h / 2))
        cv2.putText(canvas, text, origin, font, scale, color, thickness, cv2.LINE_AA)

    def _draw_debug(self, canvas: np.ndarray, info: dict[str, Any]) -> None:
        style = self.ui["debug"]
        if not bool(style["enabled"]):
            return
        color = tuple(int(v) for v in style["color_bgr"])
        scale = float(style["scale"])
        thickness = int(style["thickness"])
        origin_x, origin_y = style["origin"]
        gap = float(style["line_gap"])
        lines = [
            "STATUS {status}  LANES {lanes}  SEQ {seq}".format(**info),
            "FPS {fps:.1f}  INFER {inference_ms:.1f}ms  AGE {age_ms:.0f}ms".format(**info),
            "RENDER {render_ms:.1f}ms  DROP {dropped}  WARN {warning}".format(**info),
            f"STATE {self.state}",
            "ALIGN {align}".format(align="driver-calibrated" if self.mapper.calibrated else "UNCALIBRATED (quad fallback)"),
        ]
        for index, line in enumerate(lines):
            point = self._panel_point(float(origin_x), float(origin_y) + gap * index)
            cv2.putText(
                canvas,
                line,
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    # 한 프레임 -------------------------------------------------------------

    def render(
        self,
        *,
        lanes: list[list[list[float]]],
        warning: str = "none",
        elapsed: float = 0.0,
        debug: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        state: int = 0,
        lane_state: int | None = None,
        lane_opacity: float = 1.0,
    ) -> np.ndarray:
        # 지금은 받아서 보관만 한다
        self.state = normalize_state(state)
        # 차선 색과 밝기는 상태 바 state 와 따로 받는다. ThemeRenderer 와
        # 인터페이스를 맞춰 두어야 호출부가 렌더러를 갈아끼울 수 있다.
        self.lane_state = (
            self.state if lane_state is None else normalize_state(lane_state)
        )
        self.lane_opacity = float(min(1.0, max(0.0, lane_opacity)))
        canvas = self.static_layer.copy()
        emphasis = None
        if warning.startswith("departure"):
            emphasis = "left" if warning.endswith("_left") else "right"
        if self.lane_state != STATE_LOST and self.lane_opacity > 1.0 / 255.0:
            self._draw_lanes(canvas, lanes, emphasis)
        self._draw_warning(canvas, warning, elapsed)
        if debug is not None:
            self._draw_debug(canvas, debug)
        return canvas


def make_renderer(config: dict[str, Any], theme: bool):
    """시안 테마와 기본 선 표시 중 하나를 고른다."""
    if theme or str(config.get("ui", {}).get("style", "")) == "ar_overlay":
        return ThemeRenderer(config)
    return HudRenderer(config)


def _blank_debug(**overrides: Any) -> dict[str, Any]:
    info = {
        "status": "ok",
        "lanes": 0,
        "seq": 0,
        "fps": 0.0,
        "inference_ms": 0.0,
        "age_ms": 0.0,
        "render_ms": 0.0,
        "dropped": 0,
        "warning": "none",
    }
    info.update(overrides)
    return info


def _open_window(name: str, renderer: HudRenderer, windowed: bool, fullscreen: bool) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen and not windowed:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(name, renderer.width, renderer.height)


# ---------------------------------------------------------------------------
# preview: 네트워크도 모델도 없이 화면 구성만 조정하는 모드
# ---------------------------------------------------------------------------


def run_preview(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    renderer.ui["debug"]["enabled"] = True
    if not renderer.calibrated:
        print(
            "warning: no driver-viewpoint alignment saved. lanes will not line up\n"
            "         with the real road. run hud_align.py pick / aim first."
        )

    name = "HUD preview"
    _open_window(name, renderer, args.windowed, bool(config["display"]["fullscreen"]))
    print(
        "preview keys: W warning  L lanes  D debug  H flip  [ ] thickness  S save  Q quit"
    )

    warning_index = 0
    three_lanes = False
    state = normalize_state(getattr(args, "state", 0))
    announced = False
    started = time.monotonic()

    while True:
        elapsed = time.monotonic() - started
        phase = elapsed * 0.8
        if three_lanes:
            lanes = [
                _mock_lane(0.14, 0.42, phase),
                _mock_lane(0.50, 0.54, phase + 0.2),
                _mock_lane(0.86, 0.68, phase + 0.4),
            ]
        else:
            lanes = [
                _mock_lane(0.18, 0.43, phase),
                _mock_lane(0.82, 0.57, phase + 0.3),
            ]

        warning = WARNING_STATES[warning_index]
        start_render = time.perf_counter()
        canvas = renderer.render(
            lanes=lanes,
            warning=warning,
            elapsed=elapsed,
            state=state,
            telemetry={"confidence": 0.7, "fps": 22.0, "inference_ms": 44.6,
                       "departure_distance": 0.3},
            debug=_blank_debug(
                status="preview",
                lanes=len(lanes),
                seq=int(elapsed * 22),
                fps=22.0,
                inference_ms=44.6,
                warning=warning,
            ),
        )
        render_ms = (time.perf_counter() - start_render) * 1000.0
        if not announced:
            announced = True
            print(f"renderer state {renderer.state}")
        cv2.putText(
            canvas,
            f"render {render_ms:.1f}ms",
            (12, renderer.height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(name, canvas)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("w"):
            warning_index = (warning_index + 1) % len(WARNING_STATES)
        elif key == ord("l"):
            three_lanes = not three_lanes
        elif key == ord("d"):
            renderer.ui["debug"]["enabled"] = not renderer.ui["debug"]["enabled"]
        elif key == ord("h"):
            config["display"]["flip_horizontal"] = not config["display"]["flip_horizontal"]
            renderer.rebuild(config)
            renderer.ui["debug"]["enabled"] = True
        elif key == ord("["):
            config["ui"]["lane"]["thickness"] = max(
                2, int(config["ui"]["lane"]["thickness"]) - 1
            )
            renderer.rebuild(config)
            renderer.ui["debug"]["enabled"] = True
        elif key == ord("]"):
            config["ui"]["lane"]["thickness"] = min(
                24, int(config["ui"]["lane"]["thickness"]) + 1
            )
            renderer.rebuild(config)
            renderer.ui["debug"]["enabled"] = True
        elif key == ord("s"):
            save_config(args.config, config)
            print(f"saved: {args.config}")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# sample: PNG 로 저장해 노트북에서 확인
# ---------------------------------------------------------------------------


def run_sample(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    renderer.ui["debug"]["enabled"] = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = normalize_state(getattr(args, "state", 0))

    scenes = {
        "01_normal": ("none", False),
        "02_departure_left": ("departure_left", False),
        "03_change_right": ("change_right", True),
        "04_lost": ("none", None),
    }
    for name, (warning, three) in scenes.items():
        if three is None:
            lanes: list[Any] = []
        elif three:
            lanes = [
                _mock_lane(0.14, 0.42, 0.0),
                _mock_lane(0.50, 0.54, 0.2),
                _mock_lane(0.86, 0.68, 0.4),
            ]
        else:
            lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
        canvas = renderer.render(
            lanes=lanes,
            warning=warning,
            elapsed=0.0,
            state=state,
            telemetry={"confidence": 0.7, "fps": 22.4, "inference_ms": 44.6,
                       "departure_distance": 0.3},
            debug=_blank_debug(
                status="lost" if three is None else "ok",
                lanes=len(lanes),
                seq=120,
                fps=22.4,
                inference_ms=44.6,
                warning=warning,
            ),
        )
        path = out / f"{name}.png"
        cv2.imwrite(str(path), canvas)
        print(f"wrote {path}  state {renderer.state}")


# ---------------------------------------------------------------------------
# pattern: 패널이 제대로 그려지는지 눈으로 보는 진단 패턴
# ---------------------------------------------------------------------------
#
# 배경이 검정이고 mock 차선이 좌우 대칭이라 모니터를 바꿨을 때 레터박스가
# 있는지, 반전이 걸렸는지 알 수가 없다. 그걸 한 화면에서 판정한다.
#
#   화면 경계(흰 1px)와 렌더 영역 경계(하늘색)가 붙어 있으면 레터박스 없음.
#   벌어진 만큼이 레터박스다. 16:10 패널이면 위아래로 벌어진다.
#   L/R/TOP/BOTTOM 글자와 좌상 → 우하 대각선이 비대칭 기준이다. 글자가
#   거울상이면 그 축으로 반전이 걸려 있다.

PATTERN_GRID_PX = 100

# BGR. 요소마다 색을 달리 둬야 겹친 자리를 구분할 수 있다. 반사식 제약대로
# 전부 얇은 선과 글리프고, 큰 밝은 면적은 없다.
PATTERN_COLORS = {
    "border": (255, 255, 255),     # 흰색     화면 경계
    "design": (255, 255, 0),       # 하늘색   렌더(디자인 960x540) 영역 경계
    "grid": (70, 70, 70),          # 짙은 회색 격자
    "diagonal": (0, 255, 255),     # 노란색   비대칭 기준 대각선
    "cross": (120, 255, 120),      # 녹색     중심 십자
    "label": (255, 255, 255),      # 흰색     방향 글자
    "info": (0, 170, 255),         # 주황색   반전을 타지 않는 안내 블록
}


def _pattern_text(
    canvas: np.ndarray,
    text: str,
    center: tuple[float, float],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """문자열을 중심 좌표에 맞춰 그린다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = (int(round(center[0] - text_w / 2)), int(round(center[1] + text_h / 2)))
    cv2.putText(canvas, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _flip_code(flip_h: bool, flip_v: bool) -> int | None:
    """cv2.flip 코드. 반전이 없으면 None."""
    if flip_h and flip_v:
        return -1
    if flip_h:
        return 1
    if flip_v:
        return 0
    return None


def _lane_widths(renderer: ThemeRenderer) -> list[tuple[str, int]]:
    """차선이 실제로 그려지는 굵기 세 가지를 픽셀로."""
    return [
        (name, renderer._ratio_px(renderer.theme[key]))
        for name, key in (
            ("EDGE", "edge_width_ratio"),
            ("ALERT", "alert_edge_width_ratio"),
            ("DIM", "dim_edge_width_ratio"),
        )
    ]


def _draw_pattern_body(
    renderer: ThemeRenderer, canvas: np.ndarray, grid: bool
) -> None:
    """반전을 타는 쪽. 차선과 같은 방향으로 뒤집혀야 하는 것만 여기 그린다."""
    width, height = renderer.width, renderer.height
    scale = renderer.scale
    last_x, last_y = width - 1, height - 1

    # 5. 격자 100px. 가장 어두운 색으로 맨 아래에 깐다.
    if grid:
        for x in range(0, width, PATTERN_GRID_PX):
            cv2.line(canvas, (x, 0), (x, last_y), PATTERN_COLORS["grid"], 1)
        for y in range(0, height, PATTERN_GRID_PX):
            cv2.line(canvas, (0, y), (last_x, y), PATTERN_COLORS["grid"], 1)

    # 2. 렌더 영역 경계. 디자인 좌표 960x540 이 패널에서 차지하는 사각형이다.
    #    ThemeRenderer 가 차선을 올릴 때 쓰는 _dx/_dy 를 그대로 쓴다.
    dx0, dy0 = renderer._dp(0.0, 0.0)
    dx1 = min(last_x, int(round(renderer._dx(DESIGN_W))) - 1)
    dy1 = min(last_y, int(round(renderer._dy(DESIGN_H))) - 1)
    cv2.rectangle(canvas, (dx0, dy0), (dx1, dy1), PATTERN_COLORS["design"], 2)

    # 4. 비대칭 기준선. 디자인 좌표의 좌상 → 우하.
    cv2.line(canvas, (dx0, dy0), (dx1, dy1), PATTERN_COLORS["diagonal"], 2,
             cv2.LINE_AA)

    # 6. 중심 십자. 패널 중심이지 디자인 중심이 아니다. 둘이 어긋나면
    #    렌더 영역 사각형의 대각선 교점과 벌어진 만큼이 보인다.
    cx, cy = last_x // 2, last_y // 2
    arm = int(40 * scale)
    cv2.line(canvas, (cx - arm, cy), (cx + arm, cy), PATTERN_COLORS["cross"], 1,
             cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - arm), (cx, cy + arm), PATTERN_COLORS["cross"], 1,
             cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(10 * scale), PATTERN_COLORS["cross"], 1,
               cv2.LINE_AA)

    # 3. 방향 글자. 디자인 좌표 기준이라 반전이 걸리면 거울상으로 보인다.
    #    TOP/BOTTOM 은 디자인 위아래 끝에 바짝 붙인다. 가운데로 내려오면
    #    반전됐을 때 아래쪽 안내 블록 자리로 들어간다.
    big = 1.1 * scale
    label = PATTERN_COLORS["label"]
    _pattern_text(canvas, "L", renderer._dp(40.0, DESIGN_H / 2), big, label, 3)
    _pattern_text(canvas, "R", renderer._dp(DESIGN_W - 40.0, DESIGN_H / 2), big,
                  label, 3)
    _pattern_text(canvas, "TOP", renderer._dp(DESIGN_W / 2, 28.0), big, label, 3)
    _pattern_text(canvas, "BOTTOM", renderer._dp(DESIGN_W / 2, DESIGN_H - 28.0),
                  big, label, 3)

    # 차선 굵기 샘플. 색은 state 0 차선과 같게 두고 굵기만 바꿔야 굵기
    # 하나만 보고 판단할 수 있다. 선 끝 모양도 폴리라인과 같게 LINE_AA.
    # 패널 좌표의 가장자리 쪽 세로 중앙 띠에 둔다. 반전이 걸리면 좌우로
    # 자리를 옮길 뿐 안내 블록이 있는 위쪽 띠로는 내려오지 않는다.
    x0 = int(round(width * 0.08))
    x1 = int(round(width * 0.26))
    for index, (name, pixels) in enumerate(_lane_widths(renderer)):
        y = int(round(height * (0.38 + 0.12 * index)))
        cv2.line(canvas, (x0, y), (x1, y), STATE_COLORS[STATE_NORMAL], pixels,
                 cv2.LINE_AA)
        _pattern_text(canvas, f"{name} {pixels}px",
                      ((x0 + x1) / 2.0, y - 24.0 * scale), 0.42 * scale, label, 2)


def _draw_pattern_frame(renderer: ThemeRenderer, canvas: np.ndarray) -> None:
    """반전을 타지 않는 쪽.

    1. 화면 경계는 패널 전체 사각형이라 반전해도 제자리다. 반전 뒤에 그려
       1px 이 뒤집기 과정에서 잘리지 않게 한다.
    2. 안내 글자는 반전되면 거울상이 되어 읽을 수가 없다. 패턴의 다른
       글자와 구분되도록 색을 따로 쓰고 NOT FLIPPED 라고 못박는다.
    """
    width, height = renderer.width, renderer.height
    scale = renderer.scale
    mapper = renderer.mapper

    # 1. 화면 경계 1px. 네 변이 다 보이면 오버스캔 없음.
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1),
                  PATTERN_COLORS["border"], 1)

    # 설정값과 패턴이 실제로 탄 반전을 따로 적는다. 대응쌍 보정이 잡혀
    # 있으면 반전이 행렬에 흡수되어 둘이 갈라진다.
    def onoff(flag: bool) -> str:
        return "ON " if flag else "off"

    config_flip = (mapper.physical_flip_horizontal, mapper.physical_flip_vertical)
    applied_flip = (mapper.flip_horizontal, mapper.flip_vertical)
    widths = "  ".join(f"{name.lower()} {px}px" for name, px in _lane_widths(renderer))

    lines = [
        "TEST PATTERN -- THIS BLOCK IS NOT FLIPPED",
        f"PANEL {width}x{height}   DESIGN {int(DESIGN_W)}x{int(DESIGN_H)}"
        f" -> scale {renderer.scale:.3f}"
        f"  letterbox {renderer.offset_x:.0f},{renderer.offset_y:.0f}px",
        f"FLIP config  H {onoff(config_flip[0])} V {onoff(config_flip[1])}"
        f"    pattern rode  H {onoff(applied_flip[0])} V {onoff(applied_flip[1])}",
        f"ALIGN {'correspondence (flip absorbed in matrix)' if mapper.calibrated else 'quad (uncalibrated)'}",
        f"LANE WIDTH  {widths}",
        "mirrored L/R or TOP/BOTTOM above = that axis is flipped",
    ]
    # 패턴이 비워 둔 위쪽 띠. 굵기 샘플(0.38H 아래)과 TOP/BOTTOM 글자
    # (디자인 끝단) 사이라 어느 쪽으로 반전되든 겹치지 않는다.
    text_scale = 0.42 * scale
    gap = int(21 * scale)
    top = int(height * 0.14)
    for index, line in enumerate(lines):
        _pattern_text(canvas, line, (width / 2.0, float(top + index * gap)),
                      text_scale, PATTERN_COLORS["info"], 2)


def build_test_pattern(renderer: ThemeRenderer, *, grid: bool = True) -> np.ndarray:
    """진단 패턴 한 장. 매 프레임 다시 그릴 이유가 없어 한 번만 만든다."""
    canvas = np.zeros((renderer.height, renderer.width, 3), np.uint8)
    _draw_pattern_body(renderer, canvas, grid)

    # 차선은 AlignmentMap.project() 안에서 x -> (w-1)-x, y -> (h-1)-y 를
    # 받는다. 프레임 전체 cv2.flip 이 그와 완전히 같은 변환이므로 패턴도
    # 차선과 같은 방향으로 뒤집힌다. pan 과 trim 은 태우지 않는다. 화면
    # 경계와 렌더 영역을 재는 패턴이라 평행이동이 끼면 기준이 흔들린다.
    code = _flip_code(renderer.mapper.flip_horizontal, renderer.mapper.flip_vertical)
    if code is not None:
        canvas = cv2.flip(canvas, code)

    _draw_pattern_frame(renderer, canvas)
    return canvas


def run_pattern(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    # 차선을 실제로 그리는 렌더러여야 디자인 좌표 매핑과 굵기가 맞는다.
    renderer = ThemeRenderer(config)
    grid = not args.no_grid
    canvas = build_test_pattern(renderer, grid=grid)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
        print(f"wrote {out}  {renderer.width}x{renderer.height}")
        return

    name = "HUD test pattern"
    _open_window(name, renderer, args.windowed, bool(config["display"]["fullscreen"]))
    print("pattern keys: G grid  Q quit")
    while True:
        cv2.imshow(name, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("g"):
            grid = not grid
            canvas = build_test_pattern(renderer, grid=grid)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# receive: UDP 수신 + 이 파일의 화면 구성
# ---------------------------------------------------------------------------


class LaneFeed:
    """UDP 수신 -> 스무딩 -> 상태 계층까지. 렌더러에 넘길 인자를 만든다.

    receive 와 hud_align trim 이 같은 경로로 실데이터를 받도록 한 곳에
    둔다. 두 곳이 따로 판정하면 보정할 때 본 화면과 주행 때 화면이 달라진다.
    """

    def __init__(self, config: dict[str, Any], bind_host: str, port: int) -> None:
        ensure_state_config(config)
        display = config["display"]
        self.smoother = LaneSmoother(
            alpha=float(display.get("smoothing_alpha", 0.55)),
            association_distance=float(display.get("association_distance", 0.18)),
            jump_gate_distance=float(display.get("jump_gate_distance", 0.05)),
            jump_gate_frames=int(display.get("jump_gate_frames", 3)),
        )
        # 무수신 판정과 state 디바운싱은 전부 여기에 있다. 이 루프는 판정을
        # 하지 않고 결과만 받아 그린다.
        self.tracker = StateTracker(config)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((bind_host, port))
        self.socket.setblocking(False)

        self.latest: dict[str, Any] | None = None
        self.latest_received: float | None = None
        self.last_sequence = -1
        self.dropped = 0

    def close(self) -> None:
        self.socket.close()

    def poll(self, timeout: float) -> None:
        """패킷을 최대 한 장 받는다. 없으면 timeout 만큼 기다리고 돌아온다."""
        readable, _, _ = select.select([self.socket], [], [], timeout)
        if not readable:
            return
        payload, address = self.socket.recvfrom(65_535)
        try:
            packet = _decode_packet(payload)
            if packet["seq"] >= self.last_sequence:
                packet["lanes"] = self.smoother.update(
                    packet["lanes"], packet.get("lane_ids")
                )
                self.latest = packet
                self.latest_received = time.monotonic()
                self.last_sequence = packet["seq"]
        except (PacketError, ValueError, TypeError):
            self.dropped += 1
            if self.dropped % 30 == 1:
                print(f"ignored invalid packet from {address}")

    def frame(self, now: float) -> dict[str, Any]:
        """이번 프레임에 그릴 것. elapsed 를 뺀 render() 키워드 인자다."""
        latest = self.latest
        age = now - self.latest_received if self.latest_received is not None else float("inf")

        raw_state = STATE_NORMAL
        confidence = 1.0
        if latest is not None:
            raw_state = normalize_state(latest.get("state", STATE_NORMAL))
            if latest["status"] == "lost":
                # 차선을 못 봤다는 뜻이라 state 2 와 같은 이야기다
                raw_state = STATE_LOST
            confidence = float(latest.get("confidence", 1.0))
        status = self.tracker.update(
            now,
            raw_state=raw_state,
            confidence=confidence,
            last_received=self.latest_received,
        )

        if status.link_lost:
            # 송신기가 재시작해 seq 가 0 으로 돌아와도 다시 받아들인다.
            self.last_sequence = -1
            self.smoother.reset()

        warning = "none"
        telemetry: dict[str, Any] = {}
        info = _blank_debug(
            status="timeout", dropped=self.dropped, age_ms=min(age, 9.999) * 1000.0
        )
        if latest is not None and not status.link_lost:
            warning = str(latest.get("warning", "none"))
            if warning not in WARNING_STATES:
                warning = "none"
            # confidence 는 상태 계층이 이미 state 에 반영했다. 렌더러에
            # 다시 넘기면 디바운싱을 건너뛴 판정이 한 번 더 붙는다.
            telemetry = {
                "departure_distance": float(latest.get("departure_distance", 0.0)),
                "lkas": bool(latest.get("lkas", True)),
                "acc": bool(latest.get("acc", True)),
                "fps": latest["fps"],
                "inference_ms": latest["inference_ms"],
            }
            info = _blank_debug(
                status=latest["status"],
                lanes=len(latest["lanes"]),
                seq=latest["seq"],
                fps=latest["fps"],
                inference_ms=latest["inference_ms"],
                age_ms=age * 1000.0,
                dropped=self.dropped,
                warning=warning,
            )
        # 페이드가 남아 있는 동안은 마지막으로 받은 좌표를 그대로 어둡게
        # 깔아 둔다. 뚝 끊기는 것보다 사라지는 편이 덜 놀랍다.
        lanes: list[Any] = []
        if latest is not None and status.lanes_visible:
            lanes = latest["lanes"]

        return dict(
            lanes=lanes,
            warning=warning,
            state=status.state,
            lane_state=status.lane_state,
            lane_opacity=status.lane_opacity,
            telemetry=telemetry,
            debug=info,
        )


def run_receive(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    display = config["display"]
    network = config["network"]

    bind_host = args.bind_host or str(network["bind_host"])
    port = args.port or int(network["port"])
    feed = LaneFeed(config, bind_host, port)

    name = str(display["window_name"])
    _open_window(name, renderer, args.windowed, bool(display["fullscreen"]))
    print(f"HUD listening on {bind_host}:{port} / {renderer.width}x{renderer.height}")

    started = time.monotonic()

    try:
        while True:
            feed.poll(0.01)
            now = time.monotonic()
            canvas = renderer.render(elapsed=now - started, **feed.frame(now))

            cv2.imshow(name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                renderer.ui["debug"]["enabled"] = not renderer.ui["debug"]["enabled"]
    except KeyboardInterrupt:
        pass
    finally:
        feed.close()
        cv2.destroyAllWindows()


def run_bench(args: argparse.Namespace) -> None:
    """실제 장치에서 프레임당 렌더 시간을 잰다. 화면 출력은 하지 않는다."""
    config = load_config(args.config)
    renderer = make_renderer(config, args.theme)
    lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
    telemetry = {"confidence": 0.7, "fps": 22.4, "inference_ms": 44.6,
                 "departure_distance": 0.3}
    # 부트 애니메이션이 도는 동안은 선이 짧아 측정이 후하게 나온다. 먼저 태운다.
    warm = time.perf_counter()
    while time.perf_counter() - warm < 1.4:
        renderer.render(lanes=lanes, warning="none", elapsed=0.0, telemetry=telemetry)

    # state 마다 그리는 양이 다르다. 2 는 차선을 아예 안 그린다.
    # 무수신 페이드는 상태 바가 2 인 채로 차선을 계속 그리므로 따로 잰다.
    cases = (
        ("none", 0, None, 1.0),
        ("none", 1, None, 1.0),
        ("none", 2, None, 1.0),
        ("none", 2, 0, 0.5),
        ("departure_left", 0, None, 1.0),
    )
    for warning, state, lane_state, opacity in cases:
        kwargs = dict(lanes=lanes, warning=warning, state=state,
                      lane_state=lane_state, lane_opacity=opacity,
                      telemetry=telemetry)
        renderer.render(elapsed=0.0, **kwargs)
        start = time.perf_counter()
        for index in range(args.frames):
            renderer.render(elapsed=index * 0.03, **kwargs)
        each = (time.perf_counter() - start) / args.frames * 1000.0
        label = "fade" if lane_state is not None else ""
        print(f"{warning:16s} state {state} {label:5s} {each:6.1f} ms/frame   "
              f"{1000.0 / each:5.1f} fps")
    print("\ntarget: keep this under 33 ms for 30 fps.")
    print("if it is slower, lower theme.edge_width_ratio.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="반사식 HUD 화면 구성")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="네트워크 없이 화면 구성 조정")
    preview.add_argument("--config", type=Path, default=Path("hud_config.json"))
    preview.add_argument("--windowed", action="store_true")
    preview.add_argument("--theme", action="store_true", help="AR 오버레이 시안으로")
    preview.add_argument("--state", type=int, choices=LANE_STATES, default=0,
                         help="인식 상태 강제 지정. 0 정상 1 주의 2 인식 불가")

    sample = subparsers.add_parser("sample", help="상황별 PNG 저장")
    sample.add_argument("--config", type=Path, default=Path("hud_config.json"))
    sample.add_argument("--out", default="hud_samples")
    sample.add_argument("--theme", action="store_true", help="AR 오버레이 시안으로")
    sample.add_argument("--state", type=int, choices=LANE_STATES, default=0,
                        help="인식 상태 강제 지정. 0 정상 1 주의 2 인식 불가")

    pattern = subparsers.add_parser(
        "pattern", help="레터박스·반전·굵기 확인용 진단 패턴"
    )
    pattern.add_argument("--config", type=Path, default=Path("hud_config.json"))
    pattern.add_argument("--windowed", action="store_true")
    pattern.add_argument("--no-grid", action="store_true", help="격자 없이")
    pattern.add_argument("--out", type=Path, help="화면 대신 PNG 로 저장")

    receive = subparsers.add_parser("receive", help="UDP 수신 + 화면 구성")
    receive.add_argument("--config", type=Path, default=Path("hud_config.json"))
    receive.add_argument("--bind-host")
    receive.add_argument("--port", type=int)
    receive.add_argument("--windowed", action="store_true")
    receive.add_argument("--theme", action="store_true", help="AR 오버레이 시안으로")

    bench = subparsers.add_parser("bench", help="이 장치에서 렌더 속도 측정")
    bench.add_argument("--config", type=Path, default=Path("hud_config.json"))
    bench.add_argument("--theme", action="store_true")
    bench.add_argument("--frames", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preview":
        run_preview(args)
    elif args.command == "sample":
        run_sample(args)
    elif args.command == "pattern":
        run_pattern(args)
    elif args.command == "receive":
        run_receive(args)
    elif args.command == "bench":
        run_bench(args)


if __name__ == "__main__":
    main()
