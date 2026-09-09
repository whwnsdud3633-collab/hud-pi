#!/usr/bin/env python3
"""반사식 HUD 화면 구성 모듈.

hud_system.py 가 좌표 수신과 전송을 담당하고, 이 파일은 화면에 무엇을 어떻게
그릴지만 담당한다. 레이아웃 값은 모두 hud_config.json 의 "ui" 항목에 들어 있어
코드를 고치지 않고 숫자만 바꿔서 배치를 조정할 수 있다.

단독 실행:
    python3 hud_ui.py preview            # 네트워크 없이 화면 구성만 확인
    python3 hud_ui.py sample --out ./png # PNG 로 저장해서 노트북에서 확인
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
from hud_system import (
    PacketError,
    _decode_packet,
    _mock_lane,
    _select_hud_lanes,
    LaneSmoother,
    load_config,
    save_config,
)


WARNING_STATES = (
    "none",
    "departure_left",
    "departure_right",
    "change_left",
    "change_right",
)

# 젯슨이 보내는 인식 상태. 0 정상, 1 주의(차선이 흐릿함), 2 인식 불가.
# 프로토콜 합의 전이라 패킷에 없을 수 있고, 그때는 0 으로 본다.
LANE_STATES = (0, 1, 2)


def normalize_state(value: Any) -> int:
    """패킷에서 읽은 state 를 0/1/2 로 만든다. 이상하면 0."""
    try:
        state = int(value)
    except (TypeError, ValueError):
        return 0
    return state if state in LANE_STATES else 0


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
        color = tuple(int(v) for v in style["color_bgr"])
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
    ) -> np.ndarray:
        # 지금은 받아서 보관만 한다
        self.state = normalize_state(state)
        canvas = self.static_layer.copy()
        emphasis = None
        if warning.startswith("departure"):
            emphasis = "left" if warning.endswith("_left") else "right"
        self._draw_lanes(canvas, lanes, emphasis)
        self._draw_warning(canvas, warning, elapsed)
        if debug is not None:
            self._draw_debug(canvas, debug)
        return canvas


def make_renderer(config: dict[str, Any], theme: bool):
    """시안 테마와 기본 선 표시 중 하나를 고른다."""
    if theme or str(config.get("ui", {}).get("style", "")) == "ar_overlay":
        from hud_theme import ThemeRenderer
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
# receive: UDP 수신 + 이 파일의 화면 구성
# ---------------------------------------------------------------------------


def run_receive(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_ui_config(config)
    renderer = make_renderer(config, getattr(args, "theme", False))
    display = config["display"]
    network = config["network"]

    bind_host = args.bind_host or str(network["bind_host"])
    port = args.port or int(network["port"])
    timeout = float(network["packet_timeout_seconds"])
    smoother = LaneSmoother(
        alpha=float(display.get("smoothing_alpha", 0.55)),
        association_distance=float(display.get("association_distance", 0.18)),
    )

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind((bind_host, port))
    receiver.setblocking(False)

    name = str(display["window_name"])
    _open_window(name, renderer, args.windowed, bool(display["fullscreen"]))
    print(f"HUD listening on {bind_host}:{port} / {renderer.width}x{renderer.height}")

    latest: dict[str, Any] | None = None
    latest_received = 0.0
    last_sequence = -1
    dropped = 0
    started = time.monotonic()

    try:
        while True:
            readable, _, _ = select.select([receiver], [], [], 0.01)
            if readable:
                payload, address = receiver.recvfrom(65_535)
                try:
                    packet = _decode_packet(payload)
                    if packet["seq"] >= last_sequence:
                        packet["lanes"] = smoother.update(packet["lanes"])
                        latest = packet
                        latest_received = time.monotonic()
                        last_sequence = packet["seq"]
                except (PacketError, ValueError, TypeError):
                    dropped += 1
                    if dropped % 30 == 1:
                        print(f"ignored invalid packet from {address}")

            now = time.monotonic()
            age = now - latest_received if latest is not None else float("inf")
            fresh = latest is not None and age <= timeout

            if not fresh:
                # 송신기가 재시작해 seq 가 0 으로 돌아와도 다시 받아들인다.
                last_sequence = -1
                smoother.reset()

            start_render = time.perf_counter()
            if fresh and latest["status"] != "lost":
                warning = str(latest.get("warning", "none"))
                if warning not in WARNING_STATES:
                    warning = "none"
                state = normalize_state(latest.get("state", 0))
                telemetry = {
                    "confidence": float(latest.get("confidence", 1.0)),
                    "departure_distance": float(latest.get("departure_distance", 0.0)),
                    "lkas": bool(latest.get("lkas", True)),
                    "acc": bool(latest.get("acc", True)),
                    "fps": latest["fps"],
                    "inference_ms": latest["inference_ms"],
                }
                canvas = renderer.render(
                    lanes=latest["lanes"],
                    warning=warning,
                    elapsed=now - started,
                    state=state,
                    telemetry=telemetry,
                    debug=_blank_debug(
                        status=latest["status"],
                        lanes=len(latest["lanes"]),
                        seq=latest["seq"],
                        fps=latest["fps"],
                        inference_ms=latest["inference_ms"],
                        age_ms=age * 1000.0,
                        dropped=dropped,
                        warning=warning,
                    ),
                )
            else:
                canvas = renderer.render(lanes=[], warning="none", telemetry={},
                                         state=0)

            _ = (time.perf_counter() - start_render) * 1000.0
            cv2.imshow(name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                renderer.ui["debug"]["enabled"] = not renderer.ui["debug"]["enabled"]
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        cv2.destroyAllWindows()


def run_bench(args: argparse.Namespace) -> None:
    """실제 장치에서 프레임당 렌더 시간을 잰다. 화면 출력은 하지 않는다."""
    config = load_config(args.config)
    renderer = make_renderer(config, args.theme)
    lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
    telemetry = {"confidence": 0.7, "fps": 22.4, "inference_ms": 44.6,
                 "departure_distance": 0.3}
    for warning in ("none", "departure_left"):
        renderer.render(lanes=lanes, warning=warning, elapsed=0.0, telemetry=telemetry)
        start = time.perf_counter()
        for index in range(args.frames):
            renderer.render(lanes=lanes, warning=warning,
                            elapsed=index * 0.03, telemetry=telemetry)
        each = (time.perf_counter() - start) / args.frames * 1000.0
        print(f"{warning:16s} {each:6.1f} ms/frame   {1000.0 / each:5.1f} fps")
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
    elif args.command == "receive":
        run_receive(args)
    elif args.command == "bench":
        run_bench(args)


if __name__ == "__main__":
    main()
