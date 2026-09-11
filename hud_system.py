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
    },
    "state": {
        "lane_fade_seconds": 0.4,
        "caution_timeout_seconds": 1.2,
        "rise_frames": 3,
        "fall_frames": 5,
    },
    "display": {
        "width": 1280,
        "height": 720,
        "fullscreen": True,
        "window_name": "Rain Lane HUD",
        "line_thickness": 8,
        "line_color_bgr": [255, 255, 255],
        "flip_horizontal": False,
        "flip_vertical": False,
        "show_status": False,
        "smoothing_alpha": 0.55,
        "association_distance": 0.18,
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


def _sanitize_lane(lane: Iterable[Iterable[float]], max_points: int = 48) -> list[list[float]]:
    points: list[list[float]] = []
    for raw_point in lane:
        point = list(raw_point)
        if len(point) != 2 or not _finite(point[0]) or not _finite(point[1]):
            continue
        x = min(1.0, max(0.0, float(point[0])))
        y = min(1.0, max(0.0, float(point[1])))
        points.append([x, y])
    points.sort(key=lambda item: item[1])
    if len(points) > max_points:
        indices = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
        points = [points[index] for index in indices]
    if len(points) < 2:
        return []
    return [[round(x, 5), round(y, 5)] for x, y in points]


def _select_hud_lanes(
    lanes: Iterable[Iterable[Iterable[float]]],
    *,
    lane_change: bool,
) -> list[list[list[float]]]:
    """평상시 2개, 차선 변경 시 최대 3개 차선을 선택한다."""
    valid = [_sanitize_lane(lane) for lane in lanes]
    valid = [lane for lane in valid if lane]
    valid.sort(key=lambda lane: lane[-1][0])
    limit = 3 if lane_change else 2
    if len(valid) <= limit:
        return valid

    bottom_x = [lane[-1][0] for lane in valid]
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
    return [valid[index] for index in sorted(selected)]


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


def _decode_packet(payload: bytes) -> dict[str, Any]:
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
    packet["lanes"] = _select_hud_lanes(raw_lanes, lane_change=lane_change)
    packet["status"] = status
    packet["lane_change"] = lane_change
    packet["fps"] = float(packet.get("fps", 0.0))
    packet["inference_ms"] = float(packet.get("inference_ms", 0.0))
    return packet


class LaneSmoother:
    """아래쪽 x 위치로 차선을 대응시켜 HUD 흔들림을 줄인다."""

    def __init__(self, alpha: float, association_distance: float) -> None:
        self.alpha = float(alpha)
        self.association_distance = float(association_distance)
        self.previous: list[Any] = []

    def reset(self) -> None:
        self.previous = []

    def update(self, lanes: list[list[list[float]]]) -> list[list[list[float]]]:
        import numpy as np

        current = [np.asarray(lane, dtype=np.float32) for lane in lanes]
        current.sort(key=lambda lane: float(lane[-1, 0]))
        unused = set(range(len(self.previous)))
        smoothed = []
        for lane in current:
            best_index = None
            best_distance = float("inf")
            for index in unused:
                distance = abs(float(lane[-1, 0] - self.previous[index][-1, 0]))
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is not None and best_distance <= self.association_distance:
                previous = self.previous[best_index]
                previous_x = np.interp(lane[:, 1], previous[:, 1], previous[:, 0])
                lane[:, 0] = self.alpha * lane[:, 0] + (1.0 - self.alpha) * previous_x
                unused.remove(best_index)
            lane = np.clip(lane, 0.0, 1.0)
            smoothed.append(lane)
        self.previous = smoothed
        return [np.round(lane, 5).tolist() for lane in smoothed]


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
    )

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind((bind_host, port))
    receiver.setblocking(False)

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
                    invalid_packets += 1
                    if invalid_packets % 30 == 1:
                        print(f"ignored invalid packet from {address}")

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


def _mock_lane(bottom_x: float, top_x: float, phase: float, count: int = 24) -> list[list[float]]:
    output = []
    for index in range(count):
        y = 0.38 + 0.60 * index / (count - 1)
        progress = (y - 0.38) / 0.60
        x = top_x + (bottom_x - top_x) * progress**1.45
        x += 0.012 * math.sin(phase + progress * 2.2)
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
            if lane_change:
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
