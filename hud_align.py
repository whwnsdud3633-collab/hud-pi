#!/usr/bin/env python3
"""운전자 시점 정렬(alignment) 계산과 보정 도구.

카메라가 본 차선과 운전자가 눈으로 본 차선이 같은 자리에 겹치도록,
카메라 영상 좌표를 HUD 패널 픽셀로 옮기는 3x3 행렬을 실측으로 구한다.

차선은 노면 위에 있고 노면은 평면이므로
    카메라 영상 -> 노면 -> 운전자 시점 -> HUD 패널
전체가 호모그래피 하나로 압축된다. 카메라 장착 위치, 눈 높이, 반사 각도,
좌우 반전이 모두 이 행렬 안에 흡수되므로 따로 계산하지 않는다.

단계:
    python3 hud_align.py pick   --frame road.png   # 영상에서 표식 찍기
    python3 hud_align.py aim                       # 운전석에서 HUD 점 맞추기
    python3 hud_align.py verify                    # 겹치는지 확인
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hud_system import load_config, save_config

MIN_PAIRS = 4
MAX_PAIRS = 10

DEFAULT_ALIGNMENT: dict[str, Any] = {
    "mode": "quad",
    "camera_horizon_y": 0.0,
    "horizon_margin": 0.18,
    "pan_x": 0,
    "pan_y": 0,
    "pairs": [],
    "profiles": [],
    "blend": 0.0,
}


def ensure_alignment_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.setdefault("alignment", {})
    for key, value in DEFAULT_ALIGNMENT.items():
        section.setdefault(key, value)
    return section


class AlignmentMap:
    """카메라 정규화 좌표를 HUD 패널 픽셀로 옮긴다."""

    def __init__(self, config: dict[str, Any]) -> None:
        display = config["display"]
        self.width = int(display["width"])
        self.height = int(display["height"])
        section = ensure_alignment_config(config)
        self.mode = str(section["mode"])
        self.horizon = float(section["camera_horizon_y"])
        self.pan = (int(section["pan_x"]), int(section["pan_y"]))
        self.pairs = list(section["pairs"])
        self.flip_horizontal = bool(display.get("flip_horizontal", False))
        self.flip_vertical = bool(display.get("flip_vertical", False))

        self.horizon_margin = float(section.get("horizon_margin", 0.18))
        self.profiles = list(section.get("profiles", []))
        self.blend = float(section.get("blend", 0.0))
        self.pairs = blended_pairs(self.pairs, self.profiles, self.blend)
        if self.mode == "correspondence" and len(self.pairs) >= MIN_PAIRS:
            self.matrix = solve_alignment(self.pairs)
            self.w_floor = vanishing_floor(
                self.pairs, self.matrix, self.horizon_margin
            )
            # 대응쌍에는 반사로 인한 좌우 반전이 이미 포함되어 있다.
            self.flip_horizontal = False
            self.flip_vertical = False
            self.calibrated = True
        else:
            self.matrix = quad_matrix(
                config["destination_quad_normalized"], self.width, self.height
            )
            self.w_floor = 1e-6
            self.calibrated = False

    def project(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """정규화 좌표 배열을 패널 픽셀과 유효 여부로 돌려준다."""
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if array.size == 0:
            return np.zeros((0, 2), np.int32), np.zeros((0,), bool)

        stacked = np.hstack([array, np.ones((len(array), 1))])
        moved = stacked @ self.matrix.T
        w = moved[:, 2]
        # w 가 0 에 가까우면 소실선 근처라 좌표가 발산한다.
        valid = w > self.w_floor
        uv = np.zeros((len(array), 2), dtype=np.float64)
        uv[valid] = moved[valid, :2] / w[valid, None]

        if self.flip_horizontal:
            uv[:, 0] = (self.width - 1) - uv[:, 0]
        if self.flip_vertical:
            uv[:, 1] = (self.height - 1) - uv[:, 1]
        uv[:, 0] += self.pan[0]
        uv[:, 1] += self.pan[1]

        # 정수 변환 시 넘침을 막기 위해 화면에서 아주 먼 점은 잘라 낸다.
        limit = 10 * max(self.width, self.height)
        valid &= np.all(np.abs(uv) < limit, axis=1)
        return np.rint(uv).astype(np.int32), valid

    def clip_above_horizon(self, lane: Any) -> np.ndarray:
        """지평선 위쪽 점은 버린다. 노면 평면 가정이 깨지는 구간이다."""
        array = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
        if self.horizon <= 0.0:
            return array
        return array[array[:, 1] >= self.horizon]


def quad_matrix(
    quad_normalized: Any, width: int, height: int
) -> np.ndarray:
    """단위 사각형을 패널 위 사각형으로 보내는 예전 방식 행렬."""
    source = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    quad = np.asarray(quad_normalized, dtype=np.float32).reshape(4, 2)
    target = np.column_stack([quad[:, 0] * (width - 1), quad[:, 1] * (height - 1)])
    return cv2.getPerspectiveTransform(source, target.astype(np.float32)).astype(
        np.float64
    )


def solve_alignment(pairs: list[dict[str, Any]]) -> np.ndarray:
    """대응쌍에서 카메라 -> 패널 호모그래피를 구한다."""
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"need at least {MIN_PAIRS} pairs, got {len(pairs)}")
    source = np.asarray([p["camera"] for p in pairs], dtype=np.float64)
    target = np.asarray([p["panel"] for p in pairs], dtype=np.float64)
    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None:
        raise ValueError("could not solve alignment from these points")
    # 부호를 맞춰 유효 영역에서 w 가 양수가 되게 한다.
    center = np.append(source.mean(axis=0), 1.0)
    if float((matrix @ center)[2]) < 0:
        matrix = -matrix
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def blended_pairs(
    pairs: list[dict[str, Any]], profiles: list[dict[str, Any]], blend: float
) -> list[dict[str, Any]]:
    """눈높이가 다른 두 사람의 보정 사이를 섞는다.

    프로필이 두 개 이상 있으면 패널 쪽 점을 선형 보간한 뒤 행렬을 다시 푼다.
    운전자가 바뀔 때 전체 보정을 반복하지 않기 위한 장치다.
    """
    if len(profiles) < 2 or not pairs:
        return pairs
    first = np.asarray(profiles[0]["panel"], dtype=np.float64)
    second = np.asarray(profiles[1]["panel"], dtype=np.float64)
    if first.shape != second.shape or len(first) != len(pairs):
        return pairs
    ratio = float(np.clip(blend, 0.0, 1.0))
    mixed = first + (second - first) * ratio
    return [
        {"camera": pair["camera"], "panel": mixed[index].tolist()}
        for index, pair in enumerate(pairs)
    ]


def vanishing_floor(
    pairs: list[dict[str, Any]], matrix: np.ndarray, margin: float
) -> float:
    """소실선에 얼마나 가까운 점까지 그릴지 정하는 하한.

    보정에 쓴 표식들의 w 값을 기준 삼아 자동으로 잡는다. 카메라 장착 각도가
    달라져도 지평선 높이를 손으로 다시 넣을 필요가 없다.
    """
    source = np.asarray([p["camera"] for p in pairs], dtype=np.float64)
    stacked = np.hstack([source, np.ones((len(source), 1))])
    values = stacked @ matrix.T
    positive = values[:, 2][values[:, 2] > 0]
    if positive.size == 0:
        return 1e-6
    return max(1e-6, float(positive.min()) * float(margin))


def alignment_error(pairs: list[dict[str, Any]], matrix: np.ndarray) -> list[float]:
    """대응쌍마다 몇 픽셀 어긋나는지 돌려준다."""
    errors = []
    for pair in pairs:
        point = np.append(np.asarray(pair["camera"], dtype=np.float64), 1.0)
        moved = matrix @ point
        if abs(moved[2]) < 1e-9:
            errors.append(float("inf"))
            continue
        projected = moved[:2] / moved[2]
        errors.append(float(np.linalg.norm(projected - np.asarray(pair["panel"]))))
    return errors


# ---------------------------------------------------------------------------
# 방향키 처리: 리눅스 빌드에 따라 코드가 달라 둘 다 받는다.
# ---------------------------------------------------------------------------

ARROW_LEFT = (81, 65361, 2424832)
ARROW_UP = (82, 65362, 2490368)
ARROW_RIGHT = (83, 65363, 2555904)
ARROW_DOWN = (84, 65364, 2621440)


def _nudge(key: int) -> tuple[int, int] | None:
    """키 하나를 (dx, dy) 로 바꾼다. 소문자 명령이 먼저 걸러진 뒤에만 부른다."""
    if key in (ord("a"),) or key in ARROW_LEFT:
        return (-1, 0)
    if key in (ord("d"),) or key in ARROW_RIGHT:
        return (1, 0)
    if key in (ord("w"),) or key in ARROW_UP:
        return (0, -1)
    if key in (ord("x"),) or key in ARROW_DOWN:
        return (0, 1)
    return None


# ---------------------------------------------------------------------------
# preset: 시뮬레이터 화면 캡처용. 좌표를 이미 알고 있으므로 촬영이 필요 없다
# ---------------------------------------------------------------------------

SIM_TARGETS = [
    (0.20, 0.72), (0.80, 0.72),
    (0.15, 0.50), (0.85, 0.50),
    (0.32, 0.30), (0.68, 0.30),
]


def make_target_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """게임 모니터에 띄울 보정판을 만든다."""
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    for order, (nx, ny) in enumerate(SIM_TARGETS):
        point = (int(nx * (width - 1)), int(ny * (height - 1)))
        _draw_crosshair(image, point, (255, 255, 255), 46)
        cv2.putText(
            image,
            str(order + 1),
            (point[0] + 56, point[1] - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "HUD alignment target - show fullscreen on the sim monitor",
        (60, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (150, 150, 150),
        2,
        cv2.LINE_AA,
    )
    return image


def run_preset(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    section = ensure_alignment_config(config)
    if args.write_image:
        cv2.imwrite(str(args.write_image), make_target_image())
        print(f"wrote target image: {args.write_image}")
    section["pairs"] = [
        {"camera": [x, y], "panel": None} for x, y in SIM_TARGETS
    ]
    save_config(args.config, config)
    print(f"seeded {len(SIM_TARGETS)} camera points. now run aim.")


# ---------------------------------------------------------------------------
# eyebox: 사람이 앉을 때마다 머리 위치를 같은 자리로 맞추는 조준 표시
# ---------------------------------------------------------------------------


def run_eyebox(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    display = config["display"]
    width, height = int(display["width"]), int(display["height"])

    name = "HUD eyebox"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(name, width, height)
    else:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("move your head until this ring sits on the sticker. q to quit")
    center = (width // 2, height // 2)
    while True:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(canvas, center, 46, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.imshow(name, canvas)
        if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# profile: 지금 보정을 사람 이름으로 저장해 두고 나중에 사이를 섞는다
# ---------------------------------------------------------------------------


def run_profile(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    section = ensure_alignment_config(config)
    recorded = [p for p in section["pairs"] if p.get("panel")]
    if len(recorded) < MIN_PAIRS:
        raise SystemExit("no complete alignment to save as a profile")
    profiles = [p for p in section["profiles"] if p["name"] != args.name]
    profiles.append({"name": args.name, "panel": [p["panel"] for p in recorded]})
    section["profiles"] = profiles
    save_config(args.config, config)
    print(f"saved profile '{args.name}' ({len(profiles)} total)")
    if len(profiles) >= 2:
        print(f"blend runs between '{profiles[0]['name']}' and '{profiles[1]['name']}'")


# ---------------------------------------------------------------------------
# 패널 우선 방식 1단계 place: HUD 가 점을 찍어주고 그 자리에 꼬깔을 놓는다
# ---------------------------------------------------------------------------

PANEL_SPREAD = [
    (0.24, 0.78), (0.76, 0.78),
    (0.20, 0.52), (0.80, 0.52),
    (0.34, 0.24), (0.66, 0.24),
]


def run_place(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    section = ensure_alignment_config(config)
    display = config["display"]
    width, height = int(display["width"]), int(display["height"])

    name = "HUD place"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(name, width, height)
    else:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("put a cone where each crosshair appears on the road, then press enter.")
    print("w a x d nudge the crosshair, f step size, u back, s save, q quit")

    targets = [
        [int(x * (width - 1)), int(y * (height - 1))] for x, y in PANEL_SPREAD
    ]
    placed: list[list[int]] = []
    index = 0
    step = 4

    while index < len(targets):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for done in placed:
            cv2.circle(canvas, (done[0], done[1]), 7, (90, 90, 90), 2, cv2.LINE_AA)
        cursor = targets[index]
        _draw_crosshair(canvas, (cursor[0], cursor[1]), (255, 255, 255), 34)
        cv2.putText(
            canvas,
            f"CONE {index + 1} / {len(targets)}   STEP {step}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(name, canvas)

        key = cv2.waitKeyEx(20)
        if key == -1:
            continue
        low = key & 0xFF
        if low in (27, ord("q")):
            cv2.destroyAllWindows()
            return
        if low == ord("f"):
            step = 1 if step != 1 else 4
            continue
        if low in (13, 10):
            placed.append(list(cursor))
            index += 1
            continue
        if low == ord("u") and placed:
            placed.pop()
            index = max(0, index - 1)
            continue
        moved = _nudge(low) or _nudge(key)
        if moved:
            cursor[0] = int(np.clip(cursor[0] + moved[0] * step, 0, width - 1))
            cursor[1] = int(np.clip(cursor[1] + moved[1] * step, 0, height - 1))

    section["pairs"] = [{"camera": None, "panel": p} for p in placed]
    save_config(args.config, config)
    print(f"saved {len(placed)} panel points. now photograph the cones and run pick.")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# pick: 촬영한 도로 영상에서 표식 위치를 찍는다
# ---------------------------------------------------------------------------


def run_pick(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    section = ensure_alignment_config(config)

    frame = cv2.imread(str(args.frame))
    if frame is None:
        raise SystemExit(f"could not read frame: {args.frame}")
    height, width = frame.shape[:2]

    picked: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < MAX_PAIRS:
            picked.append((x / max(1, width - 1), y / max(1, height - 1)))

    name = "pick markers"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, min(1280, width), min(720, height))
    cv2.setMouseCallback(name, on_mouse)
    print("click each ground marker, far ones first. u undo, s save, q quit")

    while True:
        canvas = frame.copy()
        for index, (nx, ny) in enumerate(picked):
            point = (int(nx * (width - 1)), int(ny * (height - 1)))
            cv2.drawMarker(canvas, point, (0, 220, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(
                canvas,
                str(index + 1),
                (point[0] + 12, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"{len(picked)} / {MAX_PAIRS} picked   (need {MIN_PAIRS}+)",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("u") and picked:
            picked.pop()
        elif key == ord("s"):
            if len(picked) < MIN_PAIRS:
                print(f"need at least {MIN_PAIRS} points")
                continue
            existing = [p for p in section["pairs"] if p.get("panel")]
            if existing and len(existing) != len(picked):
                print(
                    f"panel points already saved: {len(existing)}. "
                    f"click exactly that many, in the same order."
                )
                continue
            pairs = []
            for order, (x, y) in enumerate(picked):
                panel = existing[order]["panel"] if existing else None
                pairs.append({"camera": [round(x, 5), round(y, 5)], "panel": panel})
            section["pairs"] = pairs
            if existing:
                matrix = solve_alignment(pairs)
                errors = alignment_error(pairs, matrix)
                print("reprojection error per marker (panel px):")
                for order, err in enumerate(errors):
                    print(f"  marker {order + 1}: {err:.1f}")
                print(f"  mean: {sum(errors) / len(errors):.1f}")
                section["mode"] = "correspondence"
            save_config(args.config, config)
            print(f"saved {len(picked)} camera points to {args.config}")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 2단계 aim: 운전석에서 HUD 위의 점을 실제 표식에 겹친다
# ---------------------------------------------------------------------------


def _draw_crosshair(
    canvas: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], size: int
) -> None:
    x, y = point
    cv2.line(canvas, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), max(4, size // 3), color, 2, cv2.LINE_AA)


def run_aim(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    section = ensure_alignment_config(config)
    display = config["display"]
    width, height = int(display["width"]), int(display["height"])

    pairs = [p for p in section["pairs"] if p.get("camera")]
    if len(pairs) < MIN_PAIRS:
        raise SystemExit("run 'pick' first: not enough camera points")

    name = "HUD aim"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(name, width, height)
    else:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("driver keeps head still. w a x d move, hold shift-less keys")
    print("f fast, enter record, u back, s save and solve, q quit")

    index = 0
    cursor = [width // 2, height // 2]
    step = 4

    while True:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for done, pair in enumerate(pairs):
            if pair.get("panel") and done != index:
                p = (int(pair["panel"][0]), int(pair["panel"][1]))
                cv2.circle(canvas, p, 7, (90, 90, 90), 2, cv2.LINE_AA)
        _draw_crosshair(canvas, (cursor[0], cursor[1]), (255, 255, 255), 34)
        cv2.putText(
            canvas,
            f"MARKER {index + 1} / {len(pairs)}   STEP {step}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(name, canvas)

        key = cv2.waitKeyEx(20)
        if key == -1:
            continue
        low = key & 0xFF

        if low in (27, ord("q")):
            break
        if low == ord("f"):
            step = 1 if step != 1 else 4
            continue
        if low == ord("s"):
            recorded = [p for p in pairs if p.get("panel")]
            if len(recorded) < MIN_PAIRS:
                print(f"need {MIN_PAIRS}+ recorded markers, have {len(recorded)}")
                continue
            matrix = solve_alignment(recorded)
            errors = alignment_error(recorded, matrix)
            print("reprojection error per marker (panel px):")
            for i, err in enumerate(errors):
                print(f"  marker {i + 1}: {err:.1f}")
            print(f"  mean: {sum(errors) / len(errors):.1f}")
            section["pairs"] = recorded
            section["mode"] = "correspondence"
            save_config(args.config, config)
            print(f"saved alignment to {args.config}")
            break
        if low in (13, 10):
            pairs[index]["panel"] = [int(cursor[0]), int(cursor[1])]
            index = min(index + 1, len(pairs) - 1)
            continue
        if low == ord("u"):
            index = max(0, index - 1)
            pairs[index]["panel"] = None
            continue

        moved = _nudge(low) or _nudge(key)
        if moved:
            cursor[0] = int(np.clip(cursor[0] + moved[0] * step, 0, width - 1))
            cursor[1] = int(np.clip(cursor[1] + moved[1] * step, 0, height - 1))

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 3단계 verify: 계산된 행렬로 표식 자리에 십자를 띄워 눈으로 확인
# ---------------------------------------------------------------------------


def run_verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    section = ensure_alignment_config(config)
    mapper = AlignmentMap(config)
    if not mapper.calibrated:
        raise SystemExit("no correspondence alignment saved yet")

    name = "HUD verify"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(name, mapper.width, mapper.height)
    else:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("crosshairs should sit on the real markers.")
    print("w a x d nudge all, - = blend between profiles, s save, q quit")
    blend = float(section.get("blend", 0.0))
    camera_points = [p["camera"] for p in section["pairs"]]

    while True:
        canvas = np.zeros((mapper.height, mapper.width, 3), dtype=np.uint8)
        projected, valid = mapper.project(camera_points)
        for point, ok in zip(projected, valid):
            if not ok:
                continue
            _draw_crosshair(canvas, (int(point[0]), int(point[1])), (255, 255, 255), 30)
        cv2.putText(
            canvas,
            f"PAN {mapper.pan[0]:+d} {mapper.pan[1]:+d}   BLEND {blend:.2f}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(name, canvas)

        key = cv2.waitKeyEx(20)
        if key == -1:
            continue
        low = key & 0xFF
        if low in (27, ord("q")):
            break
        if low == ord("s"):
            section["pan_x"], section["pan_y"] = mapper.pan
            section["blend"] = round(blend, 3)
            save_config(args.config, config)
            print(f"saved pan and blend to {args.config}")
            continue
        if low in (ord("-"), ord("=")):
            blend = float(np.clip(blend + (0.05 if low == ord("=") else -0.05), 0, 1))
            section["blend"] = blend
            pan = mapper.pan
            mapper = AlignmentMap(config)
            mapper.pan = pan
            continue
        moved = _nudge(low) or _nudge(key)
        if moved:
            mapper.pan = (mapper.pan[0] + moved[0] * 2, mapper.pan[1] + moved[1] * 2)

    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="운전자 시점 정렬 보정")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preset = subparsers.add_parser("preset", help="시뮬레이터 화면 캡처용 좌표 넣기")
    preset.add_argument("--config", type=Path, default=Path("hud_config.json"))
    preset.add_argument("--write-image", type=Path)

    eyebox = subparsers.add_parser("eyebox", help="머리 위치 맞추는 조준 표시")
    eyebox.add_argument("--config", type=Path, default=Path("hud_config.json"))
    eyebox.add_argument("--windowed", action="store_true")

    profile = subparsers.add_parser("profile", help="현재 보정을 이름 붙여 저장")
    profile.add_argument("--name", required=True)
    profile.add_argument("--config", type=Path, default=Path("hud_config.json"))

    place = subparsers.add_parser("place", help="HUD 가 찍은 자리에 꼬깔 놓기")
    place.add_argument("--config", type=Path, default=Path("hud_config.json"))
    place.add_argument("--windowed", action="store_true")

    pick = subparsers.add_parser("pick", help="도로 영상에서 표식 찍기")
    pick.add_argument("--frame", type=Path, required=True)
    pick.add_argument("--config", type=Path, default=Path("hud_config.json"))

    aim = subparsers.add_parser("aim", help="운전석에서 HUD 점 맞추기")
    aim.add_argument("--config", type=Path, default=Path("hud_config.json"))
    aim.add_argument("--windowed", action="store_true")

    verify = subparsers.add_parser("verify", help="정렬 확인과 미세 조정")
    verify.add_argument("--config", type=Path, default=Path("hud_config.json"))
    verify.add_argument("--windowed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {
        "preset": run_preset,
        "eyebox": run_eyebox,
        "profile": run_profile,
        "place": run_place,
        "pick": run_pick,
        "aim": run_aim,
        "verify": run_verify,
    }[args.command](args)


if __name__ == "__main__":
    main()
