#!/usr/bin/env python3
"""AR HUD 차선 오버레이를 OpenCV 로 구현한 렌더러.

반사식 HUD 는 화면 요소가 많을수록 운전자 시야를 가린다. 그래서 좌우 차선
경계선만 그린다. 리본 채움, 거리 틱, 상단 칩과 상태 스트립, 글로우는 전부
뺐다. 남은 것은 선 두 개와 인식 상태에 따른 색, 차선 이탈 판정에 따른
굵기·점멸뿐이다.

  - 원근 페이드: 마스크를 그린 뒤 세로 방향 알파 램프를 곱한다
  - 합성: 발광 디스플레이라 알파가 아니라 가산으로 쌓는다
  - 선 굵기: 절대 픽셀이 아니라 화면 높이 비율 (theme.*_width_ratio)

차선 폴리라인은 추론이 보낸 정규화 좌표를 운전자 시점 정렬 행렬로 투영해서
그린다.

인식 상태 state(0 정상 / 1 주의 / 2 인식 불가)는 두 군데에 나타난다.
우측 상단 신호등 3칸은 항상 세 칸을 그리고 현재 칸만 밝다. 차선에서
state 는 색만 정한다. 0 은 녹색, 1 은 노란색, 2 는 아예 그리지 않는다.
0 과 1 은 색만 다르고 굵기·모양·밝기가 완전히 같다. 차선 색은 신호등과
같은 STATE_COLORS 를 쓴다. 차선 이탈 경고도 색에 관여하지 않고 굵기와
점멸만 바꾼다.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from hud_align import AlignmentMap, ensure_alignment_config

DESIGN_W, DESIGN_H = 960.0, 540.0
HORIZON_Y = 172.0

ALERT = (61, 77, 255)        # #ff4d3d BGR

EDGE_STOPS = ((0.0, 1.0), (0.62, 0.72), (1.0, 0.0))
DIM_STOPS = ((0.0, 0.34), (0.70, 0.12), (1.0, 0.0))
ALERT_EDGE_STOPS = ((0.0, 1.0), (0.62, 0.70), (1.0, 0.0))

BLINK_PERIOD = 0.820

# 인식 상태. 0 정상, 1 주의(차선이 흐릿함), 2 인식 불가.
# 젯슨이 패킷에 실어 보낸다. 프로토콜 합의 전이라 빠져 있을 수 있고
# 그때는 0 으로 본다. hud_ui 도 이 정의를 가져다 쓴다.
STATE_NORMAL, STATE_CAUTION, STATE_LOST = 0, 1, 2
LANE_STATES = (STATE_NORMAL, STATE_CAUTION, STATE_LOST)

# 신호등 칸 색. 인덱스가 곧 state 다.
STATE_COLORS = (
    (100, 220, 100),     # 0 정상   녹색
    (100, 220, 240),     # 1 주의   노란색
    ALERT,               # 2 인식 불가  붉은색
)
STATE_DIM = 0.18         # 꺼진 칸 밝기

CONFIDENCE_FLOOR = 0.45  # 이 아래면 state 를 1 로 올린다 (대체 신호)


def normalize_state(value: Any) -> int:
    """패킷에서 읽은 state 를 0/1/2 로 만든다. 이상하면 0."""
    try:
        state = int(value)
    except (TypeError, ValueError):
        return STATE_NORMAL
    return state if state in LANE_STATES else STATE_NORMAL


def _build_ramp(height: int, bottom_y: float, top_y: float, stops) -> np.ndarray:
    """패널 행마다 알파 배수를 담은 세로 램프를 만든다."""
    rows = np.arange(height, dtype=np.float32)
    span = max(1e-6, bottom_y - top_y)
    t = np.clip((bottom_y - rows) / span, 0.0, 1.0)
    offsets = np.asarray([s[0] for s in stops], dtype=np.float32)
    values = np.asarray([s[1] for s in stops], dtype=np.float32)
    return np.interp(t, offsets, values).astype(np.float32)[:, None]


class ThemeRenderer:
    """좌우 차선 경계선만 그리는 렌더러."""

    def __init__(self, config: dict[str, Any]) -> None:
        display = config["display"]
        self.width = int(display["width"])
        self.height = int(display["height"])
        ensure_alignment_config(config)
        self.mapper = AlignmentMap(config)

        theme = config.setdefault("theme", {})
        # 굵기는 절대 픽셀이 아니라 화면 높이 비율로 둔다. 패널이 바뀌어도
        # 눈에 보이는 굵기가 유지된다. 1280x720 에서 0.0148 -> 약 11px.
        theme.setdefault("edge_width_ratio", 0.0148)
        theme.setdefault("alert_edge_width_ratio", 0.0296)
        theme.setdefault("dim_edge_width_ratio", 0.0111)
        theme.setdefault("boot_animation", True)
        # 신호등 인디케이터. 전부 디자인 960x540 좌표 기준이다.
        theme.setdefault("state_indicator", True)
        theme.setdefault("state_dot_diameter", 16.0)
        theme.setdefault("state_dot_pitch", 24.0)       # 원 중심 간격
        theme.setdefault("state_dot_margin_x", 40.0)    # 우측 끝 ~ 마지막 원 중심
        theme.setdefault("state_dot_center_y", 40.0)
        self.theme = theme
        # hud_ui 의 preview / sample 과 인터페이스를 맞추기 위한 최소 항목
        self.ui = config.setdefault("ui", {})
        self.ui.setdefault("debug", {"enabled": False})
        self.ui.setdefault(
            "lane", {"thickness": self._ratio_px(theme["edge_width_ratio"])}
        )

        # 디자인 960x540 을 패널에 레터박스로 맞춘다
        self.scale = min(self.width / DESIGN_W, self.height / DESIGN_H)
        self.offset_x = (self.width - DESIGN_W * self.scale) / 2.0
        self.offset_y = (self.height - DESIGN_H * self.scale) / 2.0

        self._ramp_range = (-1.0, -1.0)
        self.ramps: dict[str, np.ndarray] = {}
        self._update_ramps(self._dy(DESIGN_H), self._dy(HORIZON_Y))
        # 채널을 분리해 두면 합성이 훨씬 빠르다. 3채널 배열에 브로드캐스트로
        # 곱하는 것보다 1채널 연속 메모리 세 번이 6배 가까이 빠르다.
        self.planes = [np.zeros((self.height, self.width), np.float32) for _ in range(3)]
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.started = time.monotonic()
        self.state = STATE_NORMAL
        self._build_state_dots()

    def _update_ramps(self, bottom: float, top: float) -> None:
        """원근 페이드를 리본이 실제로 차지한 세로 범위에 맞춘다."""
        if abs(bottom - self._ramp_range[0]) < 2 and abs(top - self._ramp_range[1]) < 2:
            return
        self._ramp_range = (bottom, top)
        for name, stops in (("edge", EDGE_STOPS), ("dim", DIM_STOPS),
                            ("alert_edge", ALERT_EDGE_STOPS)):
            self.ramps[name] = _build_ramp(self.height, bottom, top, stops)

    @property
    def calibrated(self) -> bool:
        return self.mapper.calibrated

    def rebuild(self, config: dict[str, Any]) -> None:
        self.__init__(config)

    # 좌표 --------------------------------------------------------------

    def _dx(self, x: float) -> float:
        return self.offset_x + x * self.scale

    def _dy(self, y: float) -> float:
        return self.offset_y + y * self.scale

    def _dp(self, x: float, y: float) -> tuple[int, int]:
        return int(round(self._dx(x))), int(round(self._dy(y)))

    def _ratio_px(self, ratio: float) -> int:
        """화면 높이 비율을 선 굵기 픽셀로 바꾼다."""
        return max(1, int(round(float(ratio) * self.height)))

    # 합성 --------------------------------------------------------------

    def _clear_mask(self) -> np.ndarray:
        self.mask[:] = 0
        return self.mask

    def _paint(
        self,
        mask: np.ndarray,
        color: tuple[int, int, int],
        ramp: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        """마스크를 색으로 칠해 캔버스에 더한다.

        HUD 는 발광 디스플레이라 겹치는 빛이 밝아지므로 알파 합성이 아니라
        가산으로 쌓는다. 실제 광학계 동작과도 맞고 훨씬 빠르다.
        마스크 전체가 아니라 실제로 칠해진 사각형 범위만 처리한다.
        """
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return
        patch = mask[y:y + h, x:x + w].astype(np.float32)
        patch *= opacity / 255.0
        if ramp is not None:
            patch *= self.ramps[ramp][y:y + h]
        for channel in range(3):
            level = color[channel]
            if level:
                self.planes[channel][y:y + h, x:x + w] += level * patch

    # 상태 --------------------------------------------------------------

    def _resolve_state(
        self, state: Any, warning: str, telemetry: dict[str, Any]
    ) -> int:
        """인식 상태를 한 곳에서 정한다.

        예전에는 telemetry.confidence 로 저신뢰 판정을 따로 해서 점선과 감광을
        걸었는데, 그게 결국 state 1(주의) 과 같은 이야기였다. 점선과 감광은
        이제 아예 없고 state 1 은 색만 바뀐다. confidence 는 젯슨이 state 를
        안 실어 보낼 때만 쓰는 대체 신호로 남았다. state 가 이미 0 이 아니면
        그대로 따른다. 신호등과 차선은 여기서 나온 값 하나만 본다.

        차선 이탈 경고 중에는 대체 신호를 쓰지 않는다. 이탈 표시가 우선이라
        그때 색까지 흔들 이유가 없다.
        """
        resolved = normalize_state(state)
        if resolved == STATE_NORMAL and warning == "none":
            if float(telemetry.get("confidence", 1.0)) < CONFIDENCE_FLOOR:
                return STATE_CAUTION
        return resolved

    # 차선 --------------------------------------------------------------

    def _project_lane(self, lane: Any) -> np.ndarray | None:
        clipped = self.mapper.clip_above_horizon(lane)
        points, valid = self.mapper.project(clipped)
        if valid.sum() < 2:
            return None
        points = points[valid]
        # 패널 좌우로 빠져나간 근거리 구간은 리본에서 뺀다. 그대로 두면
        # 채움이 화면 아래쪽을 통째로 덮어 시야를 가린다.
        margin = int(self.width * 0.02)
        inside = (points[:, 0] >= -margin) & (points[:, 0] <= self.width + margin)
        if inside.sum() < 2:
            return None
        points = points[inside]
        order = np.argsort(-points[:, 1])       # 아래에서 위로
        return points[order]

    def _draw_ribbon(
        self,
        left: np.ndarray | None,
        right: np.ndarray | None,
        departure_side: str | None,
        elapsed: float,
        state: int,
        boot_progress: float,
    ) -> None:
        alert = departure_side is not None
        # state 는 색만 정한다. 신호등 인디케이터와 같은 상수를 쓰므로 두
        # 곳이 갈라지지 않는다. state 0 과 1 은 색만 다르고 굵기·모양·밝기가
        # 같다. 차선 이탈 경고 역시 색은 건드리지 않고 굵기와 점멸만 바꾼다.
        color = STATE_COLORS[state]
        present = [p for p in (left, right) if p is not None]
        if present:
            self._update_ramps(
                float(max(p[0][1] for p in present)),
                float(min(p[-1][1] for p in present)),
            )
        phase = (elapsed % BLINK_PERIOD) / BLINK_PERIOD

        # 1. 경계선
        for side, points in (("left", left), ("right", right)):
            if points is None:
                continue
            reveal = points
            if boot_progress < 1.0:
                delay = 0.0 if side == "left" else 0.24
                local = np.clip((boot_progress - delay) / max(1e-6, 1.0 - delay), 0, 1)
                keep = max(2, int(len(points) * local))
                reveal = points[:keep]
                if len(reveal) < 2:
                    continue
            mask = self._clear_mask()
            departing = alert and side == departure_side
            if departing:
                width = self._ratio_px(self.theme["alert_edge_width_ratio"])
            elif alert:
                width = self._ratio_px(self.theme["dim_edge_width_ratio"])
            else:
                width = self._ratio_px(self.theme["edge_width_ratio"])
            cv2.polylines(mask, [reveal], False, 255, width, cv2.LINE_AA)
            if departing:
                on = phase < 0.5                     # steps(1, end)
                self._paint(mask, color, "alert_edge", 1.0 if on else 0.18)
            elif alert:
                self._paint(mask, color, "dim")
            else:
                self._paint(mask, color, "edge")

    # 신호등 ------------------------------------------------------------

    def _build_state_dots(self) -> None:
        """신호등 칸 하나를 알파 패치로 미리 만들어 둔다.

        프레임마다 원을 다시 그리고 마스크 전체에 boundingRect 를 도는 대신,
        작은 패치 하나를 세 자리에 더하기만 한다. 반사식은 가장자리 계단이
        그대로 보이므로 4배로 그린 뒤 INTER_AREA 로 줄여 받는다.
        """
        theme = self.theme
        size = max(2, int(round(float(theme["state_dot_diameter"]) * self.scale)))
        supersample = 4
        big = np.zeros((size * supersample, size * supersample), np.uint8)
        center = size * supersample // 2
        cv2.circle(big, (center, center), center - supersample, 255, -1)
        self._state_dot = (
            cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
            .astype(np.float32) / 255.0
        )

        pitch = float(theme["state_dot_pitch"])
        margin = float(theme["state_dot_margin_x"])
        center_y = float(theme["state_dot_center_y"])
        last = len(LANE_STATES) - 1
        self._state_dot_origins = []
        for index in range(len(LANE_STATES)):
            cx, cy = self._dp(DESIGN_W - margin - (last - index) * pitch, center_y)
            x0 = min(max(0, cx - size // 2), max(0, self.width - size))
            y0 = min(max(0, cy - size // 2), max(0, self.height - size))
            self._state_dot_origins.append((x0, y0))

    def _draw_state_indicator(self) -> None:
        """세 칸을 항상 그린다. 현재 state 칸만 100%, 나머지는 18%.

        자리는 지평선(y=172)보다 한참 위인 우측 상단이라 차선 리본과 겹치지
        않는다. state 2 의 붉은 등과 차선 이탈의 붉은 선이 동시에 떠도
        화면 위아래로 확실히 떨어져 있다.
        """
        if not self.theme["state_indicator"]:
            return
        dot = self._state_dot
        size = dot.shape[0]
        for index, (x0, y0) in enumerate(self._state_dot_origins):
            color = STATE_COLORS[index]
            level = 1.0 if index == self.state else STATE_DIM
            for channel in range(3):
                value = color[channel] * level
                if value:
                    self.planes[channel][y0:y0 + size, x0:x0 + size] += value * dot

    # 출력 --------------------------------------------------------------

    def _compose(self) -> np.ndarray:
        """세 채널 평면을 8비트 BGR 프레임으로 합친다."""
        return cv2.convertScaleAbs(cv2.merge(self.planes))

    # 한 프레임 ----------------------------------------------------------

    def render(
        self,
        *,
        lanes: list[Any],
        warning: str = "none",
        elapsed: float = 0.0,
        debug: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        state: int = STATE_NORMAL,
    ) -> np.ndarray:
        telemetry = telemetry or {}
        self.state = self._resolve_state(state, warning, telemetry)
        for plane in self.planes:
            plane[:] = 0.0

        departure_side = None
        if warning.startswith("departure"):
            departure_side = "left" if warning.endswith("_left") else "right"

        # state 2 는 차선을 아예 그리지 않는다. 투영까지 건너뛴다.
        if self.state != STATE_LOST:
            boot = 1.0
            if self.theme["boot_animation"] and departure_side is None:
                age = time.monotonic() - self.started
                boot = float(np.clip(age / 1.2, 0.0, 1.0))

            projected = [self._project_lane(lane) for lane in lanes]
            projected = [p for p in projected if p is not None]
            left = right = None
            if len(projected) >= 2:
                projected.sort(key=lambda p: p[0][0])
                left, right = projected[0], projected[-1]
            elif len(projected) == 1:
                single = projected[0]
                if single[0][0] < self.width / 2:
                    left = single
                else:
                    right = single

            self._draw_ribbon(left, right, departure_side, elapsed,
                              self.state, boot)
        self._draw_state_indicator()

        frame = self._compose()
        if debug is not None and debug.get("show"):
            cv2.putText(frame, "ALIGN {}  FPS {:.1f}  SEQ {}  STATE {}".format(
                "cal" if self.calibrated else "uncal",
                float(debug.get("fps", 0.0)), debug.get("seq", 0), self.state),
                (12, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * self.scale + 0.2, (120, 120, 120), 1, cv2.LINE_AA)
        return frame
