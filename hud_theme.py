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
화면 아래쪽, 가장자리에서 state_bar_margin_ratio 만큼 안쪽을 가로지르는
얇은 상태 바는 state 색 한 가지로 항상 그린다. state 2 에서도 그린다.
차선에서 state 는 색만 정한다. 0 은 녹색, 1 은 노란색, 2 는 아예 그리지 않는다. 0 과 1 은 색만 다르고 굵기·모양·밝기가
완전히 같다. 차선 색은 상태 바와 같은 STATE_COLORS 를 쓴다. 차선 이탈 경고도 색에 관여하지 않고 굵기와
점멸만 바꾼다.

state 를 언제 무엇으로 볼지는 여기서 정하지 않는다. 무수신 판정, 페이드,
디바운싱은 hud_state 가 맡고 이 파일은 render() 인자로 받은 state 와
lane_state / lane_opacity 를 그대로 칠한다.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from hud_align import AlignmentMap, ensure_alignment_config
from hud_system import _clip_polyline

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

# 상태 바와 차선 색. 인덱스가 곧 state 다.
STATE_COLORS = (
    (100, 220, 100),     # 0 정상   녹색
    (100, 220, 240),     # 1 주의   노란색
    ALERT,               # 2 인식 불가  붉은색
)

CONFIDENCE_FLOOR = 0.45  # 이 아래면 state 를 1 로 올린다 (대체 신호)


def normalize_state(value: Any) -> int:
    """패킷에서 읽은 state 를 0/1/2 로 만든다. 이상하면 0."""
    try:
        state = int(value)
    except (TypeError, ValueError):
        return STATE_NORMAL
    return state if state in LANE_STATES else STATE_NORMAL


def _build_ramp(height: int, near_y: float, far_y: float, stops) -> np.ndarray:
    """패널 행마다 알파 배수를 담은 세로 램프를 만든다.

    t 는 근거리 near_y 에서 0, 원거리 far_y 에서 1 이다. 상하 반전이 켜져
    있으면 근거리가 패널 위쪽이라 near_y < far_y 가 된다. 어느 방향이든
    부호가 span 에 실려 있어 같은 식으로 풀린다.
    """
    rows = np.arange(height, dtype=np.float32)
    span = far_y - near_y
    if abs(span) < 1e-6:
        span = 1e-6
    t = np.clip((rows - near_y) / span, 0.0, 1.0)
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
        # 눈에 보이는 굵기가 유지된다. 1920x1200 에서 0.0148 -> 약 18px.
        theme.setdefault("edge_width_ratio", 0.0148)
        theme.setdefault("alert_edge_width_ratio", 0.0296)
        theme.setdefault("dim_edge_width_ratio", 0.0111)
        theme.setdefault("boot_animation", True)
        # 하단 상태 바. 굵기는 차선과 마찬가지로 화면 높이 비율이다.
        # 1920x1200 에서 0.005 -> 6px.
        theme.setdefault("state_bar", True)
        theme.setdefault("state_bar_height_ratio", 0.005)
        # 상태 바를 운전자 기준 아랫면에서 이만큼 안쪽으로 띄운다. 화면 높이
        # 비율. 반사판은 패널 가장자리를 잘 못 비춰서 끝에 붙이면 시선을
        # 한참 내려야 보인다. 1920x1200 에서 0.03 -> 36px. pattern 의
        # SAFE 사각형이 같은 값이다.
        theme.setdefault("state_bar_margin_ratio", 0.03)
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
        near, far = self._dy(DESIGN_H), self._dy(HORIZON_Y)
        if self.mapper.flip_vertical:
            near, far = self.height - 1 - near, self.height - 1 - far
        self._update_ramps(near, far)
        # 채널을 분리해 두면 합성이 훨씬 빠르다. 3채널 배열에 브로드캐스트로
        # 곱하는 것보다 1채널 연속 메모리 세 번이 6배 가까이 빠르다.
        self.planes = [np.zeros((self.height, self.width), np.float32) for _ in range(3)]
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.started = time.monotonic()
        self.state = STATE_NORMAL
        self.lane_state = STATE_NORMAL
        self.lane_opacity = 1.0
        self._build_state_bar()

    def _update_ramps(self, near: float, far: float) -> None:
        """원근 페이드를 리본이 실제로 차지한 세로 범위에 맞춘다.

        near / far 는 근거리 끝과 원거리 끝의 패널 y 다. 패널 위아래가 아니라
        원근 기준이라 상하 반전이 켜져 있어도 근거리가 밝다.
        """
        if abs(near - self._ramp_range[0]) < 2 and abs(far - self._ramp_range[1]) < 2:
            return
        self._ramp_range = (near, far)
        for name, stops in (("edge", EDGE_STOPS), ("dim", DIM_STOPS),
                            ("alert_edge", ALERT_EDGE_STOPS)):
            self.ramps[name] = _build_ramp(self.height, near, far, stops)

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
        그대로 따른다. 상태 바와 차선은 여기서 나온 값 하나만 본다.

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
        """차선을 패널 픽셀로. 점 순서는 근거리 -> 원거리다.

        순서는 투영 전에 카메라 y 로 정한다. 카메라 영상에서는 아래(y 큰
        쪽)가 항상 가까운 쪽이지만, 패널에서는 상하 반전이나 대응쌍 행렬이
        위아래를 뒤집을 수 있어 패널 y 로는 원근을 알 수 없다. 예전처럼
        투영 뒤 패널 y 로 정렬하면 flip_vertical 에서 원거리가 맨 앞에 와
        페이드가 거꾸로 걸렸다. 아래 필터링은 전부 순서를 유지한다.
        """
        clipped = self.mapper.clip_above_horizon(lane)
        clipped = clipped[np.argsort(-clipped[:, 1], kind="stable")]
        points, valid = self.mapper.project(clipped)
        if valid.sum() < 2:
            return None
        points = points[valid]
        # 패널 좌우로 빠져나간 근거리 구간은 리본에서 뺀다. 그대로 두면
        # 채움이 화면 아래쪽을 통째로 덮어 시야를 가린다. 다만 점만 걷어
        # 내면 경계에서 선이 뚝 끊기므로 경계와의 교점을 끼워 넣는다.
        margin = float(self.width) * 0.02
        bounds = (-margin, -1e9, self.width + margin, 1e9)
        inside = _clip_polyline(points.astype(np.float64).tolist(), bounds)
        if len(inside) < 2:
            return None
        return np.rint(np.asarray(inside, dtype=np.float64)).astype(np.int32)

    def _edge_width(self, side: str, departure_side: str | None) -> int:
        """차선 한쪽을 그릴 굵기. 이탈 쪽은 굵게, 반대쪽은 가늘게."""
        if departure_side is None:
            key = "edge_width_ratio"
        elif side == departure_side:
            key = "alert_edge_width_ratio"
        else:
            key = "dim_edge_width_ratio"
        return self._ratio_px(self.theme[key])

    @staticmethod
    def _cut_far(points: np.ndarray, s: np.ndarray, s_cut: float) -> np.ndarray | None:
        """근거리부터 따라가다 진행 좌표 s 가 s_cut 을 넘는 자리에서 자른다.

        넘는 선분에는 교점을 끼워 넣어 선이 정확히 s_cut 에서 끝나게 한다.
        """
        keep = [points[0].astype(np.float64)]
        for index in range(1, len(points)):
            if s[index] <= s_cut:
                keep.append(points[index].astype(np.float64))
                continue
            if s[index - 1] < s_cut:
                t = (s_cut - s[index - 1]) / (s[index] - s[index - 1])
                keep.append(points[index - 1] + t * (points[index] - points[index - 1]))
            break
        if len(keep) < 2:
            return None
        return np.rint(np.asarray(keep)).astype(np.int32)

    def _trim_far_contact(
        self, left: np.ndarray, right: np.ndarray, min_gap: float
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """좌우 차선이 원거리에서 붙거나 교차하는 끝을 잘라 낸다.

        젯슨은 두 차선을 따로 2차 피팅하고 y 구간 하나를 같이 보낸다. 그래서
        소실점 부근에서 두 포물선이 몇 픽셀씩 겹쳐 X 자로 교차한다. 실측
        캡처에서 교차는 전부 y_range 맨 위 끝에서만 났다. 와이어 포맷은 그대로
        두고 그리는 쪽에서 정리한다.

        같은 패널 행에서 두 선의 중심 간격이 min_gap (그릴 선 굵기) 보다
        좁아지면 획이 맞닿는다. 근거리부터 따라가 처음 좁아지는 행에서 두
        차선을 함께 자른다. 근거리에서부터 이미 붙어 있으면 원거리 수렴이
        아니라 인식이 엉킨 것이므로 손대지 않는다.
        """
        # 진행 좌표 s: 근거리 -> 원거리로 커지는 패널 y. 반전이면 y 가
        # 줄어드는 쪽이 원거리라 부호를 뒤집는다.
        direction = 1.0 if left[-1][1] >= left[0][1] else -1.0
        s_left = direction * left[:, 1].astype(np.float64)
        s_right = direction * right[:, 1].astype(np.float64)
        # 꼭짓점이 구간 안에 든 갈고리 차선은 s 가 되돌아올 수 있다.
        # 간격 계산용으로만 단조 포락선을 쓴다.
        mono_left = np.maximum.accumulate(s_left)
        mono_right = np.maximum.accumulate(s_right)
        low = max(mono_left[0], mono_right[0])
        high = min(mono_left[-1], mono_right[-1])
        if high - low < 2.0:
            return left, right
        rows = np.linspace(low, high, max(2, int(high - low) // 4 + 1))
        # 근거리 끝 x 로 어느 쪽이 오른쪽인지 정한다. flip_horizontal 이면
        # 패널에서 좌우가 바뀌어 있다.
        side = 1.0 if right[0][0] >= left[0][0] else -1.0
        gap = side * (
            np.interp(rows, mono_right, right[:, 0].astype(np.float64))
            - np.interp(rows, mono_left, left[:, 0].astype(np.float64))
        )
        if gap[0] < min_gap:
            return left, right
        close = np.flatnonzero(gap < min_gap)
        if len(close) == 0:
            return left, right
        k = int(close[0])
        t = (gap[k - 1] - min_gap) / (gap[k - 1] - gap[k])
        s_cut = rows[k - 1] + t * (rows[k] - rows[k - 1])
        return (self._cut_far(left, s_left, s_cut),
                self._cut_far(right, s_right, s_cut))

    def _draw_ribbon(
        self,
        left: np.ndarray | None,
        right: np.ndarray | None,
        departure_side: str | None,
        elapsed: float,
        state: int,
        boot_progress: float,
        opacity: float = 1.0,
    ) -> None:
        alert = departure_side is not None
        # state 는 색만 정한다. 하단 상태 바와 같은 상수를 쓰므로 두
        # 곳이 갈라지지 않는다. state 0 과 1 은 색만 다르고 굵기·모양·밝기가
        # 같다. 차선 이탈 경고 역시 색은 건드리지 않고 굵기와 점멸만 바꾼다.
        color = STATE_COLORS[state]
        present = [p for p in (left, right) if p is not None]
        if present:
            # 점 순서가 근거리 -> 원거리라 p[0] 이 근거리 끝이다. 차선 여럿을
            # 다 덮도록 근거리는 가장 가까운 쪽, 원거리는 가장 먼 쪽을 고른다.
            nears = [float(p[0][1]) for p in present]
            fars = [float(p[-1][1]) for p in present]
            if sum(nears) >= sum(fars):
                self._update_ramps(max(nears), min(fars))
            else:
                self._update_ramps(min(nears), max(fars))
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
            width = self._edge_width(side, departure_side)
            cv2.polylines(mask, [reveal], False, 255, width, cv2.LINE_AA)
            if departing:
                on = phase < 0.5                     # steps(1, end)
                self._paint(mask, color, "alert_edge",
                            opacity * (1.0 if on else 0.18))
            elif alert:
                self._paint(mask, color, "dim", opacity)
            else:
                self._paint(mask, color, "edge", opacity)

    # 상태 바 ----------------------------------------------------------

    @property
    def safe_margin_px(self) -> int:
        """안전 영역 여백. 패널 네 변에서 같은 픽셀만큼 들어간다."""
        ratio = max(0.0, float(self.theme["state_bar_margin_ratio"]))
        return min(int(round(ratio * self.height)), self.height // 2)

    def _build_state_bar(self) -> None:
        """상태 바가 차지할 행 범위를 미리 정해 둔다.

        운전자가 보는 화면의 아랫면에서 safe_margin_px 만큼 안쪽에 둔다.
        flip_vertical 이 켜져 있으면 패널 윗줄이 운전자에게는 아랫면이므로
        여백도 패널 위쪽에서 잰다. 좌우 반전은 화면 폭 전체를 채우는 선이라
        상관없다.
        """
        rows = self._ratio_px(self.theme["state_bar_height_ratio"])
        rows = min(rows, self.height)
        margin = min(self.safe_margin_px, self.height - rows)
        if self.mapper.physical_flip_vertical:
            self._state_bar_rows = (margin, margin + rows)
        else:
            self._state_bar_rows = (self.height - margin - rows, self.height - margin)

    def _draw_state_bar(self) -> None:
        """화면 폭 전체에 state 색 선 한 줄을 더한다.

        마스크도 boundingRect 도 필요 없다. 칠할 영역이 몇 줄짜리 띠로
        정해져 있으니 평면 슬라이스에 바로 더한다. 차선이 화면 아래까지
        내려와 겹치면 가산이라 그 자리만 조금 밝아진다.
        """
        if not self.theme["state_bar"]:
            return
        top, bottom = self._state_bar_rows
        color = STATE_COLORS[self.state]
        for channel in range(3):
            if color[channel]:
                self.planes[channel][top:bottom] += color[channel]

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
        lane_state: int | None = None,
        lane_opacity: float = 1.0,
    ) -> np.ndarray:
        telemetry = telemetry or {}
        self.state = self._resolve_state(state, warning, telemetry)
        # 차선 색과 밝기는 상태 바 state 와 따로 받을 수 있다. 무수신 페이드
        # 중에는 상태 바만 즉시 붉어지고 차선은 직전 색 그대로 어두워진다.
        # 판정은 hud_state 가 하고 여기서는 받은 값을 칠하기만 한다.
        self.lane_state = (
            self.state if lane_state is None else normalize_state(lane_state)
        )
        self.lane_opacity = float(min(1.0, max(0.0, lane_opacity)))
        for plane in self.planes:
            plane[:] = 0.0

        departure_side = None
        if warning.startswith("departure"):
            departure_side = "left" if warning.endswith("_left") else "right"

        # state 2 는 차선을 아예 그리지 않는다. 투영까지 건너뛴다.
        # 페이드가 남아 있으면 lane_state 가 직전 값이라 잠시 더 그린다.
        if self.lane_state != STATE_LOST and self.lane_opacity > 1.0 / 255.0:
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
            if left is not None and right is not None:
                # 두 획이 맞닿는 간격은 굵은 쪽 선 굵기다
                min_gap = max(self._edge_width("left", departure_side),
                              self._edge_width("right", departure_side))
                left, right = self._trim_far_contact(left, right, min_gap)

            self._draw_ribbon(left, right, departure_side, elapsed,
                              self.lane_state, boot, self.lane_opacity)
        self._draw_state_bar()

        frame = self._compose()
        if debug is not None and debug.get("show"):
            text_scale = 0.45 * self.scale + 0.2
            lines = ["ALIGN {}  FPS {:.1f}  SEQ {}  STATE {}".format(
                "cal" if self.calibrated else "uncal",
                float(debug.get("fps", 0.0)), debug.get("seq", 0), self.state)]
            if "skipped" in debug:
                # 수신 지연. 젯슨 ts 대비 LAG, 최솟값 대비 Q, 버린 패킷 수
                lag = debug.get("lag_ms")
                queue = debug.get("queue_ms")
                lines.append("LAG {}  Q {}  SKIP {}  BAD {}".format(
                    "--" if lag is None else f"{lag:.0f}ms",
                    "--" if queue is None else f"{queue:.0f}ms",
                    debug.get("skipped", 0), debug.get("dropped", 0)))
            line_h = int(28 * text_scale) + 6
            for index, line in enumerate(reversed(lines)):
                cv2.putText(frame, line, (12, self.height - 12 - index * line_h),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (120, 120, 120),
                            1, cv2.LINE_AA)
        return frame
