#!/usr/bin/env python3
"""HUD 인식 상태 계층.

수신 상태와 인식 상태를 한 곳에서 판정한다. 그리기 코드(hud_theme)는 여기서
나온 결과를 받아 색과 밝기만 정하고, 판정에는 관여하지 않는다.

여기서 지키는 규칙은 네 가지다.

1. 무수신은 곧 state 2 다. packet_timeout_seconds 를 넘기면 젯슨이 보내온
   값과 무관하게 2 로 올린다. 젯슨이 죽으면 "차선 못 봄" 이라는 신호조차
   오지 않으므로, 값이 오기를 기다리는 구조는 고장을 정상으로 표시한다.
2. 차선은 뚝 끊기지 않고 lane_fade_seconds 에 걸쳐 어두워진다. 상태 바는
   페이드를 기다리지 않고 즉시 붉어진다. 경고는 늦추지 않는다.
3. state 1 이 caution_timeout_seconds 이상 이어지면 젯슨이 계속 1 을
   보내더라도 2 로 내린다. state 1 은 추정 궤적이라 오래 띄워 두면 확정된
   차선으로 오인된다.
4. 디바운싱. 0 -> 1 상승은 rise_frames 연속, 하강은 fall_frames 연속으로
   확인해야 반영한다. 2 로 올라가는 것만은 즉시 반영한다. 경고를 늦추는
   쪽으로는 한 프레임도 쓰지 않는다.

무수신(link_lost)과 젯슨이 보낸 state 2 는 화면 표시가 같다. 운전자가 할
일이 "오버레이를 믿지 말 것" 으로 같기 때문이다. 다만 원인은 전혀 다르므로
HudStatus.link_lost 와 reason 으로 구분해서 넘긴다. 디버그 표시와 로그는 이
값을 봐야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hud_theme import (
    CONFIDENCE_FLOOR,
    STATE_CAUTION,
    STATE_LOST,
    STATE_NORMAL,
    normalize_state,
)

DEFAULT_STATE: dict[str, Any] = {
    "lane_fade_seconds": 0.4,      # 무수신 후 차선이 사라지기까지
    "caution_timeout_seconds": 1.2,  # state 1 을 최대 얼마나 보여 줄지
    "rise_frames": 3,              # 0 -> 1 상승 확인 프레임 수
    "fall_frames": 5,              # 1 -> 0, 2 -> 0 하강 확인 프레임 수
}

# 페이드가 이 아래로 내려가면 차선을 아예 그리지 않는다. 8비트로 합성하면
# 어차피 보이지 않는 밝기라 투영 비용만 남는다.
OPACITY_FLOOR = 1.0 / 255.0


def ensure_state_config(config: dict[str, Any]) -> dict[str, Any]:
    """설정에 state 항목이 없으면 기본값을 채워 넣는다."""
    section = config.setdefault("state", {})
    for key, value in DEFAULT_STATE.items():
        section.setdefault(key, value)
    return section


@dataclass(frozen=True)
class HudStatus:
    """한 프레임의 판정 결과."""

    state: int            # 상태 바에 띄울 값
    lane_state: int       # 차선 색을 정하는 값. 페이드 중에는 직전 값을 유지한다
    lane_opacity: float   # 차선 밝기 배수 0.0 ~ 1.0
    link_lost: bool       # 패킷이 끊겼는지 (젯슨이 보낸 state 2 와 구분)
    reason: str           # "packet" | "link_timeout" | "caution_timeout" | "confidence"

    @property
    def lanes_visible(self) -> bool:
        return self.lane_opacity > OPACITY_FLOOR


class StateTracker:
    """패킷 수신 상태와 젯슨이 보낸 state 를 합쳐 화면에 띄울 state 를 정한다."""

    def __init__(self, config: dict[str, Any]) -> None:
        section = ensure_state_config(config)
        self.packet_timeout = float(config["network"]["packet_timeout_seconds"])
        self.fade_seconds = max(1e-3, float(section["lane_fade_seconds"]))
        self.caution_timeout = float(section["caution_timeout_seconds"])
        self.rise_frames = max(1, int(section["rise_frames"]))
        self.fall_frames = max(1, int(section["fall_frames"]))
        self.reset()

    def reset(self) -> None:
        # 첫 패킷을 받기 전에는 2 로 시작한다. 아직 아무것도 모르는 상태를
        # 녹색으로 띄우면 그것부터가 거짓말이다.
        self.state = STATE_LOST
        self.lane_state = STATE_NORMAL
        self.link_lost = True
        self.reason = "link_timeout"
        self._pending: int | None = None
        self._pending_count = 0
        self._caution_since: float | None = None
        self._fade_since: float | None = None   # None 이면 페이드가 이미 끝났다

    # 판정 ---------------------------------------------------------------

    def _candidate(self, now: float, raw_state: Any, confidence: float) -> tuple[int, str]:
        """이번 프레임이 요구하는 state 를 정한다. 디바운싱 이전 값이다."""
        if self.link_lost:
            self._caution_since = None
            return STATE_LOST, "link_timeout"

        state = normalize_state(raw_state)
        reason = "packet"
        # 젯슨이 state 를 안 실어 보내는 동안의 대체 신호. 판정은 여기 한
        # 곳에서만 하고 렌더러에는 이미 정해진 값을 넘긴다.
        if state == STATE_NORMAL and confidence < CONFIDENCE_FLOOR:
            state, reason = STATE_CAUTION, "confidence"

        if state == STATE_CAUTION:
            if self._caution_since is None:
                self._caution_since = now
            elif now - self._caution_since >= self.caution_timeout:
                return STATE_LOST, "caution_timeout"
        else:
            self._caution_since = None
        return state, reason

    def _frames_needed(self, candidate: int) -> int:
        if candidate == STATE_LOST:
            return 1                       # 경고 쪽으로는 지연을 두지 않는다
        if candidate > self.state:
            return self.rise_frames
        return self.fall_frames

    def _commit(self, state: int, now: float) -> None:
        if state == STATE_LOST:
            if self.state != STATE_LOST:
                self._fade_since = now     # 여기서부터 차선이 어두워진다
        else:
            self._fade_since = None
            self.lane_state = state
        self.state = state

    def _opacity(self, now: float) -> float:
        if self.state != STATE_LOST:
            return 1.0
        if self._fade_since is None:
            return 0.0
        remaining = 1.0 - (now - self._fade_since) / self.fade_seconds
        if remaining <= 0.0:
            self._fade_since = None
            return 0.0
        return min(1.0, remaining)

    def update(
        self,
        now: float,
        *,
        raw_state: Any = STATE_NORMAL,
        confidence: float = 1.0,
        last_received: float | None = None,
    ) -> HudStatus:
        """한 프레임 분 판정.

        now, last_received 는 같은 시계(time.monotonic)여야 한다.
        last_received 가 None 이면 아직 한 장도 못 받았다는 뜻이다.
        """
        self.link_lost = (
            last_received is None or (now - last_received) > self.packet_timeout
        )
        candidate, reason = self._candidate(now, raw_state, confidence)
        self.reason = reason

        if candidate == self.state:
            self._pending = None
            self._pending_count = 0
        else:
            if candidate != self._pending:
                self._pending = candidate
                self._pending_count = 0
            self._pending_count += 1
            if self._pending_count >= self._frames_needed(candidate):
                self._commit(candidate, now)
                self._pending = None
                self._pending_count = 0

        return HudStatus(
            state=self.state,
            lane_state=self.lane_state,
            lane_opacity=self._opacity(now),
            link_lost=self.link_lost,
            reason=self.reason,
        )
