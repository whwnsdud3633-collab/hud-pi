#!/usr/bin/env python3
"""State 2 에서 레이더 전방 거리를 화면 가운데에 띄운다.

젯슨 HUD 메시지에 키 두 개가 붙는다(레이더 담당 인계 2026-11-03).

    dist_on : bool   거리를 띄워도 되는 상태
    dist_m  : float  전방 차량까지 거리(m), 0.5 m 단위

거리가 없는 프레임에는 키 자체가 없다. 깜빡임 방지와 평활은 젯슨이 이미
했으므로 파이는 받은 값을 그대로 띄운다. 다시 평균을 내지 않는다.

띄우는 조건 (전부 만족해야 한다):
  - radar.enabled 가 켜져 있다 (receive --no-radar 로 즉시 끔)
  - 화면 state 가 2 다. 차선이 보이는 0/1 에서는 띄우지 않는다
  - 마지막 패킷이 살아 있다. 무수신(link_lost)이면 숫자도 지운다
  - 마지막 패킷이 radar.max_age_s 보다 오래되지 않았다
  - dist_on 이 정확히 true 이고 dist_m 이 정상 숫자다

잘못된 dist_on / dist_m 이 와도 패킷은 버리지 않는다. 거리만 안 띄운다.
차선 표시에는 아무 영향이 없다.

이 파일은 import 할 때 numpy / cv2 / PIL 을 부르지 않는다. hud_system 이
패킷을 읽을 때 쓰는 read_radar 는 가볍게 남겨 둔다.
"""

from __future__ import annotations

import math
from typing import Any

RADAR_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # 마지막 패킷이 이보다 오래되면 거리를 지운다. 무수신 판정(0.35 s)보다 길게
    # 두면 무수신 쪽이 먼저 지운다. 둘 중 빠른 쪽이 이긴다.
    "max_age_s": 0.5,
    # 운전자 시점 기준 세로 위치 (0 위 ~ 1 아래)
    "y_ratio": 0.45,
    # 글자 높이 (패널 높이 비율)
    "number_height_ratio": 0.12,
    "label_height_ratio": 0.045,
    # BGR. 흰색에 가까운 연한 청록. 빨강은 쓰지 않는다 (state 2 상태 바와 혼동)
    "color_bgr": [255, 240, 190],
    "label": "전방",
    # 한글 폰트 경로. null 이면 아래 후보와 fc-match 로 찾는다
    "font_path": None,
}

# 받아 줄 거리 범위 (m). 밖이면 레이더 값이 아니라고 본다
DIST_MIN_M, DIST_MAX_M = 0.0, 250.0

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/unfonts-core/UnDotumBold.ttf",
)


def radar_settings(config: dict[str, Any]) -> dict[str, Any]:
    """config["radar"] 를 기본값 위에 얹는다. 키가 없어도 동작한다."""
    settings = dict(RADAR_DEFAULTS)
    raw = config.get("radar")
    if isinstance(raw, dict):
        settings.update(raw)
    settings["enabled"] = bool(settings["enabled"])
    return settings


def read_radar(packet: dict[str, Any]) -> float | None:
    """패킷의 dist_on / dist_m 을 읽는다. 띄울 거리가 없으면 None.

    무엇이 와도 예외를 내지 않는다. 레이더 키 때문에 차선 패킷이 버려지면 안 된다.
    """
    try:
        if packet.get("dist_on") is not True:
            return None
        value = packet.get("dist_m")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        if not math.isfinite(value) or not DIST_MIN_M < value <= DIST_MAX_M:
            return None
        return value
    except (OverflowError, TypeError, ValueError):
        return None


def _find_font(path: Any) -> str | None:
    import os
    import subprocess

    if isinstance(path, str) and path:
        return path if os.path.exists(path) else None
    for candidate in FONT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    try:
        found = subprocess.run(
            ["fc-match", "-f", "%{file}", ":lang=ko:weight=bold"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return found if found and os.path.exists(found) else None


class DistanceOverlay:
    """거리 글자를 패널 평면에 더한다. 글자 모양은 문자열마다 한 번만 만든다."""

    def __init__(
        self,
        config: dict[str, Any],
        width: int,
        height: int,
        *,
        flip_vertical: bool,
        flip_horizontal: bool,
    ) -> None:
        self.settings = radar_settings(config)
        self.width, self.height = int(width), int(height)
        self.flip_vertical = bool(flip_vertical)
        self.flip_horizontal = bool(flip_horizontal)
        self.color = tuple(float(c) for c in self.settings["color_bgr"])[:3]
        self._cache: dict[str, Any] = {}
        self._fonts = None
        self.font_path: str | None = None
        self.warned = False
        # 첫 State 2 에서 화면이 멈칫하지 않게 폰트는 시작할 때 읽어 둔다
        if self.enabled:
            self._load_fonts()

    @property
    def enabled(self) -> bool:
        return bool(self.settings["enabled"])

    # 글자 --------------------------------------------------------------

    def _load_fonts(self) -> None:
        if self._fonts is not None:
            return
        self._fonts = (None, None)
        self.font_path = _find_font(self.settings.get("font_path"))
        try:
            from PIL import ImageFont
        except ImportError:
            self.font_path = None
        if self.font_path is None:
            if not self.warned:
                print("[radar] 한글 폰트 또는 PIL 이 없어 숫자만 그립니다 "
                      "(radar.font_path 로 지정 가능)")
                self.warned = True
            return
        number_px = max(12, int(self.height * float(self.settings["number_height_ratio"])))
        label_px = max(10, int(self.height * float(self.settings["label_height_ratio"])))
        self._fonts = (
            ImageFont.truetype(self.font_path, label_px),
            ImageFont.truetype(self.font_path, number_px),
        )

    def _pil_text(self, font: Any, text: str) -> Any:
        import numpy as np
        from PIL import Image, ImageDraw

        left, top, right, bottom = font.getbbox(text)
        image = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
        ImageDraw.Draw(image).text((-left, -top), text, fill=255, font=font)
        return np.asarray(image, dtype=np.uint8)

    def _cv_text(self, text: str) -> Any:
        import cv2
        import numpy as np

        target = max(12, int(self.height * float(self.settings["number_height_ratio"])))
        scale = target / 22.0
        thick = max(2, int(round(scale * 2)))
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        mask = np.zeros((h + base + thick * 2, w + thick * 2), np.uint8)
        cv2.putText(mask, text, (thick, h + thick), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 255, thick, cv2.LINE_AA)
        return mask

    def mask_for(self, dist_m: float) -> Any:
        """운전자가 보는 방향 그대로의 글자 마스크 (uint8, 0~255)."""
        import numpy as np

        number = f"{dist_m:.0f} m"
        if number in self._cache:
            return self._cache[number]
        self._load_fonts()
        label_font, number_font = self._fonts
        if number_font is None:
            mask = self._cv_text(number)
        else:
            top = self._pil_text(label_font, str(self.settings["label"]))
            bottom = self._pil_text(number_font, number)
            gap = max(4, top.shape[0] // 3)
            width = max(top.shape[1], bottom.shape[1])
            mask = np.zeros((top.shape[0] + gap + bottom.shape[0], width), np.uint8)
            x = (width - top.shape[1]) // 2
            mask[: top.shape[0], x: x + top.shape[1]] = top
            x = (width - bottom.shape[1]) // 2
            mask[top.shape[0] + gap:, x: x + bottom.shape[1]] = bottom
        # 반사판 반전은 패널에 그릴 때의 반전과 똑같이 맞춘다.
        # 차선은 투영 단계에서, 상태 바는 행 위치로 반전하므로 글자도 여기서 뒤집는다.
        if self.flip_vertical:
            mask = mask[::-1]
        if self.flip_horizontal:
            mask = mask[:, ::-1]
        mask = np.ascontiguousarray(mask)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[number] = mask
        return mask

    # 그리기 ------------------------------------------------------------

    def draw(self, planes: list[Any], dist_m: float | None) -> bool:
        """BGR 평면 세 장(float)에 가산으로 더한다. 그렸으면 True."""
        if not self.enabled or dist_m is None:
            return False
        mask = self.mask_for(dist_m)
        h, w = mask.shape
        # 운전자 시점에서 자리를 정한 뒤 패널 쪽으로 정확히 거울상으로 옮긴다
        x0 = int(round(self.width / 2.0 - w / 2.0))
        y0 = int(round(self.height * float(self.settings["y_ratio"]) - h / 2.0))
        if self.flip_vertical:
            y0 = self.height - y0 - h
        if self.flip_horizontal:
            x0 = self.width - x0 - w
        # 화면 밖으로 나가는 부분은 자른다
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(self.width, x0 + w), min(self.height, y0 + h)
        if dx1 <= dx0 or dy1 <= dy0:
            return False
        part = mask[sy0: sy0 + (dy1 - dy0), sx0: sx0 + (dx1 - dx0)].astype("float32") / 255.0
        for channel in range(3):
            if self.color[channel]:
                planes[channel][dy0:dy1, dx0:dx1] += part * self.color[channel]
        return True
