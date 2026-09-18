#!/usr/bin/env python3
"""실데이터를 보며 차선 오버레이를 손으로 맞추는 미세 보정 도구.

    python3 hud_align.py trim              # 젯슨 패킷을 받으며 보정
    python3 hud_align.py trim --mock       # 젯슨 없이 가짜 차선으로

층을 셋으로 나눈다.

    입력 소스  ->  TrimCommand  ->  TrimAdjuster  ->  TrimParams  ->  AlignmentMap
    (키보드)       (의도)           (보정 로직)       (값 4개)        (투영)

입력 소스는 "무엇을 하라" 는 TrimCommand 만 만든다. 키 코드, 인코더 틱,
UDP 메시지 같은 입력 장치 사정은 여기서 끝난다. TrimAdjuster 는 명령을
받아 값을 바꾸고 저장·종료 확인을 처리할 뿐 입력이 어디서 왔는지 모른다.
나중에 로터리 인코더나 UDP 로 갈아끼울 때는 poll() 이 TrimCommand 목록을
돌려주는 클래스 하나만 새로 만들면 된다. 절대값을 주는 소스(아이트래킹 등)는
TrimCommand.set 으로 값을 통째로 넘긴다.

값 4개가 기존 정렬과 어떻게 합쳐지는지는 hud_align.TrimParams 에 있다.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from hud_align import TrimParams, ensure_alignment_config
from hud_system import _mock_lane, load_config, save_config


# ---------------------------------------------------------------------------
# 명령: 입력 장치와 보정 로직 사이의 약속
# ---------------------------------------------------------------------------

MOVE, KEYSTONE, ROLL = "move", "keystone", "roll"
RESET, SAVE, QUIT, TOGGLE_STEP, SET = "reset", "save", "quit", "toggle_step", "set"
OTHER = "other"     # 뜻 없는 입력. 종료 확인을 취소하는 데만 쓰인다


@dataclass(frozen=True)
class TrimCommand:
    """보정 명령 하나. 방향은 전부 운전자가 보는 화면 기준이다.

    move      dx, dy 는 -1/0/+1 방향. +x 오른쪽, +y 아래
    keystone  sign +1 이면 위쪽 폭이 넓어진다
    roll      sign +1 이면 시계 방향 (오른쪽이 내려감)
    set       value 로 값을 통째로 바꾼다. 절대값을 주는 소스용
    other     그 밖의 입력. 값은 안 바꾸고 종료 확인만 취소한다
    coarse    None 이면 지금 선택된 폭, True/False 면 이번 한 번만 그 폭
    count     인코더처럼 여러 틱이 한꺼번에 들어오는 소스용 배수
    """

    action: str
    dx: int = 0
    dy: int = 0
    sign: int = 0
    coarse: bool | None = None
    count: int = 1
    value: TrimParams | None = None


class TrimInput(Protocol):
    """입력 소스. 한 프레임에 한 번 불리고 그사이 쌓인 명령을 돌려준다."""

    def poll(self) -> list[TrimCommand]: ...


# ---------------------------------------------------------------------------
# 보정 로직
# ---------------------------------------------------------------------------


class TrimAdjuster:
    """명령을 받아 TrimParams 를 바꾸고 저장·종료 확인을 처리한다.

    handle() 이 돌려주는 사건:
        changed        값이 바뀌었다
        step           큰 폭/작은 폭이 바뀌었다
        saved          저장했다
        confirm_quit   저장 안 한 채 종료하려 해서 확인을 기다린다
        cancel         확인 대기 중에 다른 명령이 와서 종료를 취소했다
        quit           종료
        none           아무 일도 없었다
    """

    def __init__(
        self,
        trim: TrimParams,
        steps: dict[str, dict[str, float]],
        width: int,
        height: int,
        on_change: Callable[[TrimParams], None],
        on_save: Callable[[TrimParams], None],
    ) -> None:
        self.trim = trim
        self.saved = trim
        self.steps = steps
        self.width, self.height = width, height
        self.coarse = False
        self.confirming_quit = False
        self._on_change = on_change
        self._on_save = on_save

    @property
    def dirty(self) -> bool:
        return self.trim != self.saved

    def _step(self, coarse: bool | None) -> dict[str, float]:
        use = self.coarse if coarse is None else coarse
        return self.steps["coarse" if use else "fine"]

    def _set(self, trim: TrimParams) -> str:
        trim = trim.clamped(self.width, self.height)
        # 부동소수 누적 오차로 0.30000000004 같은 값이 저장되지 않게 한다
        trim = TrimParams(**trim.to_config())
        if trim == self.trim:
            return "none"
        self.trim = trim
        self._on_change(trim)
        return "changed"

    def _save(self) -> None:
        self._on_save(self.trim)
        self.saved = self.trim

    def handle(self, command: TrimCommand) -> str:
        action = command.action
        if self.confirming_quit:
            # 저장 안 된 채 q 를 눌렀다. 한 번 더 q 면 버리고 종료,
            # Enter 면 저장하고 종료, 그 밖의 명령은 종료를 취소하고 버린다.
            self.confirming_quit = False
            if action == QUIT:
                return "quit"
            if action == SAVE:
                self._save()
                return "quit"
            return "cancel"

        if action == QUIT:
            if self.dirty:
                self.confirming_quit = True
                return "confirm_quit"
            return "quit"
        if action == SAVE:
            self._save()
            return "saved"
        if action == TOGGLE_STEP:
            self.coarse = not self.coarse
            return "step"
        if action == RESET:
            return self._set(TrimParams())
        if action == SET and command.value is not None:
            return self._set(command.value)

        step = self._step(command.coarse)
        n = command.count
        t = self.trim
        if action == MOVE:
            return self._set(replace(
                t,
                dx_px=t.dx_px + command.dx * n * step["move_px"],
                dy_px=t.dy_px + command.dy * n * step["move_px"],
            ))
        if action == KEYSTONE:
            return self._set(replace(t, keystone=t.keystone + command.sign * n * step["keystone"]))
        if action == ROLL:
            return self._set(replace(t, roll_deg=t.roll_deg + command.sign * n * step["roll_deg"]))
        return "none"


# ---------------------------------------------------------------------------
# 입력 소스: 키보드 (OpenCV 창)
# ---------------------------------------------------------------------------

# cv2.waitKeyEx 가 돌려주는 방향키 코드. 리눅스(Qt, GTK)는 X keysym 을,
# 윈도우는 가상키 코드를 준다. 하위 8비트만 보면 방향키가 대문자 Q R S T
# 와 겹치므로 전체 코드로만 비교한다.
ARROW_LEFT = (65361, 2424832)
ARROW_UP = (65362, 2490368)
ARROW_RIGHT = (65363, 2555904)
ARROW_DOWN = (65364, 2621440)

KEY_ENTER = (13, 10)
KEY_ESC = 27
KEY_TAB = 9
# Shift, Ctrl, Caps Lock, Alt, Super 를 단독으로 누른 것. 종료 확인 중에
# 이것만으로 취소되면 곤란하므로 무시한다.
MODIFIER_KEYS = range(65505, 65519)


def key_to_command(key: int) -> TrimCommand | None:
    """키 코드 하나를 명령으로 바꾼다. 수식키만 누른 것은 None, 모르는 키는 OTHER.

    큰 폭/작은 폭은 f 또는 Tab 으로 켜고 끈다. Shift 조합은 쓰지 않는다.
    이 파이의 OpenCV(Qt 백엔드) 는 waitKeyEx 에 Shift 상태를 싣지 않아
    Shift+방향키가 방향키와 같은 코드로 오고, Shift+w 도 'w' 로 왔다.
    글자 키는 대소문자를 가리지 않는다. Caps Lock 이 켜져 있어도 된다.
    """
    if key in ARROW_LEFT:
        return TrimCommand(MOVE, dx=-1)
    if key in ARROW_RIGHT:
        return TrimCommand(MOVE, dx=1)
    if key in ARROW_UP:
        return TrimCommand(MOVE, dy=-1)
    if key in ARROW_DOWN:
        return TrimCommand(MOVE, dy=1)
    if key in KEY_ENTER:
        return TrimCommand(SAVE)
    if key in (KEY_ESC, ord("q"), ord("Q")):
        return TrimCommand(QUIT)
    if key in (KEY_TAB, ord("f"), ord("F")):
        return TrimCommand(TOGGLE_STEP)
    if key == ord("0"):
        return TrimCommand(RESET)
    letters = {
        "w": (KEYSTONE, 1),     # 위쪽 폭 넓게
        "s": (KEYSTONE, -1),    # 위쪽 폭 좁게
        "a": (ROLL, -1),        # 반시계. 왼쪽이 내려가고 오른쪽이 올라감
        "d": (ROLL, 1),         # 시계. 오른쪽이 내려가고 왼쪽이 올라감
    }
    if 0 <= key < 256 and chr(key).lower() in letters:
        action, sign = letters[chr(key).lower()]
        return TrimCommand(action, sign=sign)
    if key in MODIFIER_KEYS:
        return None
    return TrimCommand(OTHER)


class KeyboardTrimInput:
    """OpenCV 창의 키 입력을 명령으로 바꾼다.

    waitKeyEx 는 창 이벤트 처리까지 겸하므로 화면 루프가 부른다. 루프는
    받은 키를 push() 로 넘기기만 하고, 해석은 여기서 한다.
    """

    def __init__(self) -> None:
        self._pending: list[TrimCommand] = []

    def push(self, key: int) -> None:
        if key == -1:
            return
        command = key_to_command(key)
        if command is not None:
            self._pending.append(command)

    def poll(self) -> list[TrimCommand]:
        pending, self._pending = self._pending, []
        return pending


# ---------------------------------------------------------------------------
# 젯슨 없이 돌리는 가짜 입력
# ---------------------------------------------------------------------------


class MockFeed:
    """LaneFeed 와 같은 모양으로 가짜 차선 두 줄을 준다."""

    def poll(self, timeout: float) -> None:
        time.sleep(timeout)

    def frame(self, now: float) -> dict[str, Any]:
        lanes = [_mock_lane(0.18, 0.43, 0.0), _mock_lane(0.82, 0.57, 0.3)]
        return dict(lanes=lanes, warning="none", state=0, debug=None)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 화면 표시
# ---------------------------------------------------------------------------

OVERLAY_COLOR = (110, 110, 110)
CONFIRM_COLOR = (80, 200, 255)


def _overlay(
    width: int, height: int, lines: list[tuple[str, tuple[int, int, int]]], mirror: bool
) -> np.ndarray:
    """안내 글자를 그린 레이어. 반사 광학계에서 바로 읽히도록 반전해 둔다."""
    layer = np.zeros((height, width, 3), np.uint8)
    for index, (text, color) in enumerate(lines):
        cv2.putText(layer, text, (20, 32 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
    return cv2.flip(layer, 1) if mirror else layer


def _describe(adjuster: TrimAdjuster) -> str:
    t = adjuster.trim
    return (f"TRIM x {t.dx_px:+.0f}  y {t.dy_px:+.0f}  key {t.keystone:+.3f}  "
            f"roll {t.roll_deg:+.2f}   STEP {'coarse' if adjuster.coarse else 'fine'}"
            + ("  *" if adjuster.dirty else ""))


def run_trim(args: argparse.Namespace) -> None:
    # 렌더러와 수신 경로는 receive 와 똑같은 것을 쓴다
    from hud_ui import LaneFeed, _open_window, ensure_ui_config, make_renderer

    config = load_config(args.config)
    ensure_ui_config(config)
    section = ensure_alignment_config(config)
    renderer = make_renderer(config, False)
    mapper = renderer.mapper
    display = config["display"]

    if args.mock:
        feed: Any = MockFeed()
        source_label = "MOCK"
    else:
        network = config["network"]
        bind_host = args.bind_host or str(network["bind_host"])
        port = args.port or int(network["port"])
        feed = LaneFeed(config, bind_host, port)
        source_label = f"UDP {port}"
        print(f"listening on {bind_host}:{port}. stop 'hud_ui.py receive' first "
              f"if it is running, or it will take the packets.")

    def on_change(trim: TrimParams) -> None:
        mapper.set_trim(trim)

    def on_save(trim: TrimParams) -> None:
        # 렌더러가 채워 넣은 기본값까지 파일에 쓰지 않도록 디스크 본을
        # 다시 읽어 trim 만 바꾼다.
        disk = load_config(args.config)
        ensure_alignment_config(disk)["trim"] = trim.to_config()
        save_config(args.config, disk)
        print(f"saved trim to {args.config}: {trim.to_config()}")

    adjuster = TrimAdjuster(
        mapper.trim, section["trim_step"], mapper.width, mapper.height,
        on_change, on_save,
    )
    inputs: list[TrimInput] = []
    keyboard = KeyboardTrimInput()
    inputs.append(keyboard)

    name = "HUD trim"
    _open_window(name, renderer, args.windowed, bool(display["fullscreen"]))
    print("arrows move, w/s keystone, a/d roll, f or tab coarse/fine,")
    print("0 reset, enter save, q quit")
    print(_describe(adjuster))

    mirror = mapper.physical_flip_horizontal
    help_line = "arrows move  w/s keystone  a/d roll  f step  0 reset  enter save  q quit"
    overlay_key: tuple[Any, ...] | None = None
    overlay = None
    started = time.monotonic()

    try:
        while True:
            feed.poll(0.005)
            now = time.monotonic()
            frame_args = feed.frame(now)
            canvas = renderer.render(elapsed=now - started, **frame_args)

            debug = frame_args.get("debug") or {}
            rx = "RX timeout" if debug.get("status") == "timeout" else f"{source_label} ok"
            key_state = (_describe(adjuster), adjuster.confirming_quit, rx)
            if key_state != overlay_key:
                overlay_key = key_state
                lines = [(key_state[0] + "   " + rx, OVERLAY_COLOR),
                         (help_line, OVERLAY_COLOR)]
                if adjuster.confirming_quit:
                    lines.append(("UNSAVED. enter: save+quit  q: discard+quit  other: cancel",
                                  CONFIRM_COLOR))
                overlay = _overlay(mapper.width, mapper.height, lines, mirror)
            cv2.add(canvas, overlay, dst=canvas)
            cv2.imshow(name, canvas)

            keyboard.push(cv2.waitKeyEx(1))
            done = False
            for source in inputs:
                for command in source.poll():
                    event = adjuster.handle(command)
                    if event in ("changed", "step"):
                        print(_describe(adjuster))
                    elif event == "confirm_quit":
                        print("unsaved changes. enter: save and quit, "
                              "q: quit without saving, any other key: cancel")
                    elif event == "cancel":
                        print("quit cancelled")
                    elif event == "quit":
                        done = True
                        break
                if done:
                    break
            if done:
                break
    except KeyboardInterrupt:
        if adjuster.dirty:
            print("interrupted. unsaved trim was discarded.")
    finally:
        feed.close()
        cv2.destroyAllWindows()
