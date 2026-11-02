#!/usr/bin/env python3
"""젯슨 계수 2차/3차 수용 테스트. 표준 unittest 로 돈다.

    cd ~/hud && python3 -m unittest discover -s tests -v

젯슨 없이 실제 UDP 로 LaneFeed 까지 흘려 보는 항목도 들어 있다. 화면 창은
띄우지 않는다.
"""

from __future__ import annotations

import copy
import json
import math
import socket
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import hud_system
from hud_system import (
    JETSON_LANE_SAMPLES,
    JETSON_LANE_SAMPLES_CUBIC,
    LaneSmoother,
    PacketError,
    _decode_packet,
    _lane_degree_text,
    _poly_lane_to_normalized,
    _sanitize_lane,
    load_config,
)
from hud_theme import ThemeRenderer
from hud_ui import HudRenderer, LaneFeed, _blank_debug, ensure_ui_config

# 2026-09-12 실측 패킷 (CLAUDE.md)
LEFT_2 = [-0.003151, 0.505674, 423.193818]
RIGHT_2 = [-0.008059, 5.896502, -500.737265]
Y_RANGE = [216.0, 410.4]
IMG_W, IMG_H = 960.0, 540.0

# y=216~410 에서 x 가 250 근처를 중심으로 S 자로 휘는 3차식. 변곡점 y=313.
Y_MID = 313.2
S_SHAPE = list(
    np.polyfit([216.0, 262.0, Y_MID, 364.0, 410.4], [330.0, 215.0, 250.0, 285.0, 170.0], 3)
)


def legacy_normalized(coefficients, y_range, samples=12):
    """변경 전 _poly_lane_to_normalized 의 계산을 그대로 옮긴 기준값."""
    a, b, c = (float(value) for value in coefficients)
    y_top, y_bottom = (float(value) for value in y_range)
    if y_bottom < y_top:
        y_top, y_bottom = y_bottom, y_top
    width = max(1.0, IMG_W - 1.0)
    height = max(1.0, IMG_H - 1.0)
    points = []
    for index in range(samples):
        progress = index / (samples - 1)
        y = y_top + (y_bottom - y_top) * progress
        x = (a * y + b) * y + c
        points.append([x / width, y / height])
    return points


def jetson_packet(**overrides):
    packet = {
        "v": 1, "seq": 292, "ts": 1789149939.2586713,
        "img_w": 960, "img_h": 540,
        "state": 0, "conf": 1.0,
        "left": LEFT_2, "right": RIGHT_2,
        "fps": 0.0, "id_left": 1, "id_right": 3,
        "event": "", "intent": "", "extra": None,
        "y_range": Y_RANGE,
    }
    packet.update(overrides)
    return packet


def decode(**overrides):
    return _decode_packet(json.dumps(jetson_packet(**overrides)).encode("utf-8"))


def bits(points):
    return [[float(x).hex(), float(y).hex()] for x, y in points]


def theme_config():
    config = load_config(ROOT / "hud_config.json")
    ensure_ui_config(config)
    config = copy.deepcopy(config)
    # 부트 애니메이션은 시계에 따라 선 길이가 달라져 픽셀 비교를 흐린다
    config["theme"]["boot_animation"] = False
    return config


class QuadraticUnchanged(unittest.TestCase):
    """계수 3개는 변경 전과 비트 단위로 같아야 한다."""

    CASES = [
        (LEFT_2, Y_RANGE),
        (RIGHT_2, Y_RANGE),
        (RIGHT_2, list(reversed(Y_RANGE))),
        ([0, 0, 480], [0, 539]),
        ([1e-9, -3.3e-3, 123.456789], [17.25, 501.75]),
        ([-0.0, 0.1, -0.0], [1.0, 2.5]),
        ([1e308, 1e308, 1e308], [216.0, 410.4]),
    ]

    def test_points_are_bit_identical(self):
        for coefficients, y_range in self.CASES:
            new = _poly_lane_to_normalized(coefficients, y_range, img_w=IMG_W, img_h=IMG_H)
            self.assertEqual(bits(new), bits(legacy_normalized(coefficients, y_range)))
            self.assertEqual(len(new), 12)

    def test_samples_constant_stays_12(self):
        self.assertEqual(JETSON_LANE_SAMPLES, 12)
        self.assertEqual(JETSON_LANE_SAMPLES_CUBIC, 20)

    def test_theme_pixels_identical(self):
        lanes = [legacy_normalized(LEFT_2, Y_RANGE), legacy_normalized(RIGHT_2, Y_RANGE)]
        expected = ThemeRenderer(theme_config()).render(
            lanes=hud_system._select_hud_lanes(lanes, lane_change=False), elapsed=0.0
        )
        packet = decode()
        actual = ThemeRenderer(theme_config()).render(lanes=packet["lanes"], elapsed=0.0)
        self.assertGreater(int(expected.sum()), 0)
        self.assertTrue(np.array_equal(expected, actual))


class CubicPoints(unittest.TestCase):
    def test_zero_cubic_term_matches_quadratic_shape(self):
        for quadratic in (LEFT_2, RIGHT_2):
            for samples in (12, 20):
                short = _poly_lane_to_normalized(
                    quadratic, Y_RANGE, img_w=IMG_W, img_h=IMG_H, samples=samples
                )
                long = _poly_lane_to_normalized(
                    [0, *quadratic], Y_RANGE, img_w=IMG_W, img_h=IMG_H, samples=samples
                )
                self.assertEqual(bits(short), bits(long))

    def test_default_sample_count_by_degree(self):
        cubic = _poly_lane_to_normalized([0, *LEFT_2], Y_RANGE, img_w=IMG_W, img_h=IMG_H)
        self.assertEqual(len(cubic), 20)

    def test_wire_order_is_highest_first(self):
        a, b, c, d = 1e-5, -2e-3, 0.5, 100.0
        points = _poly_lane_to_normalized([a, b, c, d], [100.0, 300.0], img_w=IMG_W, img_h=IMG_H)
        y = 200.0 + 100.0 * (2 * 10 / 19 - 1)  # 11번째 점
        expected = a * y**3 + b * y**2 + c * y + d
        self.assertAlmostEqual(points[10][0] * (IMG_W - 1.0), expected, places=6)
        self.assertAlmostEqual(
            points[10][0] * (IMG_W - 1.0), float(np.polyval([a, b, c, d], y)), places=6
        )

    def test_s_curve_keeps_inflection(self):
        points = np.asarray(
            _poly_lane_to_normalized(S_SHAPE, Y_RANGE, img_w=IMG_W, img_h=IMG_H)
        )
        # 2차 차분의 부호가 정확히 한 번 바뀌면 곡률이 뒤집힌 것이다
        curvature = np.diff(points[:, 0], 2)
        signs = np.sign(curvature[np.abs(curvature) > 1e-9])
        self.assertEqual(int(np.sum(signs[1:] != signs[:-1])), 1)
        flip = int(np.flatnonzero(signs[1:] != signs[:-1])[0])
        inflection_y = points[flip + 1, 1] * (IMG_H - 1.0)
        self.assertLess(abs(inflection_y - Y_MID), 25.0)

    def test_s_curve_survives_sanitize(self):
        lane = _sanitize_lane(
            _poly_lane_to_normalized(S_SHAPE, Y_RANGE, img_w=IMG_W, img_h=IMG_H)
        )
        self.assertEqual(len(lane), 20)

    def test_s_curve_needs_more_than_quadratic(self):
        """같은 양 끝을 잇는 2차와 3차가 화면에서 실제로 다른 선이어야 한다."""
        ys = np.linspace(Y_RANGE[0], Y_RANGE[1], 50)
        best_quadratic = np.polyfit(ys, np.polyval(S_SHAPE, ys), 2)
        error = np.abs(np.polyval(best_quadratic, ys) - np.polyval(S_SHAPE, ys))
        self.assertGreater(float(error.max()), 20.0)


class Rejection(unittest.TestCase):
    BAD_COEFFICIENTS = [
        [1.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, float("nan"), 4.0],
        [float("inf"), 2.0, 3.0, 4.0],
        [1.0, 2.0, float("-inf")],
        [1.0, 2.0, "3", 4.0],
        [1.0, True, 3.0],
        [1.0, None, 3.0, 4.0],
        [],
        "abcd",
        {"a": 1, "b": 2, "c": 3},
        3.0,
    ]

    def test_bad_coefficients_rejected_on_left_and_right(self):
        for side in ("left", "right"):
            for bad in self.BAD_COEFFICIENTS:
                with self.subTest(side=side, bad=bad):
                    with self.assertRaises(PacketError):
                        decode(**{side: bad})

    def test_bad_side_rejected_even_with_good_extra(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                with self.assertRaises(PacketError):
                    decode(extra=S_SHAPE, **{side: [1.0, 2.0]})

    def test_bad_extra_keeps_broken_packet_rejected(self):
        """extra 를 버려 준다고 다른 필드의 거부까지 풀리면 안 된다."""
        for overrides in (dict(y_range=None), dict(img_w=None), dict(seq=-1), dict(v=2)):
            with self.subTest(overrides=overrides):
                with self.assertRaises(PacketError):
                    decode(extra=[1.0, 2.0], **overrides)


class BadExtra(unittest.TestCase):
    """잘못된 extra 는 extra 만 버린다. left/right 는 그대로 그린다."""

    BAD = Rejection.BAD_COEFFICIENTS + [0, True, 10**400, [10**400, 1.0, 2.0]]

    def test_only_extra_is_dropped(self):
        clean = decode()
        for bad in self.BAD:
            with self.subTest(bad=repr(bad)[:40]):
                packet = decode(extra=bad)
                self.assertEqual(packet["lanes"], clean["lanes"])
                self.assertEqual(packet["lane_ids"], clean["lane_ids"])
                self.assertEqual(packet["status"], clean["status"])
                self.assertEqual(packet["lane_degrees"], {"left": 2, "right": 2})

    def test_extra_needs_valid_y_range_but_left_right_may_be_null(self):
        """차선이 없으면 y_range 도 null 이다. extra 만 와도 패킷은 살아야 한다."""
        packet = decode(left=None, right=None, y_range=None, extra=S_SHAPE)
        self.assertEqual(packet["lanes"], [])
        self.assertEqual(packet["lane_degrees"], {})
        packet = decode(left=None, right=None, y_range=[216.0, 216.5], extra=S_SHAPE)
        self.assertEqual(packet["lanes"], [])
        packet = decode(left=None, right=None, y_range=[10**400, 410.4], extra=S_SHAPE)
        self.assertEqual(packet["lanes"], [])

    def test_pixels_same_as_no_extra(self):
        expected = ThemeRenderer(theme_config()).render(lanes=decode()["lanes"], elapsed=0.0)
        for bad in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, float("nan"), 3.0, 4.0]):
            with self.subTest(bad=bad):
                actual = ThemeRenderer(theme_config()).render(
                    lanes=decode(extra=bad)["lanes"], elapsed=0.0
                )
                self.assertTrue(np.array_equal(expected, actual))

    def test_good_extra_still_shown_in_debug(self):
        self.assertEqual(decode(extra=S_SHAPE)["lane_degrees"]["extra"], 3)
        self.assertEqual(decode(extra=LEFT_2)["lane_degrees"]["extra"], 2)

    def test_broken_packets_rejected(self):
        broken = [
            dict(y_range=None),
            dict(y_range=[216.0]),
            dict(y_range=[216.0, float("nan")]),
            dict(y_range=[216.0, 216.5]),
            dict(img_w=None),
            dict(img_h="540"),
            dict(img_w=1),
            dict(v=2),
            dict(seq=-1),
            dict(seq="1"),
            dict(status="weird"),
        ]
        for overrides in broken:
            for left in (LEFT_2, [0.0, *LEFT_2]):
                with self.subTest(overrides=overrides, left=left):
                    with self.assertRaises(PacketError):
                        decode(left=left, **overrides)

    def test_invalid_payloads_rejected(self):
        for payload in (b"", b"not json", b"[1,2,3]", b"\xff\xfe", b"x" * 60_001):
            with self.subTest(payload=payload[:8]):
                with self.assertRaises(PacketError):
                    _decode_packet(payload)


class Adapter(unittest.TestCase):
    def test_mixed_degree_packet(self):
        packet = decode(left=S_SHAPE, right=RIGHT_2)
        self.assertEqual(packet["lane_degrees"], {"left": 3, "right": 2})
        self.assertEqual(sorted(len(lane) for lane in packet["lanes"]), [12, 20])
        self.assertEqual(len(packet["lane_ids"]), 2)

    def test_mixed_degree_renders_on_both_renderers(self):
        packet = decode(left=S_SHAPE, right=RIGHT_2)
        lanes = LaneSmoother(0.55, 0.18, 0.05, 3).update(packet["lanes"], packet["lane_ids"])
        config = theme_config()
        theme = ThemeRenderer(config).render(lanes=lanes, elapsed=0.0)
        self.assertGreater(int(theme.sum()), 0)
        plain = HudRenderer(config).render(lanes=lanes, elapsed=0.0)
        self.assertGreater(int(plain.sum()), 0)

    def test_extra_is_validated_but_not_drawn(self):
        without = decode()
        for extra in (LEFT_2, S_SHAPE):
            packet = decode(extra=extra)
            self.assertEqual(packet["lanes"], without["lanes"])
            self.assertEqual(packet["lane_ids"], without["lane_ids"])
            self.assertEqual(packet["lane_degrees"]["extra"], len(extra) - 1)

    def test_null_side(self):
        packet = decode(left=None, right=S_SHAPE)
        self.assertEqual(packet["lane_degrees"], {"right": 3})
        self.assertEqual(len(packet["lanes"]), 1)

    def test_extra_does_not_change_wire_keys(self):
        packet = decode(left=S_SHAPE)
        self.assertEqual(packet["left"], S_SHAPE)
        self.assertEqual(packet["right"], RIGHT_2)
        self.assertIsNone(packet["extra"])

    def test_smoother_survives_degree_flip(self):
        smoother = LaneSmoother(0.55, 0.18, 0.05, 3)
        for seq, left in enumerate([LEFT_2, S_SHAPE, LEFT_2, S_SHAPE, S_SHAPE, LEFT_2]):
            packet = decode(seq=seq, left=left)
            lanes = smoother.update(packet["lanes"], packet["lane_ids"])
            self.assertEqual(len(lanes), 2)
            for lane in lanes:
                self.assertTrue(np.isfinite(np.asarray(lane)).all())


class DebugLine(unittest.TestCase):
    def test_text(self):
        self.assertEqual(_lane_degree_text({"left": 3, "right": 2}), "deg L3 R2")
        self.assertEqual(_lane_degree_text({"left": 2, "right": 2}), "deg L2 R2")
        self.assertEqual(_lane_degree_text({"right": 3}), "deg L- R3")
        self.assertEqual(_lane_degree_text({}), "deg L- R-")
        self.assertEqual(_lane_degree_text({"left": 2, "right": 2, "extra": 3}), "deg L2 R2 E3")
        self.assertEqual(_lane_degree_text(None), "deg --")

    def test_debug_line_only_changes_when_shown(self):
        packet = decode(left=S_SHAPE)
        renderer = ThemeRenderer(theme_config())
        hidden = renderer.render(lanes=packet["lanes"], elapsed=0.0,
                                 debug=_blank_debug(lane_degrees={"left": 3, "right": 2}))
        plain = renderer.render(lanes=packet["lanes"], elapsed=0.0)
        self.assertTrue(np.array_equal(hidden, plain))
        shown = renderer.render(
            lanes=packet["lanes"], elapsed=0.0,
            debug=_blank_debug(show=True, lane_degrees={"left": 3, "right": 2}),
        )
        other = renderer.render(
            lanes=packet["lanes"], elapsed=0.0,
            debug=_blank_debug(show=True, lane_degrees={"left": 2, "right": 2}),
        )
        self.assertFalse(np.array_equal(shown, plain))
        self.assertFalse(np.array_equal(shown, other))


class RealUdp(unittest.TestCase):
    """실제 소켓으로 LaneFeed 까지. 재현성을 위해 루프백을 쓴다."""

    def setUp(self):
        config = theme_config()
        config["display"]["smoothing_alpha"] = 1.0
        config["display"]["jump_gate_distance"] = 0.0
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.feed = LaneFeed(config, "127.0.0.1", self.port)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def tearDown(self):
        self.sender.close()
        self.feed.close()

    def send(self, **overrides):
        self.seq += 1
        payload = json.dumps(jetson_packet(seq=self.seq, ts=time.time(), **overrides))
        self.sender.sendto(payload.encode("utf-8"), ("127.0.0.1", self.port))

    def poll(self):
        deadline = time.monotonic() + 1.0
        before = (self.feed.last_sequence, self.feed.dropped)
        while time.monotonic() < deadline:
            self.feed.poll(0.01)
            if (self.feed.last_sequence, self.feed.dropped) != before:
                return
        self.fail("packet did not arrive")

    def lane_lengths(self):
        return sorted(len(lane) for lane in self.feed.latest["lanes"])

    def test_combinations(self):
        cases = [
            (dict(left=LEFT_2, right=RIGHT_2), [12, 12], "deg L2 R2"),
            (dict(left=S_SHAPE, right=[0.0, *RIGHT_2]), [20, 20], "deg L3 R3"),
            (dict(left=S_SHAPE, right=RIGHT_2), [12, 20], "deg L3 R2"),
            (dict(left=LEFT_2, right=[0.0, *RIGHT_2]), [12, 20], "deg L2 R3"),
            (dict(left=None, right=S_SHAPE), [20], "deg L- R3"),
            (dict(left=LEFT_2, right=RIGHT_2, extra=S_SHAPE), [12, 12], "deg L2 R2 E3"),
            (dict(left=LEFT_2, right=RIGHT_2), [12, 12], "deg L2 R2"),
        ]
        renderer = ThemeRenderer(theme_config())
        for overrides, lengths, text in cases:
            with self.subTest(text=text):
                self.send(**overrides)
                self.poll()
                self.assertEqual(self.feed.dropped, 0)
                self.assertEqual(self.lane_lengths(), lengths)
                args = self.feed.frame(time.monotonic())
                self.assertEqual(
                    _lane_degree_text(args["debug"]["lane_degrees"]), text
                )
                args["debug"]["show"] = True
                frame = renderer.render(elapsed=0.0, **args)
                self.assertGreater(int(frame.sum()), 0)

    def test_quadratic_over_udp_matches_legacy(self):
        self.send()
        self.poll()
        expected = hud_system._select_hud_lanes(
            [legacy_normalized(LEFT_2, Y_RANGE), legacy_normalized(RIGHT_2, Y_RANGE)],
            lane_change=False,
        )
        fresh = LaneSmoother(1.0, 0.18, 0.0, 3).update(expected, [1, 3])
        self.assertEqual(self.feed.latest["lanes"], fresh)

    def test_bad_packets_counted_and_last_good_kept(self):
        self.send(left=LEFT_2, right=S_SHAPE)
        self.poll()
        good = copy.deepcopy(self.feed.latest["lanes"])
        for index, bad in enumerate(
            ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, float("nan"), 3.0, 4.0]), start=1
        ):
            self.send(left=bad)
            self.poll()
            self.assertEqual(self.feed.dropped, index)
            self.assertEqual(self.feed.latest["lanes"], good)

    def test_bad_extra_over_udp_keeps_lanes(self):
        self.send()
        self.poll()
        good = copy.deepcopy(self.feed.latest["lanes"])
        for bad in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, float("nan"), 3.0, 4.0], "abc"):
            with self.subTest(bad=bad):
                self.send(extra=bad)
                self.poll()
                self.assertEqual(self.feed.dropped, 0)
                self.assertEqual(self.feed.latest["lanes"], good)
                args = self.feed.frame(time.monotonic())
                self.assertEqual(_lane_degree_text(args["debug"]["lane_degrees"]), "deg L2 R2")

    def test_huge_extra_does_not_kill_receive_loop(self):
        self.send(extra=[10**400, 1.0, 2.0])
        self.poll()
        self.assertEqual(self.feed.dropped, 0)
        self.assertEqual(self.lane_lengths(), [12, 12])

    def test_degree_flip_every_frame(self):
        for frame in range(30):
            self.send(left=S_SHAPE if frame % 2 else LEFT_2, right=RIGHT_2 if frame % 3 else S_SHAPE)
            self.poll()
            self.assertEqual(self.feed.dropped, 0)
            self.assertEqual(len(self.feed.latest["lanes"]), 2)
            for lane in self.feed.latest["lanes"]:
                for x, y in lane:
                    self.assertTrue(math.isfinite(x) and math.isfinite(y))


if __name__ == "__main__":
    unittest.main()
