#!/usr/bin/env python3
"""레이더 전방 거리 (dist_on / dist_m) 테스트. 표준 unittest 로 돈다.

    cd ~/hud && python3 -m unittest tests.test_radar -v
"""

from __future__ import annotations

import copy
import json
import socket
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from hud_radar import read_radar, radar_settings
from hud_system import _decode_packet, load_config
from hud_theme import STATE_CAUTION, STATE_LOST, STATE_NORMAL, ThemeRenderer
from hud_ui import LaneFeed, ensure_ui_config

LEFT_2 = [-0.003151, 0.505674, 423.193818]
RIGHT_2 = [-0.008059, 5.896502, -500.737265]
Y_RANGE = [216.0, 410.4]


def packet(**overrides):
    base = {
        "v": 1, "seq": 1, "ts": time.time(), "img_w": 960, "img_h": 540,
        "state": 0, "conf": 1.0, "left": LEFT_2, "right": RIGHT_2,
        "fps": 28.0, "id_left": 1, "id_right": 3, "event": "", "intent": "",
        "extra": None, "y_range": Y_RANGE,
    }
    base.update(overrides)
    return base


def decode(**overrides):
    return _decode_packet(json.dumps(packet(**overrides)).encode("utf-8"))


def theme_config(**radar):
    config = load_config(ROOT / "hud_config.json")
    ensure_ui_config(config)
    config = copy.deepcopy(config)
    config["theme"]["boot_animation"] = False
    if radar:
        config.setdefault("radar", {}).update(radar)
    return config


class ReadRadar(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(read_radar({"dist_on": True, "dist_m": 18.5}), 18.5)
        self.assertEqual(read_radar({"dist_on": True, "dist_m": 4}), 4.0)

    def test_absent_or_off(self):
        self.assertIsNone(read_radar({}))
        self.assertIsNone(read_radar({"dist_m": 12.0}))
        self.assertIsNone(read_radar({"dist_on": False, "dist_m": 12.0}))
        self.assertIsNone(read_radar({"dist_on": True}))

    def test_bad_values_ignored(self):
        for dist_on, dist_m in [
            ("true", 10.0), (1, 10.0), (True, "10"), (True, None), (True, True),
            (True, float("nan")), (True, float("inf")), (True, -3.0), (True, 0.0),
            (True, 999.0), (True, 10 ** 400), (True, [10.0]),
        ]:
            with self.subTest(dist_on=dist_on, dist_m=repr(dist_m)[:20]):
                self.assertIsNone(read_radar({"dist_on": dist_on, "dist_m": dist_m}))

    def test_settings_defaults(self):
        self.assertTrue(radar_settings({})["enabled"])
        self.assertFalse(radar_settings({"radar": {"enabled": False}})["enabled"])


class Decode(unittest.TestCase):
    def test_lanes_unchanged_by_radar_keys(self):
        plain = decode()
        for extra in ({"dist_on": True, "dist_m": 18.0},
                      {"dist_on": "yes", "dist_m": "x"},
                      {"dist_on": True, "dist_m": float("nan")},
                      {"dist_on": True, "dist_m": 10 ** 400}):
            with self.subTest(extra=repr(extra)[:40]):
                got = decode(**extra)
                self.assertEqual(got["lanes"], plain["lanes"])
                self.assertEqual(got["status"], plain["status"])

    def test_radar_field(self):
        self.assertIsNone(decode()["radar_m"])
        self.assertEqual(decode(dist_on=True, dist_m=12.5)["radar_m"], 12.5)
        self.assertIsNone(decode(dist_on=True, dist_m="12")["radar_m"])

    def test_lanes_packet_path_also_reads_radar(self):
        got = decode(lanes=[[[0.4, 0.4], [0.2, 0.9]], [[0.6, 0.4], [0.8, 0.9]]],
                     lane_ids=[1, 3], dist_on=True, dist_m=9.0)
        self.assertEqual(got["radar_m"], 9.0)


class Render(unittest.TestCase):
    def render(self, renderer, state, radar_m):
        telemetry = {} if radar_m is None else {"radar_m": radar_m}
        lanes = decode()["lanes"] if state != STATE_LOST else []
        return renderer.render(lanes=lanes, state=state, telemetry=telemetry, elapsed=0.0)

    def test_no_radar_is_identical(self):
        renderer = ThemeRenderer(theme_config())
        for state in (STATE_NORMAL, STATE_CAUTION, STATE_LOST):
            with self.subTest(state=state):
                a = self.render(renderer, state, None)
                b = ThemeRenderer(theme_config()).render(
                    lanes=decode()["lanes"] if state != STATE_LOST else [],
                    state=state, elapsed=0.0)
                self.assertTrue(np.array_equal(a, b))

    def test_only_state_2_draws(self):
        renderer = ThemeRenderer(theme_config())
        for state in (STATE_NORMAL, STATE_CAUTION):
            with self.subTest(state=state):
                self.assertTrue(np.array_equal(self.render(renderer, state, 18.0),
                                               self.render(renderer, state, None)))
        drawn = self.render(renderer, STATE_LOST, 18.0)
        empty = self.render(renderer, STATE_LOST, None)
        self.assertGreater(int((drawn != empty).any(2).sum()), 500)

    def test_disabled_draws_nothing(self):
        renderer = ThemeRenderer(theme_config(enabled=False))
        self.assertTrue(np.array_equal(self.render(renderer, STATE_LOST, 18.0),
                                       self.render(renderer, STATE_LOST, None)))

    def test_not_red(self):
        renderer = ThemeRenderer(theme_config())
        diff = (self.render(renderer, STATE_LOST, 18.0).astype(int)
                - self.render(renderer, STATE_LOST, None).astype(int))
        b, g, r = diff[..., 0].max(), diff[..., 1].max(), diff[..., 2].max()
        self.assertGreater(b, r)

    def test_flip_vertical_mirrors_text(self):
        def rows(flip):
            config = theme_config()
            config["display"]["flip_vertical"] = flip
            renderer = ThemeRenderer(config)
            diff = (self.render(renderer, STATE_LOST, 18.0)
                    != self.render(renderer, STATE_LOST, None)).any(2)
            ys = np.nonzero(diff.any(1))[0]
            return renderer.height, ys.min(), ys.max(), diff
        h, top_a, bottom_a, a = rows(False)
        _, top_b, bottom_b, b = rows(True)
        self.assertAlmostEqual(top_a, h - 1 - bottom_b, delta=2)
        self.assertTrue(np.array_equal(a, b[::-1]))

    def test_distance_changes_text(self):
        renderer = ThemeRenderer(theme_config())
        self.assertFalse(np.array_equal(self.render(renderer, STATE_LOST, 18.0),
                                        self.render(renderer, STATE_LOST, 7.0)))


class Feed(unittest.TestCase):
    def setUp(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.feed = None

    def tearDown(self):
        self.sender.close()
        if self.feed is not None:
            self.feed.close()

    def open(self, **radar):
        config = theme_config(**radar)
        self.feed = LaneFeed(config, "127.0.0.1", self.port)

    def send(self, **overrides):
        self.seq += 1
        body = json.dumps(packet(seq=self.seq, ts=time.time(), **overrides)).encode()
        self.sender.sendto(body, ("127.0.0.1", self.port))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.feed.poll(0.01)
            if self.feed.last_sequence == self.seq:
                return
        self.fail("packet did not arrive")

    def settle(self, **overrides):
        """상태 디바운싱이 끝날 때까지 같은 패킷을 보낸다."""
        args = None
        for _ in range(20):
            self.send(**overrides)
            args = self.feed.frame(time.monotonic())
            time.sleep(0.02)
        return args

    def test_state_2_with_distance(self):
        self.open()
        args = self.settle(state=2, left=None, right=None, y_range=None,
                           dist_on=True, dist_m=15.0)
        self.assertEqual(args["state"], STATE_LOST)
        self.assertEqual(args["telemetry"].get("radar_m"), 15.0)

    def test_state_0_never_shows(self):
        self.open()
        args = self.settle(state=0, dist_on=True, dist_m=15.0)
        self.assertNotIn("radar_m", args["telemetry"])

    def test_disabled(self):
        self.open(enabled=False)
        args = self.settle(state=2, left=None, right=None, y_range=None,
                           dist_on=True, dist_m=15.0)
        self.assertNotIn("radar_m", args["telemetry"])

    def test_link_loss_clears_distance(self):
        self.open()
        self.settle(state=2, left=None, right=None, y_range=None, dist_on=True, dist_m=15.0)
        args = self.feed.frame(time.monotonic() + 1.0)
        self.assertNotIn("radar_m", args["telemetry"])
        self.assertEqual(args["state"], STATE_LOST)

    def test_missing_keys_clear_distance(self):
        self.open()
        self.settle(state=2, left=None, right=None, y_range=None, dist_on=True, dist_m=15.0)
        args = self.settle(state=2, left=None, right=None, y_range=None)
        self.assertNotIn("radar_m", args["telemetry"])


if __name__ == "__main__":
    unittest.main()
