# -*- coding: utf-8 -*-
"""R4 입력 계층 단위 테스트 (mock — 실제 키 없음, stdlib만으로 가능)."""
import sys
import unittest
from unittest import mock

sys.path.insert(0, r"C:\Users\ROCmAdmin\Desktop\test\server")
from core.input_controller import InputController, ACTION_KEYS


class TestInputController(unittest.TestCase):
    def setUp(self):
        self.ctl = InputController(hwnd=0, dry_run=True)
        self.ctl.arm(True)
        # 테스트 환경엔 포커스된 게임 창이 없으므로 포커스 게이트만 스텁
        patcher = mock.patch.object(
            InputController, "game_has_focus", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_not_armed_blocks(self):
        self.ctl.arm(False)
        ok, why = self.ctl.act(0)
        self.assertFalse(ok)
        self.assertEqual(why, "NOT_ARMED")

    def test_kill_switch_blocks(self):
        # KILL 파일 채널 트리거 (F12 채널은 이 세션에서 감지 불가 — 실측 확인)
        import os
        kf = self.ctl.kill.kill_file
        os.makedirs(os.path.dirname(kf), exist_ok=True)
        open(kf, "w").write("stop")
        self.ctl.kill._file_triggered = True
        self.ctl.kill._event.set()
        ok, why = self.ctl.act(0)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("KILL_SWITCH"))
        self.ctl.kill._event.clear()
        if os.path.exists(kf):
            os.remove(kf)

    def test_stay_action_no_key(self):
        with mock.patch.object(self.ctl, "_tap") as tap:
            ok, why = self.ctl.act(4)
            self.assertTrue(ok)
            self.assertEqual(why, "stay")
            tap.assert_not_called()

    def test_action_maps(self):
        with mock.patch.object(self.ctl, "_tap") as tap:
            self.ctl.act(0)   # 좌 + 서브 0
            tap.assert_called_once_with(ACTION_KEYS["left"], 0.10)
            tap.reset_mock()
            self.ctl.act(8)   # 우 + 서브 2
            tap.assert_called_once_with(ACTION_KEYS["right"], 0.26)
            tap.reset_mock()
            self.ctl.act(9)   # HARVEST
            tap.assert_called_once_with(ACTION_KEYS["harvest"], 0.05)

    def test_same_key_cooldown(self):
        self.ctl._last_key_at[ACTION_KEYS["left"]] = 1e18  # 미래 시간 = 쿨다운 중
        ok, why = self.ctl.act(0)
        self.assertTrue(ok)          # 행동은 성공으로 기록
        # _tap이 쿨다운으로 스킵했는지: dry_run 출력 대신 내부 시간 기록 불변 확인
        self.assertEqual(self.ctl._last_key_at[ACTION_KEYS["left"]], 1e18)

    def test_release_all(self):
        self.ctl._pressed = {"left", "right"}
        released = []
        with mock.patch.object(self.ctl, "_force_release", side_effect=released.append):
            self.ctl.release_all()
            self.assertIn("left", released)
            self.assertIn("right", released)
            self.assertIn(ACTION_KEYS["harvest"], released)


if __name__ == "__main__":
    unittest.main()
