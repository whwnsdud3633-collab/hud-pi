#!/bin/bash
cd ~/hud
export DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0
python3 hud_ui.py receive --theme "$@"
