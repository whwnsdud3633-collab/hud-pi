# HUD 화면 구성 사용법

`hud_ui.py` 는 기존 `hud_system.py` 와 같은 폴더에 두고 씁니다. 좌표 수신과 전송은
`hud_system.py` 가 그대로 담당하고, 이 파일은 **화면에 무엇을 어떻게 그릴지**만 맡습니다.

## 1. 모델 없이 화면부터 만들기

```bash
cd ~/HUD_전달파일
python3 hud_system.py init-config
python3 hud_ui.py preview --windowed
```

네트워크도 추론 코드도 필요 없습니다. 가짜 차선이 흐르는 상태로 바로 뜹니다.

미리보기 조작:

| 키 | 동작 |
|---|---|
| `W` | 경고 상태 순환 (없음 → 좌이탈 → 우이탈 → 좌변경 → 우변경) |
| `L` | 차선 2개 ↔ 3개 |
| `D` | 디버그 정보 표시 전환 |
| `H` | 좌우 반전 (반사 필름 때문에 뒤집힐 때) |
| `[` `]` | 차선 굵기 조절 |
| `S` | 현재 값을 `hud_config.json` 에 저장 |
| `Q` | 종료 |

노트북에서 그림만 확인하려면:

```bash
python3 hud_ui.py sample --out samples
```

상황별 PNG 4장이 `samples/` 에 저장됩니다.

## 2. 운전자 시점 정렬

화면 구성이 마음에 들면, 실제 도로와 겹치게 만드는 보정을 해야 합니다.
`HUD_시점정렬_사용법.md` 와 `hud_align.py` 를 참고하세요. 이걸 하지 않으면
차선이 그려지기는 해도 실제 도로와 어긋납니다. `preview` 실행 시 정렬이 안 된
상태면 경고가 출력되고, 디버그 정보에도 `ALIGN UNCALIBRATED` 로 표시됩니다.

## 3. 해상도 맞추기

파이에서 실제 해상도를 확인합니다.

```bash
xrandr | grep '\*'
```

`hud_config.json` 의 `display.width`, `display.height` 를 그 값으로 바꿉니다.
7인치 공식 터치는 보통 `800 x 480`, 일반 HDMI 7인치는 `1024 x 600` 이나 `1280 x 800` 입니다.

## 4. 레이아웃 조정

모든 배치 값은 `hud_config.json` 의 `ui` 항목에 있습니다. 좌표는 전부 `0.0~1.0`
정규화 값이라 해상도를 바꿔도 배치가 그대로 유지됩니다.

```json
"ui": {
  "lane":    { "thickness": 8, "emphasis_thickness": 13 },
  "warning": { "color_bgr": [70,190,255], "side_margin": 0.085,
               "center_y": 0.52, "blink_hz": 2.4, "label": true },
  "safe_area": [0.06, 0.06, 0.94, 0.94],
  "debug":   { "enabled": false, "origin": [0.03, 0.06] },
  "center_tick": { "enabled": false }
}
```

자주 만지게 될 값:

- `warning.side_margin` — 꺾쇠가 화면 가장자리에서 얼마나 떨어질지
- `warning.center_y` — 꺾쇠 높이
- `warning.blink_hz` — 이탈 경고 깜빡임 속도
- `warning.label` — `DEPARTURE` / `LANE CHANGE` 글자 표시 여부
- `debug.enabled` — 실제 주행에서는 `false` 권장. 실행 중 `D` 키로 켤 수 있습니다.
- `center_tick.enabled` — 화면 하단 중앙 기준선. 보정 기준점이 필요하면 켭니다.
- `safe_area` — 경고·글자 같은 화면 고정 요소가 들어갈 영역. 차선은 여기 영향을 받지 않고
  운전자 시점 정렬을 따릅니다.

색은 `BGR` 순서입니다. `[70,190,255]` 가 호박색, `[255,255,255]` 가 흰색입니다.
반사 필름에 따라 호박색이 잘 안 뜨면 경고도 흰색으로 바꾸고 굵기로 구분하세요.

## 5. 실제 수신 실행

```bash
python3 hud_ui.py receive
```

기존 `hud_system.py receive` 대신 이걸 쓰면 됩니다. UDP 포트와 타임아웃은
`hud_config.json` 의 `network` 항목을 그대로 따릅니다. 실행 중 `D` 키로 디버그 정보를
켜고 끌 수 있습니다.

systemd 파일을 이미 만드셨다면 `ExecStart` 의 `hud_system.py` 를 `hud_ui.py` 로만
바꾸면 됩니다.

## 6. 추론 코드에서 경고 보내기

기존 `HudSender` 에는 경고 항목이 없어서, 그 자리에 `HudUiSender` 를 씁니다.

```python
from hud_ui import HudUiSender

hud = HudUiSender("192.168.50.20", 5005)

hud.send(
    lane_polylines_px,
    frame_width=frame.shape[1],
    frame_height=frame.shape[0],
    warning="departure_left",   # none, departure_left, departure_right,
                                # change_left, change_right
    lane_change=False,
    fps=22.4,
    inference_ms=44.6,
)

hud.close()
```

이미 0~1 정규화 좌표라면 `frame_width`, `frame_height` 를 빼고 부르면 됩니다.
차선을 못 찾았으면 빈 배열을 보내세요. `status` 가 `lost` 로 잡혀 화면이 즉시 지워집니다.

경고 판정 자체는 추론 코드 몫입니다. 가장 단순한 방법은 화면 아래쪽 차선 x 좌표가
중앙에서 얼마나 치우쳤는지 보는 것입니다.

```python
LANE_CENTER_NEUTRAL = 0.5   # 아래 방법으로 측정해서 채웁니다
DEPARTURE_MARGIN = 0.10
HOLD_FRAMES = 5

warning = "none"
if len(lanes_normalized) >= 2:
    left_x = lanes_normalized[0][-1][0]
    right_x = lanes_normalized[1][-1][0]
    offset = (left_x + right_x) / 2 - LANE_CENTER_NEUTRAL
    if offset > DEPARTURE_MARGIN:
        warning = "departure_left"    # 차가 왼쪽으로 밀림
    elif offset < -DEPARTURE_MARGIN:
        warning = "departure_right"
```

### 기준값 측정이 먼저입니다

`LANE_CENTER_NEUTRAL` 을 0.5 로 두면 안 됩니다. 카메라가 차량 중심선에 정확히
있지 않으면, 차선 한가운데 있어도 영상에서 계산한 중심은 0.5 가 아닙니다.
룸미러 오른쪽처럼 옆으로 치우쳐 달면 특히 그렇습니다.

차선 한가운데 똑바로 정차한 상태에서 이 값을 몇십 프레임 찍어 평균을 내세요.

```python
samples.append((left_x + right_x) / 2)
print(sum(samples) / len(samples))   # 이 값이 LANE_CENTER_NEUTRAL
```

`DEPARTURE_MARGIN` 은 0.10 에서 시작해 실차에서 조정합니다. 값이 경계에 걸쳐
경고가 떨리면, 같은 조건이 `HOLD_FRAMES` 회 연속될 때만 켜지도록 카운터를 두세요.
끌 때는 더 오래 (예: 10프레임) 버티게 하면 깜빡임이 줄어듭니다.

## 7. 같이 고친 것

기존 `hud_system.py receive` 에는 송신기만 재시작하면 HUD 가 영구히 검은 화면으로
남는 문제가 있었습니다. 수신기가 마지막 `seq` 보다 큰 패킷만 받는데, 송신기가 다시
켜지면 `seq` 가 0 부터 시작하기 때문입니다. `hud_ui.py receive` 는 패킷이 끊긴 것으로
판정되는 순간 이 값을 초기화해서, 추론 코드만 껐다 켜도 바로 복구됩니다.
