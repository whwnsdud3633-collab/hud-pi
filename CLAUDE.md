# AR HUD — 라즈베리파이 렌더러

빗길 차선 인식 결과를 차량 윈드실드의 반사식 HUD 패널에 표시하는 시스템.
이 저장소는 **파이 쪽(수신 + 렌더링)** 만 담당한다.

```
카메라 → Jetson Orin Nano (추론) → UDP → [ Raspberry Pi 5 ] → Touch Display 2 → 반사 패널 → 윈드실드
                                            이 저장소
```

---

## 실행 환경 (실물 확인됨)

| 항목 | 값 |
|---|---|
| 보드 | Raspberry Pi 5, RAM 8GB, Active Cooler, 27W 전원 |
| OS | Raspberry Pi OS 64-bit, Debian 13 Trixie, 커널 6.18 aarch64 |
| 데스크톱 | **labwc (Wayland)**. X11 아님 |
| 디스플레이 | Raspberry Pi Touch Display 2, DSI 연결, 출력 이름 `DSI-1` |
| 패널 해상도 | **1280×720** (패널 원본은 720×1280 세로, labwc가 `--transform 90`으로 회전) |
| OpenCV | 4.10.0, **apt 설치본** (`python3-opencv`) |
| 사용자 | `visionajou`, 홈 `/home/visionajou`, 코드 `~/hud` |
| 네트워크 | 폰 핫스팟(개발용) `wlan0` + 젯슨 직결 이더넷 `eth0` |
| 젯슨 직결 대역 | `192.168.50.0/24`. 파이 `192.168.50.20`, 젯슨 `192.168.50.10`. NetworkManager 프로파일 이름은 `lane-link` |

### 환경 관련 하드 규칙

- **`pip install opencv-python` 금지.** 소스 빌드로 들어가 한 시간 이상 걸리고 자주 실패한다. apt 패키지를 쓴다.
- **`RPi.GPIO` 사용 금지.** Pi 5는 RP1 I/O 칩을 거치므로 동작하지 않는다. `gpiozero` + `lgpio`를 쓴다.
- **`xset`, `lcd_rotate`, `display_rotate` 사용 금지.** Wayland라 전부 무효다. 화면 설정은 `wlr-randr` 또는 `raspi-config`.
- 화면 회전은 `~/.config/labwc/autostart`에 이미 설정되어 있다. 코드에서 다시 회전하지 말 것.
- 추가 파이썬 패키지가 필요하면 `apt`를 우선하고, 불가피하면 `python3 -m venv --system-site-packages`.

---

## 반사식 HUD 제약 — 렌더링 시 반드시 지킬 것

이 프로젝트가 일반 화면 앱과 다른 지점이다.

1. **배경은 순수 검정(0,0,0)이어야 한다.** 반사 광학계에서 검정 = 투명이다. 어두운 회색 배경이나 그라디언트 배경을 깔면 윈드실드에 회색 사각형이 그대로 떠서 운전자 시야를 가린다.
2. **큰 면적의 밝은 요소를 피한다.** 선과 글리프 위주로 그린다.
3. **좌우 반전이 필요하다.** 패널 → 반사판 → 눈 경로에서 상이 뒤집힌다. `display.flip_horizontal` 설정으로 제어한다.
4. **합성은 알파가 아니라 가산(additive)이다.** 발광 디스플레이라 겹치는 빛이 밝아진다. 실제 광학 동작과도 맞고 더 빠르다.
5. 밝기는 최대로 둔다 (`/sys/class/backlight/`).

---

## 파일 구성

### 존재함

| 파일 | 상태 |
|---|---|
| `hud_theme.py` | 시안 2a/2b 렌더러. 완성도 높음. 12ms/프레임까지 최적화됨 |
| `hud_ui.py` | 화면 구성, 경고 판정, preview/sample/receive/bench 서브커맨드 |

### 없음 — 작성 필요

`hud_theme.py`와 `hud_ui.py`가 import하지만 저장소에 없다. 두 파일이 기대하는 인터페이스에 정확히 맞춰야 한다.

**`hud_align.py`**
```python
ensure_alignment_config(config) -> None      # config에 alignment 기본값 주입
class AlignmentMap:
    def __init__(self, config): ...
    calibrated: bool                          # 실측 보정 여부
    def clip_above_horizon(self, lane): ...    # 지평선 위 구간 제거
    def project(self, lane): ...               # -> (points Nx2 int, valid bool 배열)
```

**`hud_system.py`**
```python
class PacketError(Exception): ...
def _decode_packet(data) -> dict              # UDP 바이트 → 프레임 dict
def _mock_lane(...)                           # 테스트용 가짜 차선
def _select_hud_lanes(lanes)                  # 자차 좌/우 차선 선택
class LaneSmoother: ...                       # 시간축 스무딩
def load_config(path) / save_config(path, cfg)
```

**`hud_proto.py`** — 신규 분리
패킷 스키마만 담는다. `_decode_packet`의 파싱 로직은 여기로 옮기고 `hud_system`은 이를 호출한다.

> **`hud_proto.py`는 젯슨 쪽 담당자와 바이트 단위로 동일해야 한다.**
> 필드를 추가·삭제·재배치하면 젯슨 쪽 코드가 즉시 깨진다.
> 변경이 필요하면 코드를 고치지 말고 사람에게 먼저 알릴 것. 변경 시 `VERSION` 상수를 올리고 양쪽을 동시에 교체한다.

---

## 좌표계

- 디자인 좌표: `960×540` (`hud_theme.DESIGN_W/H`)
- 패널 좌표: `1280×720`
- 둘 다 16:9라 `scale = 1.3333`, `offset_x = offset_y = 0`. **레터박스가 생기지 않는다.**
- 차선 폴리라인은 정규화 좌표로 수신 → `AlignmentMap.project()`로 패널 픽셀 변환

---

## 알려진 문제 / 작업 항목

### 1. 깊이 정보 유실 (우선순위 높음)

`hud_theme._project_lane()`이 불리언 마스크로 점을 걸러내면서 **"이 점이 몇 번째 슬롯이었는지"** 가 사라진다.

```python
points = points[valid]      # 인덱스 소실
points = points[inside]     # 또 소실
```

눈 위치 보정(아래 참조)은 점마다 깊이에 따라 다른 계수를 곱해야 하므로, 깊이 배열을 같이 끌고 다녀야 한다.

**요구사항**: `project()`와 `_project_lane()`이 `(points, depths)`를 함께 반환하고, 모든 필터링·정렬을 두 배열에 동일하게 적용할 것.

### 2. 한글 폰트

`FONT_CANDIDATES["korean"]` 후보가 기본 설치에 없으면 `ImageFont.load_default()`로 떨어져 한글이 네모로 나온다.
설치: `sudo apt install -y fonts-nanum fonts-nanum-extra fonts-noto-cjk`
폰트 탐색 실패 시 조용히 넘어가지 말고 경고를 남길 것.

### 3. `rebuild()` 부작용

`ThemeRenderer.rebuild()`가 `self.__init__(config)`를 호출해 `self.started`가 리셋된다.
설정을 바꿀 때마다 부트 애니메이션이 다시 재생된다.

### 4. 출력 백엔드 미결정

현재 `cv2.imshow` + `waitKey`. Wayland에서는 XWayland를 경유해 지연이 붙는다.

- 대안 A: `raspi-config`로 X11 전환
- 대안 B: pygame KMSDRM (`SDL_VIDEODRIVER=kmsdrm`) — 지연 최소지만 **컴포지터를 거치지 않으므로 패널이 720×1280 세로로 나온다.** 소프트웨어 회전 비용을 감안해야 함

실측 후 결정한다. 어느 쪽이든 `hud_theme`의 그리기 함수는 건드리지 않도록 출력 계층을 분리해 둘 것.

### 5. 안티에일리어싱

반사식은 선이 얇을수록 좋은데 계단 현상이 보인다.
2배 해상도 버퍼에 그린 뒤 `cv2.resize(..., interpolation=cv2.INTER_AREA)`로 축소하는 방식 검토.

---

## 눈 위치 보정 설계

운전자마다 눈 높이가 달라 오버레이가 실제 차선과 어긋난다. 아래 구조로 간다.

```
[로터리 인코더]  →  (트림 dy 또는 가상 eye_y, eye_z)  ─┐
                                                     ├→  hud_align  →  H  →  렌더
[아이트래킹]     →  (실측 eye_y, eye_z)               ─┘
```

`hud_align`은 **"눈 위치 파라미터 → 호모그래피"** 만 담당한다. 파라미터를 누가 채우는지는 관심 없다.
이렇게 하면 수동 조정을 먼저 만들어도 버리는 코드가 없고, 아이트래킹은 소스만 갈아끼우면 된다.

### 채택: 깊이 가중 오프셋 (Level 2)

단순 픽셀 오프셋은 물리적으로 틀리다. 눈이 `e`만큼 움직이면 거리 `d`의 노면 점은 가상상 평면에서 `e × (1 − d_vi/d)`만큼 움직인다. 가상상 거리 `d_vi = 1m`일 때 5m 점은 `0.8e`, 40m 점은 `0.975e`. 근거리와 원거리 이동량이 다르므로 단일 오프셋으로는 한쪽이 어긋난다.

```python
DEPTHS = [5, 8, 11, ..., 40]   # hud_proto의 리샘플 거리와 동일해야 함
D_VI   = 1.0                    # 가상상 거리(m), panel.yaml에서 로드

points[:, 1] += trim_dy * (1.0 - D_VI / depths)
```

**이것이 위 "깊이 정보 유실" 문제를 반드시 먼저 고쳐야 하는 이유다.**

### 향후 (Level 3)

아이박스 4코너에서 한 번 보정 → `(eye_y, eye_z)` 이중선형 보간으로 임의 눈 위치의 H 생성.
**행렬 원소를 보간하지 말 것.** 대응점 4쌍의 목적지 좌표를 보간한 뒤 `cv2.getPerspectiveTransform`으로 H를 다시 계산한다. 수치적으로 훨씬 안정적이다.

---

## 통신

- UDP, 포트 5005, 파이가 수신 전담
- 젯슨은 차선 좌표만 보낸다. 영상은 보내지 않는다
- 슬롯별 시간축 필터를 걸려면 점 개수가 고정이어야 한다. 가변 길이면 오버레이가 떨린다
- 렌더 주기와 수신 주기를 분리한다. 젯슨이 30fps여도 파이는 60fps로 그리고 사이 프레임은 외삽으로 채운다
- 무수신 시 동작(페이드아웃, 심볼 모드 전환)은 UI 코드가 아니라 상태 계층에 둔다

### 젯슨이 실제로 보내는 패킷 (2026-09-12 실측)

`192.168.50.10` 에서 받은 실물이다. **추측하지 말고 이걸 기준으로 한다.**

```json
{"v":1,"seq":292,"ts":1789149939.2586713,
 "img_w":960,"img_h":540,
 "state":0,"conf":1.0,
 "left":[-0.003151,0.505674,423.193818],
 "right":[-0.008059,5.896502,-500.737265],
 "fps":0.0,"id_left":1,"id_right":3,
 "event":"","intent":"","extra":null,
 "y_range":[216.0,410.4]}
```

가장 헷갈리는 지점: **`left`/`right` 는 점 배열이 아니라 2차 다항식 계수 3개다.**

| 필드 | 뜻 |
|---|---|
| `left` / `right` | `[a, b, c]` 이고 `x = a·y² + b·y + c`. 차선을 못 잡으면 `null` |
| `y_range` | `[y_top, y_bottom]`. 위 식이 유효한 y 구간. 차선이 없으면 `null` |
| `img_w` / `img_h` | 위 x, y 가 기준으로 삼는 프레임 크기 (픽셀). 정규화는 이걸로 한다 |
| `state` | 0 정상 / 1 주의 / 2 인식 불가 |
| `conf` | 0~1 신뢰도 |
| `ts` | 송신 시각. **Unix epoch 초(float)** 다 |
| `id_left` / `id_right` | 차선 트랙 id |
| `event` / `intent` | 문자열. 어휘 미확인 (실측 시 둘 다 `""`) |
| `extra` | 미사용 |

계수를 풀면 이런 값이 나온다. 위에서 소실점으로 수렴하고 아래에서 벌어진다:

| | y=216 (원거리) | y=410 (근거리) |
|---|---|---|
| `left` | x=385 | x=100 |
| `right` | x=397 | x=562 |

수신 쪽 어댑터는 `hud_system._adapt_jetson_packet` 이다. 계수를 `y_range`
위에서 12점으로 펴고 `img_w`/`img_h` 로 0~1 정규화해 기존 `lanes` 포맷으로
바꾼다. `lanes` 키가 있는 패킷은 예전 경로로 가므로 mock 은 그대로 돈다.

### 문서가 실물과 어긋나 있던 것 — 아직 안 맞음

아래는 원래 설계 의도인데 젯슨은 아직 그렇게 보내지 않는다.
**파이 코드만 고쳐서 될 일이 아니라 젯슨 쪽과 같이 바꿔야 한다.**

- **점 12개 고정 리샘플**: 설계는 "젯슨이 고정 종방향 거리로 리샘플링해 12점을
  보낸다" 였다. 실제로는 계수를 보내고 파이가 12점으로 편다. 그래서 지금 12점은
  **고정 종방향 거리가 아니라 영상 y 등간격**이다. 눈 위치 보정의 `DEPTHS` 가
  종방향 거리를 전제하므로, 깊이 가중 오프셋을 붙이기 전에 이걸 먼저 맞춰야 한다
- **`t_capture` (ns, `CLOCK_MONOTONIC`)**: 실제로는 `ts` 하나뿐이고 Unix epoch
  초다. 파이의 `CLOCK_MONOTONIC` 과 직접 뺄 수 없다. 지연 보정 외삽을 붙이려면
  이 필드부터 정리해야 한다
- `fps` 는 계속 `0.0` 으로 온다. `inference_ms` 필드는 아예 없다

개발 중에는 젯슨 대신 노트북에서 더미 패킷을 쏜다. 젯슨을 기다리지 않는다.

---

## 코드 스타일

- 주석과 커밋 메시지는 한국어
- 안전 인증 시스템이 아니므로 **"이것은 실제 ADAS가 아닙니다" 류의 면책 문구를 코드나 문서에 넣지 말 것**
- 성능이 중요한 렌더 경로에서는 다음 최적화를 유지할 것:
  - 채널 분리 가산 합성 (3채널 브로드캐스트보다 빠름)
  - `cv2.boundingRect(mask)`로 실제 칠할 영역만 처리
  - 글로우는 축소 → 블러 → 확대
  - 글자 마스크는 `lru_cache`
- 위 최적화를 "가독성을 위해" 되돌리지 말 것. 12ms/프레임의 핵심이다

---

## 작업 방식

- **한 번에 하나씩.** "전부 고쳐줘" 대신 구체적인 항목 하나를 지시받는다
- 변경 후에는 실제로 실행해서 확인한다:
  ```bash
  cd ~/hud && python3 hud_ui.py preview
  ```
- 화면 출력 결과(색, 굵기, 글자 깨짐 등)는 사람이 눈으로 확인해야 한다. 판단이 필요하면 물어볼 것
- 커밋은 의미 단위로 나눈다
