# NBA 2K26 Shot Meter Holder & Auto-Shoot Suite 🏀⚡

An advanced, high-precision automation, computer vision, and closed-loop tuning suite designed for **NBA 2K26** shot timing and rhythm-stick execution on Windows. 

This repository provides both **hardware-level timing interception** (Keyboard & Controller via ViGEmBus) and **computer vision phase-locked meter detection** (OpenCV, Windows OCR, YOLOv4).

---

## 📋 Table of Contents
- [Overview & Architecture](#-overview--architecture)
- [Comprehensive File & Directory Documentation](#-comprehensive-file--directory-documentation)
  - [Core Root Scripts](#1-core-root-scripts)
  - [Root Configurations & Launchers](#2-root-configurations--launchers)
  - [Vision & Phase-Locked Suite (`k_meter/`)](#3-vision--phase-locked-suite-k_meter)
  - [Output & Data Analytics Folders](#4-output--data-analytics-folders)
- [System Requirements & Prerequisites](#-system-requirements--prerequisites)
- [Module Usage Guides](#-module-usage-guides)
  - [1. Controller Rhythm-Stick Timing (`precision_timer_pad.py`)](#1-controller-rhythm-stick-timing-precision_timer_padpy)
  - [2. Live Closed-Loop Feedback Tuning (`live_coach.py`)](#2-live-closed-loop-feedback-tuning-live_coachpy)
  - [3. Keyboard Precision Timer (`precision_timer.py`)](#3-keyboard-precision-timer-precision_timerpy)
  - [4. K2 Phase-Locked Render Clock Engine (`k2.py`)](#4-k2-phase-locked-render-clock-engine-k2py)
  - [5. Vision Shot Meter Engine (`k_meter.py`)](#5-vision-shot-meter-engine-k_meterpy)
  - [6. Gameplay Video Analysis (`tempo_learn.py` & `meter_watch.py`)](#6-gameplay-video-analysis-tempo_learnpy--meter_watchpy)
- [Configuration Reference](#-configuration-reference)
- [License & Disclaimer](#-license--disclaimer)

---

## 🏀 Overview & Architecture

Shot timing in NBA 2K26 requires sub-millisecond precision. Server tick rates, jump shot animations, stamina depletion, and render frame pacing all affect the release window. This repository provides a multi-tiered approach:

1. **Fixed Precision Engine**: Suppresses physical key/button releases and injects synthetic releases using Windows high-resolution waitable timers (`CreateWaitableTimerExW`) and busy-spin sub-ms stabilization.
2. **Virtual Controller Injection**: Emulates an Xbox 360 controller via `vgamepad` and `ViGEmBus`, executing precise stick-down gather and upward flick release sequence while mirroring all other inputs.
3. **Closed-Loop Realtime Tuning**: Uses Windows OCR (`winrt`) to read game feedback banners (*"Rushed"*, *"Slightly Rushed"*, *"Great Tempo"*, *"Slightly Late"*) and automatically adjusts hold parameters dynamically on the fly.
4. **Phase-Locked Render Clock Fitting**: Models the 2K26 shot meter as a staircase function of rendered frames, performing least-squares linear fits to predict exact frame crossing times and overcome video capture jitter.

```
                    ┌─────────────────────────────────────────┐
                    │               NBA 2K26                  │
                    └────┬───────────────────────────────▲────┘
                         │ Game Window Display           │ Synthetic Release
                         ▼                               │ (ViGEm / Win32)
┌───────────────────────────────────────────────┐ ┌──────┴────────────────────────┐
│             Computer Vision & OCR             │ │       Precision Injection     │
│ ┌───────────────────┐   ┌───────────────────┐ │ │ ┌───────────────────────────┐ │
│ │  WinRT Banner OCR │   │  Meter HSV Tracker│ │ │ │ precision_timer_pad.py    │ │
│ └─────────┬─────────┘   └─────────┬─────────┘ │ │ └─────────────▲─────────────┘ │
└───────────┼───────────────────────┼───────────┘ └──────────────┼───────────────┘
            │ Feedback Banner       │ Meter Step                 │ Hot-reload
            ▼                       ▼                            │ hold_ms
┌───────────────────────┐ ┌─────────────────────┐   ┌────────────┴──────────────┐
│     live_coach.py     │ │     k2.py (Engine)  │───► precision_timer_pad.json │
│ Closed-Loop Auto-Nudge│ │ Phase-Locked Clock  │   └───────────────────────────┘
└───────────────────────┘ └─────────────────────┘
```

---

## 📁 Comprehensive File & Directory Documentation

### 1. Core Root Scripts

* [`live_coach.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/live_coach.py)
  * **Description**: Real-time closed-loop tempo tuning assistant for NBA 2K26.
  * **Mechanism**: Captures the game window several times a second using `mss`, uses Windows OCR (`winrt`) to detect shot feedback text (*"Rushed"*, *"Great Tempo"*, *"Slow"*), and checks for green release paint splash particles using HSV thresholding.
  * **Behavior**: Automatically nudges `hold_ms` inside `precision_timer_pad.json`. The running pad timer dynamically hot-reloads the file, continuously zeroing in on *"Great"* tempo.

* [`precision_timer_pad.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/precision_timer_pad.py)
  * **Description**: Controller-driven rhythm-stick timing daemon (v1.0).
  * **Mechanism**: Polls a physical Xbox/XInput controller at 1 kHz. When the trigger button (default: `X` / Button 3) is tapped, the script suppresses the raw trigger output to the game, instantly snaps the virtual right stick DOWN, holds for `hold_ms`, flicks UP for `flick_ms`, and returns to neutral.
  * **Features**: Mirroring of all other physical inputs, HidHide integration support, stamina fatigue mode toggle (`F2`), statistics tracking, and press logging to `pad_log.txt`.

* [`precision_timer.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/precision_timer.py)
  * **Description**: Keyboard button-hold timing tool (v3.1).
  * **Mechanism**: Uses Low-Level Keyboard Hooks (`WH_KEYBOARD_LL`) via `keyboard`. Tapping the action key (default `K`) passes the press through immediately, suppresses physical release, and schedules a high-precision synthetic release via a persistent `TIME_CRITICAL` worker thread.
  * **Features**: Sub-millisecond busy-spin stabilization (`spin_margin_ms`), hotkey adjustments (`F5`-`F10`), and crash handler logging.

* [`meter_watch.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/meter_watch.py)
  * **Description**: Full gameplay video meter tracker and offline shot analytics engine.
  * **Mechanism**: Scans every frame of an OBS gameplay recording without sampling tricks. Locates the dynamic rhythm-shot meter (green target window + white/gray arc stroke) regardless of player movement or camera scaling.
  * **Outputs**: Extracts shot events (`t_start`, `t_full`, `splash`, `t_end`), matches them with `pad_log.txt` timestamps, and generates per-hold outcome reports in `meter_watch_out/`.

* [`tempo_learn.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/tempo_learn.py)
  * **Description**: Post-game video analysis and tempo recommendation tool.
  * **Mechanism**: Performs Windows OCR across OBS recordings to find shot feedback banners. Groups OCR hits into distinct shot events and calculates recommended `hold_ms` adjustments.
  * **Outputs**: Generates CSV shot logs and summary reports in `tempo_learn_out/`.

---

### 2. Root Configurations & Launchers

* [`precision_timer_pad.json`](file:///D:/Users/rodnee/Desktop/Dev/zxc/precision_timer_pad.json)
  * Configuration file for `precision_timer_pad.py` and `live_coach.py`. Controls trigger mapping, `hold_ms`, `flick_ms`, `fatigue_offset_ms`, `poll_hz`, and `spin_margin_ms`.
* [`precision_timer.json`](file:///D:/Users/rodnee/Desktop/Dev/zxc/precision_timer.json)
  * Configuration file for `precision_timer.py`. Controls `action_key`, `hold_ms`, `release_offset_ms`, and `spin_margin_ms`.
* [`run_pad.bat`](file:///D:/Users/rodnee/Desktop/Dev/zxc/run_pad.bat)
  * Windows batch launcher for starting `precision_timer_pad.py`.
* [`run_coach.bat`](file:///D:/Users/rodnee/Desktop/Dev/zxc/run_coach.bat)
  * Windows batch launcher for starting `live_coach.py`.
* [`precision_timer.bat`](file:///D:/Users/rodnee/Desktop/Dev/zxc/precision_timer.bat)
  * Windows batch launcher for starting `precision_timer.py`.

---

### 3. Vision & Phase-Locked Suite (`k_meter/`)

> [!IMPORTANT]
> All vision algorithms, HSV thresholds, ROI search boxes, and geometric target ratios (`arrow_ratio`) inside the `k_meter/` directory are specifically calibrated and designed for the **"Arrow 2"** shot meter style in NBA 2K26 (horizontal purple fill bar racing into the green tick window at the bar's tip).

* [`k_meter/k2.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/k2.py)
  * **Description**: Next-generation render-clock phase-locked shot release engine for the **Arrow 2** shot meter style.
  * **Theory**: NBA 2K26 updates the shot meter once per rendered frame, creating a discrete staircase fill rather than a continuous line. `k2.py` fits the step transitions `t_k = t0 + k * T` using least-squares regression, estimates the pixel step size, and calculates the exact crossing frame `K = ceil(target / step_px)`.
  * **Features**: Includes built-in benchmark self-tests (`--selftest`) and offline video frame evaluation (`--dump`).

* [`k_meter/k2_runtime.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/k2_runtime.py)
  * **Description**: Live execution engine for `k2.py`.
  * **Mechanism**: Monitors game window focus, manages low-latency screen capture via `mss`, continuously feeds frame observations into `StaircaseClock`, and executes scancode key releases via Win32 `SendInput`.

* [`k_meter/k_meter.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/k_meter.py)
  * **Description**: Vision-assisted keyboard auto-shot release tool for **Arrow 2** shot meter (v1.2).
  * **Mechanism**: Tracks the horizontal purple meter fill racing into the green tick at the bar's tip using OpenCV HSV color space bounds. Implements velocity-based latency compensation and safe fallback timers.
  * **Controls**: Operates via `Ctrl+Alt` hotkey combinations to avoid accidental keypresses.

* [`k_meter/vision_probe.py`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/vision_probe.py)
  * **Description**: Offline player detection and jersey color probe.
  * **Mechanism**: Uses OpenCV `cv2.dnn` to run a COCO-pretrained YOLOv4 ONNX model (without requiring PyTorch or ONNX Runtime). Evaluates player detection accuracy and jersey color classification on court.

* [`k_meter/k2.json`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/k2.json)
  * Configuration file for `k2.py` and `k2_runtime.py`. Controls `latency_ms`, `arrow_ratio` (calibrated for Arrow 2 green tip geometry), `hold_ms`, `min_steps`, `spin_ms`, and `no_meter_ms`.
* [`k_meter/k_meter.json`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/k_meter.json)
  * Configuration file for `k_meter.py`. Controls Arrow 2 HSV color bounds, ROI search boxes, lead times, and action key definitions.
* [`k_meter/run_k2.bat`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/run_k2.bat)
  * Windows batch launcher for starting `k2_runtime.py`.
* [`k_meter/run_k_meter.bat`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/run_k_meter.bat)
  * Windows batch launcher for starting `k_meter.py`.
* [`k_meter/badges/`](file:///D:/Users/rodnee/Desktop/Dev/zxc/k_meter/badges)
  * Reference image assets (`early.png`, `excellent.png`, `slightly_early.png`, `slightly_late.png`) used for UI badge template matching and verification.

---

### 4. Output & Data Analytics Folders

* [`tempo_learn_out/`](file:///D:/Users/rodnee/Desktop/Dev/zxc/tempo_learn_out)
  * Directory containing post-game analytics output files (`.report.txt` and `.shots.csv`), generated by `tempo_learn.py`.
* [`meter_watch_out/`](file:///D:/Users/rodnee/Desktop/Dev/zxc/meter_watch_out)
  * Output folder populated by `meter_watch.py` with frame-by-frame shot analysis CSVs and accuracy reports.

---

## 🛠 System Requirements & Prerequisites

### Platform & OS
* **Operating System**: Windows 10 or Windows 11 (64-bit).
* **Execution Privileges**: Administrator privileges required (for Win32 low-level hooks, window capture, and process injection).

### Hardware & Controllers
* **Gamepad**: Xbox 360 / Xbox One compatible controller.
* **Drivers**:
  * [ViGEmBus Driver](https://github.com/nefarius/ViGEmBus/releases) *(Required for virtual controller creation)*.
  * [HidHide](https://github.com/nefarius/HidHide/releases) *(Required to hide physical controller from NBA 2K26 so the game only responds to the virtual pad)*.

### Python Dependencies
Install required packages using `pip`:

```bash
pip install opencv-python numpy mss keyboard vgamepad winrt-Windows.Graphics.Imaging winrt-Windows.Media.Ocr winrt-Windows.Security.Cryptography
```

---

## 🚀 Module Usage Guides

### 1. Controller Rhythm-Stick Timing (`precision_timer_pad.py`)

1. Ensure **ViGEmBus** is installed.
2. Hide your physical controller in **HidHide** and add `python.exe` to HidHide's application allowlist.
3. Run as Administrator:
   ```cmd
   run_pad.bat
   ```
4. **Hotkeys**:
   * `F5`: Pause / Resume timing interception.
   * `F6` / `F7`: Decrease / Increase `hold_ms` by step (default 5 ms).
   * `F3` / `F4`: Fine tune `hold_ms` by ±1 ms.
   * `F2`: Toggle **Fatigue Mode** (adds `fatigue_offset_ms` when player stamina is depleted).
   * `F8`: Output timing status & statistics to log.
   * `F9` / `Ctrl+C`: Exit program.

---

### 2. Live Closed-Loop Feedback Tuning (`live_coach.py`)

Run `live_coach.py` simultaneously with `precision_timer_pad.py` while playing NBA 2K26 in windowed or borderless mode:

```cmd
run_coach.bat
```

* **Modes**:
  * Closed-Loop Auto-Adjustment: `python live_coach.py`
  * Observation Only (No JSON edits): `python live_coach.py --observe`
  * Custom Window Title: `python live_coach.py --window "NBA 2K26"`

---

### 3. Keyboard Precision Timer (`precision_timer.py`)

For keyboard players who want fixed millisecond button hold precision:

```cmd
precision_timer.bat
```
Tap `K` (or your configured action key). The tool holds the key for exactly `hold_ms` and fires a high-precision synthetic release.

---

### 4. K2 Phase-Locked Render Clock Engine (`k2.py`)

To use the phase-locked render-clock meter detection engine:

```cmd
cd k_meter
run_k2.bat
```

* **Run Self-Test Benchmark**:
  ```cmd
  python k2.py --selftest
  ```
* **Offline Video Frame Dump**:
  ```cmd
  python k2.py "C:\Path\To\Recording.mp4" --dump "C:\Path\To\Output"
  ```

---

### 5. Vision Shot Meter Engine (`k_meter.py`)

Vision tracking for the horizontal purple shot meter:

```cmd
cd k_meter
run_k_meter.bat
```
* Hotkeys operate with `Ctrl+Alt` prefix (e.g., `Ctrl+Alt+F6` / `Ctrl+Alt+F7` adjust latency).

---

### 6. Gameplay Video Analysis (`tempo_learn.py` & `meter_watch.py`)

Analyze OBS recordings to evaluate shot timing consistency and gather statistics:

* **Tempo Learn**:
  ```cmd
  python tempo_learn.py "D:\Recordings\2026-07-18 07-21-29.mp4"
  ```
* **Meter Watch**:
  ```cmd
  python meter_watch.py "D:\Recordings\2026-07-18 07-21-29.mp4"
  ```

---

## ⚙ Configuration Reference

### `precision_timer_pad.json`
```json
{
  "trigger_button": 3,
  "hold_ms": 555,
  "release_offset_ms": 0,
  "tune_step_ms": 5,
  "spin_margin_ms": 3.0,
  "flick_ms": 80,
  "fatigue_offset_ms": 8,
  "poll_hz": 1000,
  "pad_index": -1,
  "debug": true
}
```

### `k2.json`
```json
{
  "latency_ms": 66.0,
  "arrow_ratio": 1.022,
  "hold_ms": 650,
  "no_meter_ms": 550,
  "max_hold_ms": 1200,
  "min_steps": 6,
  "spin_ms": 3.0,
  "debug": true
}
```

---

## ⚖ License & Disclaimer

This project is created strictly for educational, research, and offline timing analysis purposes. Users are responsible for complying with game terms of service when using input automation utilities.
