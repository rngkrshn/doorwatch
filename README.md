project:
  name: DoorWatch
  description: >
    DoorWatch is a Raspberry Pi–based computer vision system that detects
    door open/close events using a live camera feed. The system is designed
    to emit exactly one event per open–close cycle and run headlessly on boot.

architecture:
  pipeline:
    - camera_input: USB / Raspberry Pi camera
    - capture: OpenCV live frame capture (YUYV or MJPG)
    - preprocessing: grayscale + Gaussian blur
    - detection: frame differencing within ROI
    - scoring: motion fraction (changed pixels / total)
    - state_machine: debounced OPEN / CLOSED transitions
    - output: console events (notifications planned)

features:
  implemented:
    - live video processing (no stored video)
    - frame-by-frame motion detection
    - debounced state machine (prevents repeated triggers)
    - camera warmup to reduce startup false positives
    - empirical calibration tooling
    - modular Python package structure
  not_yet_implemented:
    - push notifications
    - systemd service (run on boot)
    - logging/metrics
    - adaptive lighting handling

repository_structure:
  root:
    - src:
        - __init__.py: marks Python package
        - main.py: main runtime loop
        - detector.py: motion detection logic
        - state.py: OPEN/CLOSED state machine
    - scripts:
        - camera_test.py: single-frame camera test
        - calibrate_roi.py: motion calibration utility
    - README.md
    - .gitignore

runtime:
  execution:
    command: python -m src.main
    output:
      - periodic motion status
      - OPEN event on door open
      - CLOSED event after door closes and stabilizes

calibration:
  tool: scripts/calibrate_roi.py
  purpose:
    - observe raw motion values
    - determine open/close thresholds
    - validate noise floor
  output: motion_frac values (0.0–1.0)

known_issues:
  - full_frame_roi:
      description: >
        Entire frame is currently used as ROI, making the system sensitive
        to lighting changes, exposure drift, and background motion.
  - thresholds_not_final:
      description: >
        Conservative thresholds are used to avoid missed events, which may
        allow false positives in noisy scenes.
  - camera_noise:
      description: >
        MJPG streams may produce intermittent decoder warnings; YUYV is
        preferred when supported.
  - no_notifications:
      description: Events are printed to stdout only.
  - no_boot_service:
      description: Program does not yet run automatically on startup.

planned_work:
  - tighten_roi_to_door_edge
  - add_push_notifications
  - add_systemd_service
  - improve exposure/lighting robustness
  - add logging and metrics

engineering_focus:
  - reliability_over_flash
  - deterministic behavior
  - debounced real-world event detection
  - constrained-hardware computer vision

status: work_in_progress
