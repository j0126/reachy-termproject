"""Reachy HW4 Motion Lab: design, check, play (Mock / local MuJoCo), experiment (R1-R3) and analyse HW4 motions.

Layers (the GUI only calls the motion engine; it adds no motion logic of its own):

* ``model``       - Qt-free editing state: load / edit / save ``motions/*.yaml``, analysis via planner + safety
                    checker, expressive vs neutral comparison, export bundle. Unit-tested without a display.
* ``plots``       - Qt-free matplotlib figure builders of the Motion tab (also used for PNG export).
* ``playback``    - Qt-free Mock playback controller: a ``MotionService`` on its own worker thread.
* ``experiments`` - Qt-free R1 / R2 / R3 trials and their computed checks (docs/phase4/PHASE4_PLAN.md).
* ``lab``         - Qt-free experiment runner: one worker thread per backend (Mock or local MuJoCo), a queue to the
                    GUI, run files on one time base.
* ``results``     - Qt-free loading, tracking metrics, R3 / expressive-vs-neutral comparisons, exports, summary.
* ``lab_plots``   - Qt-free result and live figures.
* ``lab_tabs``    - PySide6 Experiment and Results tabs.
* ``ko``          - Korean UI labels, the glossary (tooltips, help dialog), refusal sentences and the Korean fonts
                    (Qt and matplotlib); technical terms stay in English.
* ``guide``       - Qt-free button logic of the Experiment tab: which buttons run and why one is disabled.
* ``robot_panel`` / ``robot_session`` - the real-robot box (IP / port, connect, wake-up, torque-off) and its session
                    (one SDK client; each click of the person at the GUI is the approval of that action).
* ``video``       - Qt-free expressive-vs-neutral side-by-side video jobs (offline MuJoCo replay, simulated): job
                    folders, finding a finished video by spec digests, and the render process (``python -m ...``).
* ``video_tab``   - PySide6 Motion-tab page that renders (separate process) and plays that video (QMovie, WebP).
* ``app``         - PySide6 window (``reachy-motion gui``): Motion / Experiment / Results on one screen (no editing,
                    no start screen, no basic / advanced mode since the 2026-11-12 declutter).

Safety rules (CLAUDE.md rule 3, docs/lab/MOTION_LAB.md, docs/lab/GUI_ROBOT.md): only the run buttons call a backend,
and only for plans that pass the initial policy; the backends are our local MuJoCo daemon (127.0.0.1), the real robot
(one click per action by the person at the GUI) and, hidden, the Mock for the tests; the motion files are never written.
"""
