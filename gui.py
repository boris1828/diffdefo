#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DiffXPBD GUI -- configure and run experiments."""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import threading
import queue
import re
from pathlib import Path

PROJ_ROOT = Path(__file__).parent
CONFS_DIR = PROJ_ROOT / "confs"
DEFAULT_CONF = PROJ_ROOT / "src" / "param.conf"
CPP_EXE = str(PROJ_ROOT / "build" / "bin" / "Release" / "xpbd.exe")
TEMP_CONF = str(CONFS_DIR / "_temp_run.conf")

_DONE = object()


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_vec3(vx, vy, vz):
    return f"({vx.get()}, {vy.get()}, {vz.get()})"


def _parse_vec3(s):
    s = s.strip().strip("()")
    parts = [p.strip() for p in s.split(",")]
    return (parts + ["0.0", "0.0", "0.0"])[:3]


def _read_conf(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            idx = line.find("#")
            if idx >= 0:
                line = line[:idx]
            if "=" in line:
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    return cfg


# ── App ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DiffXPBD Launcher")
        self.geometry("1300x860")
        self.minsize(960, 600)
        self._proc = None
        self._q: queue.Queue = queue.Queue()
        self._after_id = None
        self._build_ui()
        CONFS_DIR.mkdir(exist_ok=True)
        if DEFAULT_CONF.exists():
            self._load_conf(str(DEFAULT_CONF))

    # ── root layout ───────────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=410)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── left scrollable panel ─────────────────────────────────────────────────

    def _build_left(self):
        outer = ttk.Frame(self, relief="groove", borderwidth=1)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas, padding=(6, 6, 6, 6))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"))

        inner.columnconfigure(0, weight=1)
        for i, fn in enumerate([
            self._build_simulation, self._build_physics, self._build_positions,
            self._build_object, self._build_collision, self._build_experiment,
            self._build_loss, self._build_optimizer, self._build_output,
        ]):
            fn(inner, i)

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right(self):
        right = ttk.Frame(self, padding=(8, 8, 8, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        # run buttons
        bb = ttk.Frame(right)
        bb.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._kill_btn = None
        for col, (lbl, cmd, w) in enumerate([
            ("Exec C++",  self._exec_cpp,     12),
            ("Exec JAX",  self._exec_jax,     12),
            ("Compare",   self._exec_compare, 12),
            ("Kill",      self._kill,          8),
            ("Clear",     self._clear,         8),
        ]):
            btn = ttk.Button(bb, text=lbl, command=cmd, width=w)
            btn.grid(row=0, column=col, padx=3)
            if lbl == "Kill":
                self._kill_btn = btn
                btn.configure(state="disabled")

        # file bar
        fb = ttk.Frame(right)
        fb.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        fb.columnconfigure(1, weight=1)
        ttk.Label(fb, text="path:").grid(row=0, column=0, padx=(0, 4))
        self.v_save_path = tk.StringVar(value=str(CONFS_DIR / "untitled.conf"))
        ttk.Entry(fb, textvariable=self.v_save_path).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(fb, text="Save", command=self._save,        width=7).grid(row=0, column=2, padx=2)
        ttk.Button(fb, text="Load", command=self._load_dialog, width=7).grid(row=0, column=3, padx=2)

        # terminal
        self._terminal = scrolledtext.ScrolledText(
            right, state="disabled",
            font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", wrap="none",
        )
        self._terminal.grid(row=2, column=0, sticky="nsew")

        # status
        self._status_var = tk.StringVar(value="Ready")
        self._status_lbl = ttk.Label(right, textvariable=self._status_var, anchor="w")
        self._status_lbl.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self._status_lbl.configure(foreground="gray")

    # ── widget helpers ────────────────────────────────────────────────────────

    def _section(self, parent, row, title):
        f = ttk.LabelFrame(parent, text=title, padding=(8, 4, 8, 6))
        f.grid(row=row, column=0, sticky="ew", pady=4)
        f.columnconfigure(1, weight=1)
        return f

    def _row_int(self, f, row, label, default, from_=0, to=100000):
        ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=1, padx=(0, 8))
        v = tk.IntVar(value=default)
        ttk.Spinbox(f, from_=from_, to=to, textvariable=v, width=10
                    ).grid(row=row, column=1, sticky="w")
        return v

    def _row_float(self, f, row, label, default):
        ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=1, padx=(0, 8))
        v = tk.StringVar(value=str(default))
        ttk.Entry(f, textvariable=v, width=14).grid(row=row, column=1, sticky="w")
        return v

    def _row_vec3(self, f, row, label, defaults):
        ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=1, padx=(0, 8))
        sub = ttk.Frame(f)
        sub.grid(row=row, column=1, sticky="w")
        vars_ = []
        for i, (ax, val) in enumerate(zip("xyz", defaults)):
            ttk.Label(sub, text=ax).grid(row=0, column=i * 2, padx=(6, 2))
            v = tk.StringVar(value=str(val))
            ttk.Entry(sub, textvariable=v, width=8).grid(row=0, column=i * 2 + 1)
            vars_.append(v)
        return vars_

    def _subframe(self, parent, row=1):
        """Utility sub-frame that fills both columns; not gridded yet."""
        f = ttk.Frame(parent)
        f.columnconfigure(1, weight=1)
        return f

    def _show(self, f, row=1):
        f.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _hide(self, f):
        f.grid_remove()

    # ── Simulation ────────────────────────────────────────────────────────────

    def _build_simulation(self, p, r):
        f = self._section(p, r, "Simulation")
        self.v_sim_rate  = self._row_int(f, 0, "sim_rate (Hz)", 144, from_=1)
        self.v_n_seconds = self._row_int(f, 1, "n_seconds",      10, from_=1)
        self.v_fps       = self._row_int(f, 2, "fps",            24, from_=1)
        self.v_adj_win   = self._row_int(f, 3, "adjoint_jacobian_window", 1, from_=1)

    # ── Physics ───────────────────────────────────────────────────────────────

    def _build_physics(self, p, r):
        f = self._section(p, r, "Physics")
        self.v_gravity           = self._row_vec3(f, 0, "gravity",           [0.0, -9.81, 0.0])
        self.v_compliance        = self._row_float(f, 1, "compliance",        0.001)
        self.v_target_compliance = self._row_float(f, 2, "target_compliance", 0.005)

    # ── Initial Positions ─────────────────────────────────────────────────────

    def _build_positions(self, p, r):
        f = self._section(p, r, "Initial Positions")
        self.v_offset        = self._row_vec3(f, 0, "offset",        [0.0, 0.0, 0.0])
        self.v_target_offset = self._row_vec3(f, 1, "target_offset", [0.0, 0.0, 0.0])

    # ── Object ────────────────────────────────────────────────────────────────

    def _build_object(self, p, r):
        f = self._section(p, r, "Object")

        ttk.Label(f, text="type").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_obj_type = tk.StringVar(value="cloth")
        cb = ttk.Combobox(f, textvariable=self.v_obj_type,
                          values=["cloth", "chain"], state="readonly", width=10)
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", self._on_obj_type)

        # cloth sub-frame
        self._obj_cloth_f = self._subframe(f)
        self.v_cloth_w   = self._row_int(self._obj_cloth_f, 0, "width",    8, from_=1)
        self.v_cloth_h   = self._row_int(self._obj_cloth_f, 1, "height",   8, from_=1)
        ttk.Label(self._obj_cloth_f, text="pin_mode").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_cloth_pin = tk.StringVar(value="none")
        ttk.Combobox(self._obj_cloth_f, textvariable=self.v_cloth_pin,
                     values=["none", "corners", "row"], state="readonly", width=10
                     ).grid(row=2, column=1, sticky="w")
        ttk.Label(self._obj_cloth_f, text="constraints").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=1)
        cbox_row = ttk.Frame(self._obj_cloth_f)
        cbox_row.grid(row=3, column=1, sticky="w")
        self.v_cloth_stretch = tk.BooleanVar(value=True)
        self.v_cloth_shear   = tk.BooleanVar(value=True)
        self.v_cloth_bending = tk.BooleanVar(value=True)
        ttk.Checkbutton(cbox_row, text="stretch", variable=self.v_cloth_stretch,
                        state="disabled").pack(side="left", padx=2)
        ttk.Checkbutton(cbox_row, text="shear",   variable=self.v_cloth_shear
                        ).pack(side="left", padx=2)
        ttk.Checkbutton(cbox_row, text="bending", variable=self.v_cloth_bending
                        ).pack(side="left", padx=2)

        # chain sub-frame
        self._obj_chain_f = self._subframe(f)
        self.v_chain_len = self._row_int(self._obj_chain_f, 0, "length", 8, from_=1)
        ttk.Label(self._obj_chain_f, text="pin_mode").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_chain_pin = tk.StringVar(value="corners")
        ttk.Combobox(self._obj_chain_f, textvariable=self.v_chain_pin,
                     values=["none", "corners", "row"], state="readonly", width=10
                     ).grid(row=1, column=1, sticky="w")

        self._show(self._obj_cloth_f)

    def _on_obj_type(self, _e=None):
        if self.v_obj_type.get() == "cloth":
            self._hide(self._obj_chain_f)
            self._show(self._obj_cloth_f)
        else:
            self._hide(self._obj_cloth_f)
            self._show(self._obj_chain_f)

    # ── Collision ─────────────────────────────────────────────────────────────

    def _build_collision(self, p, r):
        f = self._section(p, r, "Collision")

        ttk.Label(f, text="collision_mode").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_coll_mode = tk.StringVar(value="constraints")
        cm = ttk.Combobox(f, textvariable=self.v_coll_mode,
                          values=["projection", "constraints"], state="readonly", width=14)
        cm.grid(row=0, column=1, sticky="w")
        cm.bind("<<ComboboxSelected>>", self._on_coll_mode)

        self._coll_compl_f = self._subframe(f)
        self.v_coll_compl = self._row_float(self._coll_compl_f, 0, "compliance", 0.001)
        self._show(self._coll_compl_f)

        ttk.Separator(f, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=6)

        ttk.Label(f, text="collider").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_collider = tk.StringVar(value="sphere")
        cc = ttk.Combobox(f, textvariable=self.v_collider,
                          values=["none", "sphere", "halfspace"], state="readonly", width=10)
        cc.grid(row=3, column=1, sticky="w")
        cc.bind("<<ComboboxSelected>>", self._on_collider)

        self._coll_sphere_f = ttk.Frame(f)
        self._coll_sphere_f.columnconfigure(1, weight=1)
        self.v_sphere_center = self._row_vec3(self._coll_sphere_f, 0, "center", [5.0, -5.0, 5.0])
        self.v_sphere_radius = self._row_float(self._coll_sphere_f, 1, "radius", 3.0)

        self._coll_hs_f = ttk.Frame(f)
        self._coll_hs_f.columnconfigure(1, weight=1)
        self.v_hs_origin = self._row_vec3(self._coll_hs_f, 0, "origin", [0.0, 0.0, 0.0])
        self.v_hs_normal = self._row_vec3(self._coll_hs_f, 1, "normal", [0.0, 1.0, 0.0])

        self._coll_sphere_f.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _on_coll_mode(self, _e=None):
        if self.v_coll_mode.get() == "constraints":
            self._show(self._coll_compl_f)
        else:
            self._hide(self._coll_compl_f)

    def _on_collider(self, _e=None):
        self._coll_sphere_f.grid_remove()
        self._coll_hs_f.grid_remove()
        t = self.v_collider.get()
        if t == "sphere":
            self._coll_sphere_f.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        elif t == "halfspace":
            self._coll_hs_f.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    # ── Experiment ────────────────────────────────────────────────────────────

    def _build_experiment(self, p, r):
        f = self._section(p, r, "Experiment")

        ttk.Label(f, text="experiment").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_experiment = tk.StringVar(value="compliance_gradient")
        cb = ttk.Combobox(f, textvariable=self.v_experiment,
                          values=["forward_simulation", "single_step_jacobian",
                                  "compliance_gradient", "x0_gradient",
                                  "compliance_optimization", "loss_scan_compliance"],
                          state="readonly", width=26)
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", self._on_experiment)

        self._exp_step_f = self._subframe(f)
        self.v_exp_step = self._row_int(self._exp_step_f, 0, "step", 1, from_=1)

        self._exp_iters_f = self._subframe(f)
        self.v_exp_iters = self._row_int(self._exp_iters_f, 0, "iters", 200, from_=1)

        self._exp_scan_f = self._subframe(f)
        self.v_exp_scan_min   = self._row_float(self._exp_scan_f, 0, "min_compliance", 0.0001)
        self.v_exp_scan_max   = self._row_float(self._exp_scan_f, 1, "max_compliance", 0.01)
        self.v_exp_scan_steps = self._row_int(self._exp_scan_f,   2, "sub_steps",      20, from_=1)

        self._exp_subframes = [self._exp_step_f, self._exp_iters_f, self._exp_scan_f]

    def _on_experiment(self, _e=None):
        for sf in self._exp_subframes:
            sf.grid_remove()
        mapping = {
            "single_step_jacobian":    self._exp_step_f,
            "compliance_optimization": self._exp_iters_f,
            "loss_scan_compliance":    self._exp_scan_f,
        }
        sf = mapping.get(self.v_experiment.get())
        if sf:
            self._show(sf)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _build_loss(self, p, r):
        f = self._section(p, r, "Loss")

        ttk.Label(f, text="loss").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_loss = tk.StringVar(value="mse_frames_trajectory")
        cb = ttk.Combobox(f, textvariable=self.v_loss,
                          values=["mse_final_position", "mse_full_trajectory",
                                  "mse_frames_trajectory"],
                          state="readonly", width=24)
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", self._on_loss)

        self._loss_fps_f = self._subframe(f)
        self.v_loss_fps = self._row_int(self._loss_fps_f, 0, "fps", 24, from_=1)
        self._show(self._loss_fps_f)

    def _on_loss(self, _e=None):
        if self.v_loss.get() == "mse_frames_trajectory":
            self._show(self._loss_fps_f)
        else:
            self._hide(self._loss_fps_f)

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _build_optimizer(self, p, r):
        f = self._section(p, r, "Optimizer")

        self.v_opt_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="enable optimizer", variable=self.v_opt_enabled,
                        command=self._on_opt_enabled
                        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self._opt_body = ttk.Frame(f)
        self._opt_body.columnconfigure(1, weight=1)
        self._opt_body.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        ttk.Label(self._opt_body, text="optimizer").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=1)
        self.v_opt_type = tk.StringVar(value="momentum")
        cb = ttk.Combobox(self._opt_body, textvariable=self.v_opt_type,
                          values=["GD", "momentum", "ADAM"], state="readonly", width=12)
        cb.grid(row=0, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", self._on_opt_type)

        self._opt_gd_f = self._subframe(self._opt_body)
        self.v_gd_lr = self._row_float(self._opt_gd_f, 0, "lr", "1e-8")

        self._opt_mom_f = self._subframe(self._opt_body)
        self.v_mom_lr   = self._row_float(self._opt_mom_f, 0, "lr",   "1e-8")
        self.v_mom_beta = self._row_float(self._opt_mom_f, 1, "beta", "0.75")

        self._opt_adam_f = self._subframe(self._opt_body)
        self.v_adam_lr      = self._row_float(self._opt_adam_f, 0, "lr",      "1e-8")
        self.v_adam_beta1   = self._row_float(self._opt_adam_f, 1, "beta1",   "0.9")
        self.v_adam_beta2   = self._row_float(self._opt_adam_f, 2, "beta2",   "0.999")
        self.v_adam_epsilon = self._row_float(self._opt_adam_f, 3, "epsilon", "1e-8")

        self._opt_subframes = [self._opt_gd_f, self._opt_mom_f, self._opt_adam_f]
        self._show(self._opt_mom_f)

    def _on_opt_enabled(self):
        if self.v_opt_enabled.get():
            self._opt_body.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        else:
            self._opt_body.grid_remove()

    def _on_opt_type(self, _e=None):
        for sf in self._opt_subframes:
            sf.grid_remove()
        mapping = {
            "GD":       self._opt_gd_f,
            "momentum": self._opt_mom_f,
            "ADAM":     self._opt_adam_f,
        }
        sf = mapping.get(self.v_opt_type.get())
        if sf:
            self._show(sf)

    # ── Output ────────────────────────────────────────────────────────────────

    def _build_output(self, p, r):
        f = self._section(p, r, "Output")
        self.v_export_obj = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="export_obj (write .obj frames to animation/)",
                        variable=self.v_export_obj
                        ).grid(row=0, column=0, columnspan=2, sticky="w")

    # ── conf generation ───────────────────────────────────────────────────────

    def _build_conf_str(self):
        lines = []

        def add(k, v):
            lines.append(f"{k} = {v}")

        add("sim_rate",  self.v_sim_rate.get())
        add("n_seconds", self.v_n_seconds.get())
        add("gravity",   _fmt_vec3(*self.v_gravity))
        add("target_compliance", self.v_target_compliance.get())
        add("compliance",        self.v_compliance.get())
        add("target_offset", _fmt_vec3(*self.v_target_offset))
        add("offset",        _fmt_vec3(*self.v_offset))

        if self.v_coll_mode.get() == "constraints":
            add("collision_mode", f"constraints({self.v_coll_compl.get()})")
        else:
            add("collision_mode", "projection")

        ct = self.v_collider.get()
        if ct == "sphere":
            c = _fmt_vec3(*self.v_sphere_center)
            add("colliders", f"[ sphere({c}, {self.v_sphere_radius.get()}) ]")
        elif ct == "halfspace":
            o = _fmt_vec3(*self.v_hs_origin)
            n = _fmt_vec3(*self.v_hs_normal)
            add("colliders", f"[ halfspace({o}, {n}) ]")
        else:
            add("colliders", "[]")

        add("export_obj", "true" if self.v_export_obj.get() else "false")
        add("fps", self.v_fps.get())

        ot = self.v_obj_type.get()
        if ot == "cloth":
            flags = ["stretch"]
            if self.v_cloth_shear.get():   flags.append("shear")
            if self.v_cloth_bending.get(): flags.append("bending")
            add("obj", f"cloth({self.v_cloth_w.get()}, {self.v_cloth_h.get()}, "
                       f"{self.v_cloth_pin.get()}, {' | '.join(flags)})")
        else:
            add("obj", f"chain({self.v_chain_len.get()}, {self.v_chain_pin.get()})")

        exp = self.v_experiment.get()
        if exp == "single_step_jacobian":
            add("experiment", f"single_step_jacobian({self.v_exp_step.get()})")
        elif exp == "compliance_optimization":
            add("experiment", f"compliance_optimization({self.v_exp_iters.get()})")
        elif exp == "loss_scan_compliance":
            add("experiment", f"loss_scan_compliance({self.v_exp_scan_min.get()}, "
                              f"{self.v_exp_scan_max.get()}, {self.v_exp_scan_steps.get()})")
        else:
            add("experiment", exp)

        if self.v_opt_enabled.get():
            ot2 = self.v_opt_type.get()
            if ot2 == "GD":
                add("optimizer", f"GD({self.v_gd_lr.get()})")
            elif ot2 == "momentum":
                add("optimizer", f"momentum({self.v_mom_lr.get()}, {self.v_mom_beta.get()})")
            else:
                add("optimizer", f"ADAM({self.v_adam_lr.get()}, {self.v_adam_beta1.get()}, "
                                 f"{self.v_adam_beta2.get()}, {self.v_adam_epsilon.get()})")

        loss = self.v_loss.get()
        if loss == "mse_frames_trajectory":
            add("loss", f"mse_frames_trajectory({self.v_loss_fps.get()})")
        else:
            add("loss", loss)

        add("adjoint_jacobian_window", self.v_adj_win.get())

        return "\n".join(lines) + "\n"

    def _write_temp(self):
        CONFS_DIR.mkdir(exist_ok=True)
        with open(TEMP_CONF, "w") as fh:
            fh.write(self._build_conf_str())

    # ── conf loading ──────────────────────────────────────────────────────────

    def _load_conf(self, path):
        try:
            cfg = _read_conf(path)
        except Exception as exc:
            self._append(f"[ERROR] Cannot load {path}: {exc}\n")
            return

        def set_int(v, key):
            if key in cfg:
                try:
                    v.set(int(cfg[key]))
                except Exception:
                    pass

        def set_str(v, key):
            if key in cfg:
                v.set(cfg[key])

        set_int(self.v_sim_rate,  "sim_rate")
        set_int(self.v_n_seconds, "n_seconds")
        set_int(self.v_fps,       "fps")
        set_int(self.v_adj_win,   "adjoint_jacobian_window")

        if "gravity" in cfg:
            for var, val in zip(self.v_gravity, _parse_vec3(cfg["gravity"])):
                var.set(val)

        set_str(self.v_compliance,        "compliance")
        set_str(self.v_target_compliance, "target_compliance")

        if "offset" in cfg:
            for var, val in zip(self.v_offset, _parse_vec3(cfg["offset"])):
                var.set(val)
        if "target_offset" in cfg:
            for var, val in zip(self.v_target_offset, _parse_vec3(cfg["target_offset"])):
                var.set(val)

        if "obj" in cfg:
            self._parse_obj(cfg["obj"])

        if "collision_mode" in cfg:
            cm = cfg["collision_mode"]
            if cm.startswith("constraints"):
                self.v_coll_mode.set("constraints")
                m = re.search(r'constraints\(([^)]+)\)', cm)
                if m:
                    self.v_coll_compl.set(m.group(1).strip())
            else:
                self.v_coll_mode.set("projection")
            self._on_coll_mode()

        if "colliders" in cfg:
            self._parse_colliders(cfg["colliders"])

        if "export_obj" in cfg:
            self.v_export_obj.set(cfg["export_obj"].strip().lower() == "true")

        if "experiment" in cfg:
            self._parse_experiment(cfg["experiment"])

        if "optimizer" in cfg:
            self._parse_optimizer(cfg["optimizer"])
            self.v_opt_enabled.set(True)
        else:
            self.v_opt_enabled.set(False)
        self._on_opt_enabled()

        if "loss" in cfg:
            self._parse_loss(cfg["loss"])

    def _parse_obj(self, s):
        m = re.match(r'(\w+)\((.+)\)', s.strip())
        if not m:
            return
        kind = m.group(1).strip()
        args = [a.strip() for a in m.group(2).split(",")]
        self.v_obj_type.set(kind)
        self._on_obj_type()
        if kind == "cloth":
            try:
                if len(args) > 0: self.v_cloth_w.set(int(args[0]))
                if len(args) > 1: self.v_cloth_h.set(int(args[1]))
            except Exception:
                pass
            if len(args) > 2:
                self.v_cloth_pin.set(args[2].strip())
            if len(args) > 3:
                flags = args[3]
                self.v_cloth_shear.set("shear" in flags)
                self.v_cloth_bending.set("bending" in flags)
        elif kind == "chain":
            try:
                if len(args) > 0: self.v_chain_len.set(int(args[0]))
            except Exception:
                pass
            if len(args) > 1:
                self.v_chain_pin.set(args[1].strip())

    def _parse_colliders(self, s):
        inner = s.strip().strip("[]").strip()
        if not inner:
            self.v_collider.set("none")
            self._on_collider()
            return
        m = re.match(r'sphere\(\(([^)]+)\),\s*([^)]+)\)', inner)
        if m:
            self.v_collider.set("sphere")
            for var, val in zip(self.v_sphere_center, _parse_vec3(m.group(1))):
                var.set(val)
            self.v_sphere_radius.set(m.group(2).strip())
            self._on_collider()
            return
        m = re.match(r'halfspace\(\(([^)]+)\),\s*\(([^)]+)\)\)', inner)
        if m:
            self.v_collider.set("halfspace")
            for var, val in zip(self.v_hs_origin, _parse_vec3(m.group(1))):
                var.set(val)
            for var, val in zip(self.v_hs_normal, _parse_vec3(m.group(2))):
                var.set(val)
            self._on_collider()
            return
        self.v_collider.set("none")
        self._on_collider()

    def _parse_experiment(self, s):
        s = s.strip()
        m = re.match(r'(\w+)\((.+)\)', s)
        name = m.group(1) if m else s
        args = [a.strip() for a in m.group(2).split(",")] if m else []
        self.v_experiment.set(name)
        try:
            if name == "single_step_jacobian"    and args: self.v_exp_step.set(int(args[0]))
            if name == "compliance_optimization" and args: self.v_exp_iters.set(int(args[0]))
            if name == "loss_scan_compliance" and len(args) >= 3:
                self.v_exp_scan_min.set(args[0])
                self.v_exp_scan_max.set(args[1])
                self.v_exp_scan_steps.set(int(args[2]))
        except Exception:
            pass
        self._on_experiment()

    def _parse_optimizer(self, s):
        m = re.match(r'(\w+)\((.+)\)', s.strip())
        if not m:
            return
        name = m.group(1).strip()
        args = [a.strip() for a in m.group(2).split(",")]
        self.v_opt_type.set(name)
        try:
            if name == "GD"       and args:         self.v_gd_lr.set(args[0])
            if name == "momentum" and len(args) >= 2:
                self.v_mom_lr.set(args[0]); self.v_mom_beta.set(args[1])
            if name == "ADAM"     and len(args) >= 4:
                self.v_adam_lr.set(args[0]); self.v_adam_beta1.set(args[1])
                self.v_adam_beta2.set(args[2]); self.v_adam_epsilon.set(args[3])
        except Exception:
            pass
        self._on_opt_type()

    def _parse_loss(self, s):
        s = s.strip()
        m = re.match(r'mse_frames_trajectory\((\d+)\)', s)
        if m:
            self.v_loss.set("mse_frames_trajectory")
            try:
                self.v_loss_fps.set(int(m.group(1)))
            except Exception:
                pass
        elif s in ("mse_final_position", "mse_full_trajectory"):
            self.v_loss.set(s)
        self._on_loss()

    # ── subprocess ────────────────────────────────────────────────────────────

    def _run(self, cmd):
        self._kill()
        self._write_temp()
        self._clear()
        self._append(f"$ {' '.join(str(c) for c in cmd)}\n\n")
        self._set_status("running", "Running...")

        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(PROJ_ROOT),
            )
        except FileNotFoundError as exc:
            self._append(f"[ERROR] {exc}\n")
            self._set_status("error", f"✗ Error: {exc}")
            return

        self._kill_btn.configure(state="normal")
        threading.Thread(target=self._stream, args=(self._proc,), daemon=True).start()
        self._poll()

    def _stream(self, proc):
        for line in proc.stdout:
            self._q.put(line)
        proc.wait()
        self._q.put(_DONE)

    def _poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                if item is _DONE:
                    rc = self._proc.returncode if self._proc else -1
                    self._set_status("ok" if rc == 0 else "error",
                                     f"Done (exit {rc})")
                    self._kill_btn.configure(state="disabled")
                    self._proc = None
                    return
                self._append(item)
        except queue.Empty:
            pass
        self._after_id = self.after(50, self._poll)

    def _exec_cpp(self):
        self._run([CPP_EXE, TEMP_CONF])

    def _exec_jax(self):
        self._run(["python", "jax_impl.py", TEMP_CONF])

    def _exec_compare(self):
        self._run(["python", "tester.py", TEMP_CONF])

    def _kill(self):
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._append("\n[KILLED]\n")
            self._set_status("error", "✗ Killed")
        self._proc = None
        if self._kill_btn:
            self._kill_btn.configure(state="disabled")

    # ── terminal ──────────────────────────────────────────────────────────────

    def _append(self, text):
        self._terminal.configure(state="normal")
        self._terminal.insert("end", text)
        self._terminal.see("end")
        self._terminal.configure(state="disabled")

    def _clear(self):
        self._terminal.configure(state="normal")
        self._terminal.delete("1.0", "end")
        self._terminal.configure(state="disabled")
        self._set_status("ready", "Ready")

    def _set_status(self, kind, text):
        colors = {"ready": "gray", "running": "#4caf50", "ok": "#4caf50", "error": "#f44336"}
        self._status_var.set(text)
        self._status_lbl.configure(foreground=colors.get(kind, "gray"))

    # ── save / load ───────────────────────────────────────────────────────────

    def _save(self):
        path = self.v_save_path.get().strip()
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            fh.write(self._build_conf_str())
        self._append(f"[Saved] {p}\n")

    def _load_dialog(self):
        path = filedialog.askopenfilename(
            initialdir=str(CONFS_DIR) if CONFS_DIR.exists() else str(PROJ_ROOT),
            filetypes=[("Config files", "*.conf"), ("All files", "*.*")],
            title="Load config",
        )
        if path:
            self.v_save_path.set(path)
            self._load_conf(path)


if __name__ == "__main__":
    App().mainloop()
