"""
=============================================================
  AutoML Neural Network Studio
  Upload any CSV  →  Auto-configure best MLP  →  Predict
=============================================================
  Run:  python app.py
=============================================================
"""

import sys, os, time, threading, warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.neural_network   import MLPClassifier, MLPRegressor
from sklearn.model_selection  import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing    import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose          import ColumnTransformer
from sklearn.metrics          import (accuracy_score, f1_score, confusion_matrix,
                                      mean_squared_error, mean_absolute_error, r2_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# ─────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────
BG   = "#0D0D1A"
PNL  = "#131325"
PNL2 = "#1A1A30"
BDR  = "#252548"
ACC  = "#4FC3F7"
GRN  = "#66BB6A"
ORG  = "#FF8A65"
PUR  = "#CE93D8"
YLW  = "#FFD54F"
RED2 = "#EF5350"
TXT  = "#E8E8F4"
TX2  = "#7777AA"
BTN  = "#1A3A5C"
BTH  = "#1565C0"

FH = ("Segoe UI", 15, "bold")
FS = ("Segoe UI", 10, "bold")
FB = ("Segoe UI", 10)
FM = ("Consolas",  9)
FK = ("Segoe UI",  9)

ACT_CLR = {"relu": ACC, "tanh": ORG, "logistic": GRN}

# ─────────────────────────────────────────────────────────────
#  WIDGETS
# ─────────────────────────────────────────────────────────────
class HBtn(tk.Button):
    """Button that changes colour on hover."""
    def __init__(self, master, hover_bg=None, **kw):
        kw.setdefault("relief", "flat")
        kw.setdefault("bd", 0)
        kw.setdefault("cursor", "hand2")
        kw.setdefault("activeforeground", TXT)
        kw.setdefault("activebackground", BTH)
        kw.setdefault("padx", 14)
        kw.setdefault("pady", 8)
        self._bg  = kw.get("bg", BTN)
        self._hbg = hover_bg or BTH
        super().__init__(master, **kw)
        self.config(bg=self._bg)
        self.bind("<Enter>", lambda _: self.config(bg=self._hbg))
        self.bind("<Leave>", lambda _: self.config(bg=self._bg))


class PBar(tk.Canvas):
    """Animated indeterminate progress bar."""
    def __init__(self, master, **kw):
        kw.setdefault("bg", BG)
        kw.setdefault("height", 5)
        kw.setdefault("highlightthickness", 0)
        super().__init__(master, **kw)
        self._on = False
        self._p  = 0.0

    def start(self):
        self._on = True
        self._p  = 0.0
        self._tick()

    def stop(self):
        self._on = False
        self.delete("all")

    def _tick(self):
        if not self._on:
            return
        w = self.winfo_width() or 600
        self.delete("all")
        self.create_rectangle(0, 0, w, 5, fill=BDR, outline="")
        sw = int(w * 0.30)
        x  = int((self._p % 1.0) * (w + sw)) - sw
        self.create_rectangle(x, 0, x + sw, 5, fill=ACC, outline="")
        self._p += 0.009
        self.after(14, self._tick)


# ─────────────────────────────────────────────────────────────
#  DATA ANALYZER
# ─────────────────────────────────────────────────────────────
class DataAnalyzer:
    def __init__(self):
        self.df            = None
        self.path          = None
        self.target_col    = None
        self.feature_cols  = []
        self.task          = None        # 'classification' | 'regression'
        self.n_classes     = None
        self.class_names   = []
        self.numeric_cols  = []
        self.cat_cols      = []
        self.preprocessor  = None       # fitted ColumnTransformer
        self.target_enc    = None       # fitted LabelEncoder
        self._cat_vals     = {}         # {col: [unique str values]}
        self.X_train       = None
        self.X_test        = None
        self.y_train       = None
        self.y_test        = None
        self.X_train_raw   = None       # raw DataFrame for hints

    # ── load CSV ─────────────────────────────────────────────
    def load(self, path):
        self.path = path
        for enc in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
            try:
                df = pd.read_csv(path, encoding=enc)
                self.df = df
                self.df.columns = self.df.columns.str.strip()
                self.df.dropna(how="all", inplace=True)
                return self.df
            except Exception:
                continue
        raise ValueError("Cannot read file — ensure it is a valid CSV.")

    # ── set target column + detect task ──────────────────────
    def set_target(self, col, task_override=None):
        self.target_col   = col
        self.feature_cols = [c for c in self.df.columns if c != col]
        tgt = self.df[col].dropna()

        if task_override and task_override not in ("Auto", ""):
            self.task = task_override.lower()
        elif tgt.dtype == object or tgt.dtype.name == "category":
            self.task = "classification"
        elif pd.api.types.is_integer_dtype(tgt) and tgt.nunique() <= 20:
            self.task = "classification"
        else:
            self.task = "regression"

        if self.task == "classification":
            self.n_classes   = int(tgt.nunique())
            self.class_names = sorted(tgt.astype(str).unique().tolist())
        else:
            self.n_classes   = None
            self.class_names = []

        feats = self.df[self.feature_cols]
        self.numeric_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols     = feats.select_dtypes(exclude=[np.number]).columns.tolist()

        self._cat_vals = {}
        for c in self.cat_cols:
            self._cat_vals[c] = sorted(self.df[c].dropna().astype(str).unique().tolist())

        return {
            "n_samples":   len(self.df),
            "n_features":  len(self.feature_cols),
            "task":        self.task,
            "n_classes":   self.n_classes,
            "class_names": self.class_names,
            "missing":     int(self.df.isnull().sum().sum()),
            "num_cols":    self.numeric_cols,
            "cat_cols":    self.cat_cols,
        }

    # ── preprocess: scale, encode, split ─────────────────────
    def preprocess(self, test_size=0.2, seed=42):
        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()

        for c in self.numeric_cols:
            X[c] = X[c].fillna(X[c].median())
        for c in self.cat_cols:
            md = X[c].mode()
            X[c] = X[c].fillna(md.iloc[0] if len(md) else "unknown")

        if self.task == "classification":
            self.target_enc = LabelEncoder()
            y = self.target_enc.fit_transform(y.astype(str))
        else:
            y = y.values.astype(float)

        tf = []
        if self.numeric_cols:
            tf.append(("num", StandardScaler(), self.numeric_cols))
        if self.cat_cols:
            tf.append(("cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        self.cat_cols))
        self.preprocessor = ColumnTransformer(transformers=tf, remainder="drop")

        strat = y if self.task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat)

        self.X_train_raw = Xtr
        self.X_train = self.preprocessor.fit_transform(Xtr)
        self.X_test  = self.preprocessor.transform(Xte)
        self.y_train = ytr
        self.y_test  = yte
        return self.X_train, self.X_test, ytr, yte

    # ── transform one sample dict → model input ──────────────
    def transform_single(self, feat_dict):
        row = {c: [feat_dict.get(c, "")] for c in self.feature_cols}
        df  = pd.DataFrame(row)
        for c in self.numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return self.preprocessor.transform(df)

    # ── decode encoded label back to original ────────────────
    def decode_label(self, y_enc):
        if self.target_enc is not None:
            return self.target_enc.inverse_transform(np.atleast_1d(y_enc))
        return np.atleast_1d(y_enc)


# ─────────────────────────────────────────────────────────────
#  MODEL SEARCHER
# ─────────────────────────────────────────────────────────────
class ModelSearcher:
    ARCHS = [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128, 64),
        (64, 32, 16),
        (128, 64, 32, 16),
    ]
    ACTS = ["relu", "tanh", "logistic"]

    def __init__(self, analyzer, callback=None):
        self.analyzer       = analyzer
        self.cb             = callback          # fn(event_name, data_dict)
        self.best_model     = None
        self.best_config    = None
        self.best_score     = -np.inf
        self.search_results = []
        self.final_metrics  = {}
        self.loss_curve     = []
        self.y_pred         = None
        self.y_test_saved   = None
        self._stop          = False

    def stop(self):
        self._stop = True

    def _emit(self, event, data):
        if self.cb:
            self.cb(event, data)

    # ── 18-combo cross-val grid search ───────────────────────
    def search(self, n_folds=3, max_iter=250):
        task  = self.analyzer.task
        X, y  = self.analyzer.X_train, self.analyzer.y_train
        total = len(self.ARCHS) * len(self.ACTS)
        done  = 0

        for arch in self.ARCHS:
            for act in self.ACTS:
                if self._stop:
                    return
                p = dict(hidden_layer_sizes=arch, activation=act,
                         max_iter=max_iter, random_state=42,
                         early_stopping=True, n_iter_no_change=12)
                if task == "classification":
                    mdl = MLPClassifier(**p)
                    cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                    sc  = cross_val_score(mdl, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
                else:
                    mdl = MLPRegressor(**p)
                    cv  = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    sc  = cross_val_score(mdl, X, y, cv=cv, scoring="r2", n_jobs=-1)

                mean, std = float(sc.mean()), float(sc.std())
                done += 1
                rec = {"arch": str(arch), "act": act,
                       "score": mean, "std": std, "pct": done / total}
                self.search_results.append(rec)

                if mean > self.best_score:
                    self.best_score  = mean
                    self.best_config = {"arch": arch, "act": act}
                self._emit("result", rec)

        self._emit("search_done", {"best": self.best_config, "score": self.best_score})

    # ── train the winning config on full training data ────────
    def train_final(self, max_iter=500):
        task  = self.analyzer.task
        arch  = self.best_config["arch"]
        act   = self.best_config["act"]
        Xtr, ytr = self.analyzer.X_train, self.analyzer.y_train
        Xte, yte = self.analyzer.X_test,  self.analyzer.y_test

        p = dict(hidden_layer_sizes=arch, activation=act,
                 max_iter=max_iter, random_state=42,
                 early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=20)
        mdl = (MLPClassifier(**p) if task == "classification" else MLPRegressor(**p))

        t0 = time.time()
        mdl.fit(Xtr, ytr)
        elapsed = time.time() - t0

        self.best_model   = mdl
        self.loss_curve   = mdl.loss_curve_
        yp                = mdl.predict(Xte)
        self.y_pred       = yp
        self.y_test_saved = yte

        _best_loss = float(mdl.best_loss_) if mdl.best_loss_ is not None else float(mdl.loss_)

        if task == "classification":
            self.final_metrics = {
                "Accuracy":   float(accuracy_score(yte, yp)),
                "F1 (macro)": float(f1_score(yte, yp, average="macro", zero_division=0)),
                "Iterations": int(mdl.n_iter_),
                "Best Loss":  _best_loss,
                "Train Time": float(elapsed),
            }
        else:
            mse = float(mean_squared_error(yte, yp))
            self.final_metrics = {
                "R²":         float(r2_score(yte, yp)),
                "RMSE":       float(np.sqrt(mse)),
                "MAE":        float(mean_absolute_error(yte, yp)),
                "Iterations": int(mdl.n_iter_),
                "Best Loss":  _best_loss,
                "Train Time": float(elapsed),
            }
        self._emit("train_done", self.final_metrics)

    # ── predict a single input dict ──────────────────────────
    def predict_one(self, feat_dict):
        X    = self.analyzer.transform_single(feat_dict)
        pred = self.best_model.predict(X)
        proba = None
        if hasattr(self.best_model, "predict_proba"):
            try:
                proba = self.best_model.predict_proba(X)
            except Exception:
                pass
        return pred, proba


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION — 4-Step Wizard
# ─────────────────────────────────────────────────────────────
class App(tk.Tk):
    STEPS = ["1  Upload", "2  Auto-Configure", "3  Results", "4  Predict"]

    def __init__(self):
        super().__init__()
        self.title("AutoML Neural Network Studio")
        self.geometry("1340x900")
        self.minsize(1050, 720)
        self.configure(bg=BG)

        self.analyzer   = DataAnalyzer()
        self.searcher   = None
        self._step      = -1
        self._max_step  = -1            # highest step ever reached
        self._chart_obj = None          # FigureCanvasTkAgg ref to avoid GC
        self._pv        = {}            # {feature_col: tk.StringVar}
        self._pred_hist = []            # [(input_dict, result_str)]

        self._ttk_style()
        self._build_header()
        self._build_stepper()

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._frames = []
        for builder in [self._s1, self._s2, self._s3, self._s4]:
            f = tk.Frame(body, bg=BG)
            f.place(relwidth=1, relheight=1)
            builder(f)
            self._frames.append(f)

        self._go(0)

    # ── TTK Style ────────────────────────────────────────────
    def _ttk_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=PNL, foreground=TXT, font=FB,
                    fieldbackground=PNL2, insertcolor=ACC)
        s.configure("Treeview", background=PNL2, foreground=TXT,
                    fieldbackground=PNL2, rowheight=23, font=FM)
        s.configure("Treeview.Heading", background=BDR, foreground=ACC,
                    relief="flat", font=("Segoe UI", 9, "bold"))
        s.map("Treeview",
              background=[("selected", BTN)], foreground=[("selected", TXT)])
        s.configure("TCombobox", fieldbackground=PNL2, background=PNL2,
                    foreground=TXT, arrowcolor=ACC, selectbackground=BTN)
        s.map("TCombobox",
              fieldbackground=[("readonly", PNL2)],
              selectbackground=[("readonly", BTN)])
        s.configure("TScrollbar", background=BDR, troughcolor=PNL)
        s.configure("TScale", background=PNL, troughcolor=BDR, slidercolor=ACC)

    # ── Header ───────────────────────────────────────────────
    def _build_header(self):
        h = tk.Frame(self, bg=PNL, height=58)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Frame(h, bg=ACC, width=4).pack(side="left", fill="y")
        tk.Label(h, text="⚡  AutoML Neural Network Studio",
                 font=FH, fg=ACC, bg=PNL, padx=16).pack(side="left")
        tk.Label(h, text="Upload any dataset  →  Auto-configure best MLP  →  Predict",
                 font=FK, fg=TX2, bg=PNL).pack(side="left")
        self._stat_dot = tk.Label(h, text="●", font=("Segoe UI", 13), fg=GRN, bg=PNL)
        self._stat_dot.pack(side="right", padx=14)
        self._stat_lbl = tk.Label(h, text="Ready", font=FK, fg=TX2, bg=PNL)
        self._stat_lbl.pack(side="right")

    # ── Step Indicator ───────────────────────────────────────
    def _build_stepper(self):
        sf = tk.Frame(self, bg=PNL2, height=46)
        sf.pack(fill="x")
        sf.pack_propagate(False)
        c = tk.Frame(sf, bg=PNL2)
        c.place(relx=0.5, rely=0.5, anchor="center")
        self._step_labels = []
        for i, name in enumerate(self.STEPS):
            lbl = tk.Label(c, text=f"  {name}  ",
                           font=("Segoe UI", 9, "bold"),
                           fg=TX2, bg=PNL2, padx=8, pady=5, cursor="hand2")
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, idx=i: self._on_step_click(idx))
            self._step_labels.append(lbl)
            if i < len(self.STEPS) - 1:
                tk.Label(c, text="→", fg=BDR, bg=PNL2,
                         font=("Segoe UI", 10)).pack(side="left")

    def _go(self, step):
        if step < 0 or step >= len(self._frames):
            return
        self._step     = step
        self._max_step = max(self._max_step, step)   # never shrinks
        self._frames[step].lift()
        for i, lbl in enumerate(self._step_labels):
            if i == step:
                lbl.config(fg=ACC, bg=PNL2)          # current  → cyan
            elif i <= self._max_step:
                lbl.config(fg=GRN, bg=PNL2)          # reached  → green (clickable)
            else:
                lbl.config(fg=TX2, bg=PNL2)          # unreached → grey

    def _on_step_click(self, idx):
        # allow jumping to any step that has already been reached
        if idx <= self._max_step:
            self._go(idx)

    def _set_status(self, txt, color=None):
        self._stat_lbl.config(text=txt)
        if color:
            self._stat_dot.config(fg=color)

    # ─────────────────────────────────────────────────────────
    #  STEP 1 — Upload & Preview
    # ─────────────────────────────────────────────────────────
    def _s1(self, parent):
        # ── file browser bar ──
        top = tk.Frame(parent, bg=PNL)
        top.pack(fill="x", padx=14, pady=(12, 6))
        tk.Frame(top, bg=ACC, height=3).pack(fill="x")
        row = tk.Frame(top, bg=PNL, padx=14, pady=10)
        row.pack(fill="x")
        tk.Label(row, text="📂  Dataset File", font=FS, fg=TXT, bg=PNL).pack(side="left")
        self._filepath_var = tk.StringVar(value="No file selected  —  click Browse to load any CSV")
        tk.Label(row, textvariable=self._filepath_var,
                 font=FK, fg=TX2, bg=PNL, anchor="w").pack(side="left", fill="x", expand=True, padx=12)
        HBtn(row, text="Browse…", bg=BTN, hover_bg=BTH,
             fg=TXT, font=FS, command=self._browse).pack(side="right")

        # ── body: left config + right preview ──
        body = tk.Frame(parent, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        # ─ Left panel ─
        left = tk.Frame(body, bg=PNL, width=295)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        tk.Frame(left, bg=PUR, height=3).pack(fill="x")
        tk.Label(left, text="Configuration", font=FS, fg=TXT, bg=PNL,
                 padx=12, pady=8).pack(anchor="w")
        tk.Frame(left, bg=BDR, height=1).pack(fill="x")

        cfg = tk.Frame(left, bg=PNL, padx=12, pady=10)
        cfg.pack(fill="both", expand=True)

        # info labels
        self._info = {}
        for key, default in [("Samples", "—"), ("Features", "—"),
                              ("Missing", "—"), ("Task", "—"), ("Classes", "—")]:
            r = tk.Frame(cfg, bg=PNL)
            r.pack(fill="x", pady=3)
            tk.Label(r, text=key + ":", font=("Segoe UI", 9, "bold"),
                     fg=TX2, bg=PNL, width=10, anchor="w").pack(side="left")
            lbl = tk.Label(r, text=default, font=FK, fg=TXT, bg=PNL, anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            self._info[key] = lbl

        tk.Frame(cfg, bg=BDR, height=1).pack(fill="x", pady=8)

        tk.Label(cfg, text="Target Column  (Y to predict)",
                 font=("Segoe UI", 9, "bold"), fg=TX2, bg=PNL, anchor="w").pack(fill="x")
        self._target_var = tk.StringVar()
        self._target_cb  = ttk.Combobox(cfg, textvariable=self._target_var,
                                         state="readonly", font=FK)
        self._target_cb.pack(fill="x", pady=(4, 8))
        self._target_cb.bind("<<ComboboxSelected>>", self._on_target)

        tk.Label(cfg, text="Task Type  (auto-detected)",
                 font=("Segoe UI", 9, "bold"), fg=TX2, bg=PNL, anchor="w").pack(fill="x")
        self._task_var = tk.StringVar(value="Auto")
        ttk.Combobox(cfg, textvariable=self._task_var,
                     values=["Auto", "Classification", "Regression"],
                     state="readonly", font=FK).pack(fill="x", pady=(4, 8))

        tk.Label(cfg, text="Test Split",
                 font=("Segoe UI", 9, "bold"), fg=TX2, bg=PNL, anchor="w").pack(fill="x")
        split_row = tk.Frame(cfg, bg=PNL)
        split_row.pack(fill="x", pady=(4, 0))
        self._split_var = tk.DoubleVar(value=0.2)
        self._split_lbl = tk.Label(split_row, text="20%", font=FK, fg=ACC, bg=PNL, width=4)
        self._split_lbl.pack(side="right")
        ttk.Scale(split_row, from_=0.05, to=0.40, variable=self._split_var,
                  orient="horizontal",
                  command=lambda v: self._split_lbl.config(
                      text=f"{round(float(v) * 100)}%")
                  ).pack(side="left", fill="x", expand=True)

        # Continue button
        tk.Frame(left, bg=BDR, height=1).pack(fill="x")
        bf = tk.Frame(left, bg=PNL, padx=12, pady=10)
        bf.pack(fill="x")
        self._s1_btn = HBtn(bf, text="Continue  →",
                            bg=ACC, hover_bg="#0288D1",
                            fg="#000D1A", font=FS,
                            command=self._s1_continue,
                            padx=0, pady=10, state="disabled")
        self._s1_btn.pack(fill="x")

        # ─ Right panel: preview table ─
        right = tk.Frame(body, bg=PNL)
        right.pack(side="left", fill="both", expand=True)
        tk.Frame(right, bg=GRN, height=3).pack(fill="x")
        tk.Label(right, text="Data Preview  (up to 100 rows)",
                 font=FS, fg=TXT, bg=PNL, padx=12, pady=8).pack(anchor="w")
        tk.Frame(right, bg=BDR, height=1).pack(fill="x")

        tc = tk.Frame(right, bg=PNL, padx=8, pady=8)
        tc.pack(fill="both", expand=True)
        self._prev_tree = ttk.Treeview(tc, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(tc, orient="vertical",   command=self._prev_tree.yview)
        hsb = ttk.Scrollbar(tc, orient="horizontal", command=self._prev_tree.xview)
        self._prev_tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        self._prev_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tc.rowconfigure(0, weight=1)
        tc.columnconfigure(0, weight=1)
        self._prev_tree.tag_configure("even", background=PNL2)
        self._prev_tree.tag_configure("odd",  background=PNL)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select CSV dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = self.analyzer.load(path)
            self._filepath_var.set(os.path.basename(path))
            cols = list(df.columns)
            self._target_cb.config(values=cols)
            self._target_var.set(cols[-1])
            self._load_preview(df)
            self._on_target()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _load_preview(self, df):
        self._prev_tree.delete(*self._prev_tree.get_children())
        cols = list(df.columns)
        self._prev_tree.configure(columns=cols)
        for c in cols:
            self._prev_tree.heading(c, text=c)
            self._prev_tree.column(c, width=max(80, len(c) * 9),
                                   anchor="center", minwidth=55)
        for i, (_, row) in enumerate(df.head(100).iterrows()):
            tag = "even" if i % 2 == 0 else "odd"
            self._prev_tree.insert("", "end", values=list(row.astype(str)), tags=(tag,))

    def _on_target(self, event=None):
        col  = self._target_var.get()
        task = self._task_var.get()
        if not col or self.analyzer.df is None:
            return
        info = self.analyzer.set_target(col, task if task != "Auto" else None)
        self._info["Samples"].config(text=f"{info['n_samples']:,}")
        self._info["Features"].config(text=str(info["n_features"]))
        self._info["Missing"].config(text=str(info["missing"]))
        task_txt = info["task"].title()
        self._info["Task"].config(text=task_txt,
                                  fg=ACC if task_txt == "Classification" else ORG)
        nc = info["n_classes"]
        clns = info["class_names"]
        classes_txt = ("N/A" if nc is None else
                        f"{nc}  {clns}" if nc and nc <= 6 else str(nc))
        self._info["Classes"].config(text=classes_txt)
        self._s1_btn.config(state="normal")

    def _s1_continue(self):
        if self.analyzer.df is None or not self._target_var.get():
            messagebox.showwarning("Missing", "Please load a dataset and select a target column.")
            return
        task = self._task_var.get()
        self.analyzer.set_target(self._target_var.get(),
                                  task if task != "Auto" else None)
        try:
            self.analyzer.preprocess(test_size=self._split_var.get())
        except Exception as e:
            messagebox.showerror("Preprocessing Error", str(e))
            return
        self._go(1)
        self.after(80, self._start_search)

    # ─────────────────────────────────────────────────────────
    #  STEP 2 — Auto-Configure (live grid search)
    # ─────────────────────────────────────────────────────────
    def _s2(self, parent):
        # top bar
        top = tk.Frame(parent, bg=PNL)
        top.pack(fill="x", padx=14, pady=(12, 6))
        tk.Frame(top, bg=ORG, height=3).pack(fill="x")
        head_row = tk.Frame(top, bg=PNL, padx=14, pady=10)
        head_row.pack(fill="x")
        tk.Label(head_row, text="🔍  Searching Best MLP Configuration  (18 combos · 3-fold CV)",
                 font=FS, fg=TXT, bg=PNL).pack(side="left")
        self._s2_status = tk.Label(head_row, text="", font=FK, fg=TX2, bg=PNL)
        self._s2_status.pack(side="right")

        self._s2_pbar = PBar(parent)
        self._s2_pbar.pack(fill="x", padx=14, pady=(0, 4))

        body = tk.Frame(parent, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        # ─ Left: best config card ─
        left = tk.Frame(body, bg=PNL, width=290)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        tk.Frame(left, bg=YLW, height=3).pack(fill="x")
        tk.Label(left, text="🏆  Best Config So Far",
                 font=FS, fg=TXT, bg=PNL, padx=12, pady=8).pack(anchor="w")
        tk.Frame(left, bg=BDR, height=1).pack(fill="x")

        bc_inner = tk.Frame(left, bg=PNL, padx=14, pady=14)
        bc_inner.pack(fill="both", expand=True)
        self._best_labels = {}
        for key in ["Architecture", "Activation", "Best CV Score", "Progress"]:
            r = tk.Frame(bc_inner, bg=PNL)
            r.pack(fill="x", pady=6)
            tk.Label(r, text=key + ":", font=("Segoe UI", 9, "bold"),
                     fg=TX2, bg=PNL, anchor="w").pack(fill="x")
            lbl = tk.Label(r, text="—", font=("Segoe UI", 11, "bold"),
                           fg=TXT, bg=PNL, anchor="w")
            lbl.pack(fill="x")
            self._best_labels[key] = lbl

        tk.Frame(bc_inner, bg=BDR, height=1).pack(fill="x", pady=10)
        self._s2_phase_lbl = tk.Label(bc_inner, text="", font=FS,
                                       fg=ACC, bg=PNL, wraplength=240, justify="left")
        self._s2_phase_lbl.pack(fill="x")

        # ─ Right: live results treeview ─
        right = tk.Frame(body, bg=PNL)
        right.pack(side="left", fill="both", expand=True)
        tk.Frame(right, bg=ACC, height=3).pack(fill="x")
        tk.Label(right, text="Search Results",
                 font=FS, fg=TXT, bg=PNL, padx=12, pady=8).pack(anchor="w")
        tk.Frame(right, bg=BDR, height=1).pack(fill="x")

        tc = tk.Frame(right, bg=PNL, padx=8, pady=8)
        tc.pack(fill="both", expand=True)
        cols = ("#", "Architecture", "Activation", "CV Score", "Std Dev", "Status")
        self._s2_tree = ttk.Treeview(tc, columns=cols, show="headings",
                                      selectmode="browse")
        for c, w in zip(cols, [36, 210, 100, 110, 90, 80]):
            self._s2_tree.heading(c, text=c)
            self._s2_tree.column(c, width=w, anchor="center", minwidth=40)
        vsb = ttk.Scrollbar(tc, orient="vertical", command=self._s2_tree.yview)
        self._s2_tree.configure(yscroll=vsb.set)
        self._s2_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._s2_tree.tag_configure("best",    foreground=YLW,
                                     font=("Segoe UI", 9, "bold"))
        self._s2_tree.tag_configure("relu",    foreground=ACC)
        self._s2_tree.tag_configure("tanh",    foreground=ORG)
        self._s2_tree.tag_configure("logistic",foreground=GRN)
        self._s2_tree.tag_configure("even",    background=PNL2)
        self._s2_tree.tag_configure("odd",     background=PNL)
        self._s2_iids = []

    def _start_search(self):
        self.searcher = ModelSearcher(self.analyzer, callback=self._search_cb)
        # reset UI
        self._s2_tree.delete(*self._s2_tree.get_children())
        self._s2_iids.clear()
        for k in self._best_labels:
            self._best_labels[k].config(text="—", fg=TXT)
        self._s2_status.config(text="Starting…")
        self._s2_phase_lbl.config(text="Grid search running…", fg=ACC)
        self._s2_pbar.start()
        self._set_status("Searching…", YLW)
        threading.Thread(target=self._search_thread, daemon=True).start()

    def _search_thread(self):
        try:
            self.searcher.search()
            if not self.searcher._stop:
                self.searcher.train_final()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Error", f"{e}\n\n{tb}"))
            self.after(0, lambda: self._set_status("Error", RED2))
            self.after(0, self._s2_pbar.stop)

    def _search_cb(self, event, data):
        """Relay background-thread callbacks safely to main thread."""
        self.after(0, lambda e=event, d=data: self._handle_search_event(e, d))

    def _handle_search_event(self, event, data):
        if event == "result":
            self._add_s2_row(data)
        elif event == "search_done":
            self._s2_phase_lbl.config(text="⚙  Training final model on full dataset…", fg=ORG)
            self._s2_status.config(text="Training…")
            self._set_status("Training…", ORG)
            self._highlight_best_row()
        elif event == "train_done":
            self._s2_pbar.stop()
            self._s2_phase_lbl.config(text="✔  Done!  Generating results…", fg=GRN)
            self._s2_status.config(text="Complete")
            self._set_status("Ready", GRN)
            # Unlock ALL steps in stepper so user can freely jump to Predict
            self._max_step = 3
            for i, lbl in enumerate(self._step_labels):
                lbl.config(fg=(GRN if i != self._step else ACC), bg=PNL2)
            self.after(150, self._populate_s3)
            self.after(600, lambda: self._go(2))

    def _add_s2_row(self, rec):
        i    = len(self._s2_iids)
        even = "even" if i % 2 == 0 else "odd"
        act  = rec["act"]
        iid  = self._s2_tree.insert("", "end",
                                     values=(i + 1, rec["arch"], act,
                                             f"{rec['score']:.4f}",
                                             f"±{rec['std']:.4f}", "✓"),
                                     tags=(even, act))
        self._s2_iids.append(iid)
        self._s2_tree.see(iid)
        # live-update best card
        if self.searcher.best_config:
            bc = self.searcher.best_config
            self._best_labels["Architecture"].config(
                text=str(bc["arch"]), fg=ACC)
            self._best_labels["Activation"].config(
                text=bc["act"], fg=ACT_CLR.get(bc["act"], TXT))
            self._best_labels["Best CV Score"].config(
                text=f"{self.searcher.best_score:.4f}", fg=GRN)
        pct = round(rec["pct"] * 100)
        self._best_labels["Progress"].config(text=f"{pct}%  ({i+1}/18)", fg=TX2)
        self._s2_status.config(text=f"{pct}% complete…")

    def _highlight_best_row(self):
        if not self.searcher or not self.searcher.best_config:
            return
        ba = str(self.searcher.best_config["arch"])
        bk = self.searcher.best_config["act"]
        for iid in self._s2_iids:
            vals = self._s2_tree.item(iid, "values")
            if vals and vals[1] == ba and vals[2] == bk:
                self._s2_tree.item(iid, tags=("best",))
                self._s2_tree.see(iid)
                break

    # ─────────────────────────────────────────────────────────
    #  STEP 3 — Results
    # ─────────────────────────────────────────────────────────
    def _s3(self, parent):
        # metric cards row placeholder
        self._s3_cards = tk.Frame(parent, bg=BG)
        self._s3_cards.pack(fill="x", padx=14, pady=(12, 6))

        # config summary bar
        self._s3_cfg_bar = tk.Frame(parent, bg=PNL)
        self._s3_cfg_bar.pack(fill="x", padx=14, pady=(0, 6))
        tk.Frame(self._s3_cfg_bar, bg=GRN, height=3).pack(fill="x")
        self._s3_cfg_lbl = tk.Label(self._s3_cfg_bar,
                                     text="Training complete — summary will appear here.",
                                     font=FK, fg=TX2, bg=PNL, padx=12, pady=8)
        self._s3_cfg_lbl.pack(anchor="w")

        # chart container
        self._s3_chart_frame = tk.Frame(parent, bg=PNL2)

        # "Go to Predict" button row — packed BEFORE chart so it stays visible
        bf = tk.Frame(parent, bg=BG)
        bf.pack(fill="x", padx=14, pady=(0, 8))
        HBtn(bf, text="Go to Predict  →",
             bg=PUR, hover_bg="#8E44AD", fg="white", font=FS,
             command=lambda: self._go(3)).pack(side="right")

        # chart takes all remaining vertical space
        self._s3_chart_frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        tk.Label(self._s3_chart_frame,
                 text="Charts will appear after training completes.",
                 fg=TX2, bg=PNL2, font=FB).place(relx=0.5, rely=0.5, anchor="center")

    def _populate_s3(self):
        if not self.searcher or not self.searcher.final_metrics:
            return
        m    = self.searcher.final_metrics
        task = self.analyzer.task

        # ── metric cards ──
        for w in self._s3_cards.winfo_children():
            w.destroy()
        tk.Label(self._s3_cards, text="Final Model Metrics",
                 font=("Segoe UI", 9, "bold"), fg=TX2, bg=BG).pack(side="left", padx=(0, 16))

        if task == "classification":
            defs = [
                ("Accuracy",   f"{m['Accuracy'] * 100:.2f}%", GRN),
                ("F1 (macro)", f"{m['F1 (macro)']:.4f}",      ACC),
                ("Iterations", str(m["Iterations"]),            TX2),
                ("Best Loss",  f"{m['Best Loss']:.4f}",        ORG),
                ("Train Time", f"{m['Train Time']:.1f}s",      PUR),
            ]
        else:
            defs = [
                ("R²",         f"{m['R²']:.4f}",         GRN),
                ("RMSE",       f"{m['RMSE']:.4f}",        ACC),
                ("MAE",        f"{m['MAE']:.4f}",         ORG),
                ("Iterations", str(m["Iterations"]),       TX2),
                ("Train Time", f"{m['Train Time']:.1f}s", PUR),
            ]

        for label, value, color in defs:
            card = tk.Frame(self._s3_cards, bg=PNL, padx=18, pady=8)
            card.pack(side="left", padx=(0, 8))
            tk.Frame(card, bg=color, height=3).pack(fill="x")
            tk.Label(card, text=value, font=("Segoe UI", 15, "bold"),
                     fg=color, bg=PNL, pady=4).pack()
            tk.Label(card, text=label, font=FK, fg=TX2, bg=PNL).pack()

        # ── config summary ──
        bc = self.searcher.best_config
        n_tr = len(self.analyzer.y_train)
        n_te = len(self.analyzer.y_test)
        self._s3_cfg_lbl.config(
            text=(f"  ✔  Best Config →  Architecture: {bc['arch']}  |  "
                  f"Activation: {bc['act']}  |  "
                  f"CV Score: {self.searcher.best_score:.4f}  |  "
                  f"Train: {n_tr:,} rows  ·  Test: {n_te:,} rows"),
            fg=TXT)

        # ── destroy old chart embed ──
        for w in self._s3_chart_frame.winfo_children():
            w.destroy()
        if self._chart_obj:
            plt.close(self._chart_obj.figure)
            self._chart_obj = None

        fig = plt.figure(figsize=(16, 5.5), facecolor="#1A1A30")
        if task == "classification":
            self._draw_clf_charts(fig)
        else:
            self._draw_reg_charts(fig)

        canvas = FigureCanvasTkAgg(fig, master=self._s3_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._chart_obj = canvas

        # build prediction form now that model is ready
        self._build_predict_form()

    def _draw_clf_charts(self, fig):
        yte  = self.searcher.y_test_saved
        yp   = self.searcher.y_pred
        cm   = confusion_matrix(yte, yp)
        clns = (list(self.analyzer.target_enc.classes_)
                if self.analyzer.target_enc else self.analyzer.class_names)
        TEXT = "#E8E8F4"; BG2 = "#1A1A30"; GRD = "#252548"

        gs = fig.add_gridspec(1, 3, wspace=0.40,
                              left=0.05, right=0.97, top=0.88, bottom=0.15)

        # confusion matrix
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(BG2)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        sns.heatmap(cm_norm, ax=ax1, cmap="YlOrRd", annot=cm, fmt="d",
                    xticklabels=clns, yticklabels=clns,
                    linewidths=0.5, cbar=False,
                    annot_kws={"size": 9, "color": "#1a1a1a"})
        ax1.set_title("Confusion Matrix", color=TEXT, fontsize=10, pad=8)
        ax1.tick_params(colors=TEXT, labelsize=8, rotation=0)
        ax1.set_xlabel("Predicted", color=TEXT, fontsize=8)
        ax1.set_ylabel("Actual",    color=TEXT, fontsize=8)
        for sp in ax1.spines.values(): sp.set_color(GRD)

        # per-class accuracy
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(BG2)
        per_class = cm.diagonal() / cm.sum(axis=1).clip(1)
        cmap   = plt.cm.get_cmap("cool", len(clns))
        colors = [cmap(i) for i in range(len(clns))]
        bars   = ax2.barh(clns, per_class * 100, color=colors, height=0.6)
        for bar, v in zip(bars, per_class):
            ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                     f"{v*100:.1f}%", va="center", ha="left",
                     color=TEXT, fontsize=8)
        ax2.set_title("Per-Class Accuracy", color=TEXT, fontsize=10, pad=8)
        ax2.tick_params(colors=TEXT, labelsize=8)
        ax2.set_xlabel("Accuracy %", color=TEXT, fontsize=8)
        ax2.set_xlim(0, 118)
        ax2.grid(axis="x", color=GRD, alpha=0.5)
        for sp in ax2.spines.values(): sp.set_color(GRD)

        # loss curve
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(BG2)
        lc = self.searcher.loss_curve
        ax3.plot(lc, color=ACC, linewidth=1.8)
        ax3.fill_between(range(len(lc)), lc, color=ACC, alpha=0.12)
        ax3.set_title("Training Loss Curve", color=TEXT, fontsize=10, pad=8)
        ax3.set_xlabel("Epoch", color=TEXT, fontsize=8)
        ax3.set_ylabel("Loss",  color=TEXT, fontsize=8)
        ax3.tick_params(colors=TEXT, labelsize=8)
        ax3.grid(color=GRD, alpha=0.5)
        for sp in ax3.spines.values(): sp.set_color(GRD)

    def _draw_reg_charts(self, fig):
        yte  = self.searcher.y_test_saved
        yp   = self.searcher.y_pred
        res  = yte - yp
        TEXT = "#E8E8F4"; BG2 = "#1A1A30"; GRD = "#252548"

        gs = fig.add_gridspec(1, 3, wspace=0.40,
                              left=0.07, right=0.97, top=0.88, bottom=0.15)

        # actual vs predicted
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(BG2)
        ax1.scatter(yte, yp, alpha=0.30, s=8, color=ACC)
        mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
        ax1.plot([mn, mx], [mn, mx], color=GRN, linewidth=1.5, linestyle="--")
        ax1.set_title("Actual vs Predicted", color=TEXT, fontsize=10, pad=8)
        ax1.set_xlabel("Actual",    color=TEXT, fontsize=8)
        ax1.set_ylabel("Predicted", color=TEXT, fontsize=8)
        ax1.tick_params(colors=TEXT, labelsize=8)
        ax1.grid(color=GRD, alpha=0.5)
        for sp in ax1.spines.values(): sp.set_color(GRD)

        # residuals distribution
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(BG2)
        ax2.hist(res, bins=40, color=ORG, alpha=0.85, edgecolor="none")
        ax2.axvline(0, color=GRN, linewidth=1.5, linestyle="--")
        ax2.set_title("Residuals Distribution", color=TEXT, fontsize=10, pad=8)
        ax2.set_xlabel("Residual", color=TEXT, fontsize=8)
        ax2.set_ylabel("Count",    color=TEXT, fontsize=8)
        ax2.tick_params(colors=TEXT, labelsize=8)
        ax2.grid(color=GRD, alpha=0.5)
        for sp in ax2.spines.values(): sp.set_color(GRD)

        # loss curve
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(BG2)
        lc = self.searcher.loss_curve
        ax3.plot(lc, color=ACC, linewidth=1.8)
        ax3.fill_between(range(len(lc)), lc, color=ACC, alpha=0.12)
        ax3.set_title("Training Loss Curve", color=TEXT, fontsize=10, pad=8)
        ax3.set_xlabel("Epoch", color=TEXT, fontsize=8)
        ax3.set_ylabel("Loss",  color=TEXT, fontsize=8)
        ax3.tick_params(colors=TEXT, labelsize=8)
        ax3.grid(color=GRD, alpha=0.5)
        for sp in ax3.spines.values(): sp.set_color(GRD)

    # ─────────────────────────────────────────────────────────
    #  STEP 4 — Predict
    # ─────────────────────────────────────────────────────────
    def _s4(self, parent):
        # ─ Left: scrollable input form ─
        left = tk.Frame(parent, bg=PNL, width=380)
        left.pack(side="left", fill="y", padx=(14, 6), pady=12)
        left.pack_propagate(False)

        tk.Frame(left, bg=PUR, height=3).pack(fill="x")
        tk.Label(left, text="🔮  Enter Feature Values",
                 font=FS, fg=TXT, bg=PNL, padx=12, pady=8).pack(anchor="w")
        tk.Frame(left, bg=BDR, height=1).pack(fill="x")

        # scrollable form area
        outer = tk.Frame(left, bg=PNL)
        outer.pack(fill="both", expand=True)

        self._form_canvas = tk.Canvas(outer, bg=PNL, highlightthickness=0, width=360)
        form_vsb = tk.Scrollbar(outer, orient="vertical",
                                 command=self._form_canvas.yview)
        self._form_canvas.configure(yscrollcommand=form_vsb.set)
        form_vsb.pack(side="right", fill="y")
        self._form_canvas.pack(side="left", fill="both", expand=True)

        self._form_inner = tk.Frame(self._form_canvas, bg=PNL)
        self._form_win   = self._form_canvas.create_window(
            (0, 0), window=self._form_inner, anchor="nw")

        self._form_inner.bind("<Configure>", lambda e: self._form_canvas.configure(
            scrollregion=self._form_canvas.bbox("all")))
        self._form_canvas.bind("<Configure>", lambda e: self._form_canvas.itemconfig(
            self._form_win, width=e.width))

        # mousewheel scroll
        def _on_wheel(event):
            self._form_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        self._form_canvas.bind_all("<MouseWheel>", _on_wheel)

        tk.Label(self._form_inner,
                 text="Form will appear after training completes.",
                 font=FB, fg=TX2, bg=PNL, pady=40, padx=20).pack()

        # Predict button
        tk.Frame(left, bg=BDR, height=1).pack(fill="x")
        pbf = tk.Frame(left, bg=PNL, padx=12, pady=10)
        pbf.pack(fill="x")
        self._pred_btn = HBtn(pbf, text="🔮  Predict",
                              bg=PUR, hover_bg="#8E44AD",
                              fg="white", font=FS,
                              command=self._run_predict,
                              padx=0, pady=10, state="disabled")
        self._pred_btn.pack(fill="x")

        # ─ Right: result + history ─
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 14), pady=12)

        # result card
        self._result_frame = tk.Frame(right, bg=PNL)
        self._result_frame.pack(fill="x")
        tk.Frame(self._result_frame, bg=YLW, height=3).pack(fill="x")
        tk.Label(self._result_frame, text="Prediction Result",
                 font=FS, fg=TXT, bg=PNL, padx=12, pady=8).pack(anchor="w")
        tk.Frame(self._result_frame, bg=BDR, height=1).pack(fill="x")
        self._result_inner = tk.Frame(self._result_frame, bg=PNL, padx=16, pady=12)
        self._result_inner.pack(fill="both")
        tk.Label(self._result_inner,
                 text="Run a prediction to see the result here.",
                 font=FB, fg=TX2, bg=PNL).pack(pady=8)

        # probability bars (classification)
        self._proba_frame = tk.Frame(right, bg=PNL, padx=14, pady=10)
        self._proba_frame.pack(fill="x", pady=(6, 0))

        # history table
        hist = tk.Frame(right, bg=PNL)
        hist.pack(fill="both", expand=True, pady=(8, 0))
        tk.Frame(hist, bg=BDR, height=3).pack(fill="x")
        tk.Label(hist, text="Prediction History",
                 font=FS, fg=TXT, bg=PNL, padx=12, pady=8).pack(anchor="w")
        tk.Frame(hist, bg=BDR, height=1).pack(fill="x")
        hc = tk.Frame(hist, bg=PNL, padx=8, pady=8)
        hc.pack(fill="both", expand=True)
        self._hist_tree = ttk.Treeview(hc, columns=("#", "Result", "Input Summary"),
                                        show="headings", selectmode="browse", height=8)
        self._hist_tree.heading("#",             text="#")
        self._hist_tree.heading("Result",        text="Prediction")
        self._hist_tree.heading("Input Summary", text="Feature Values")
        self._hist_tree.column("#",             width=40,  anchor="center")
        self._hist_tree.column("Result",        width=170, anchor="center")
        self._hist_tree.column("Input Summary", width=600, anchor="w")
        hsb2 = ttk.Scrollbar(hc, orient="vertical", command=self._hist_tree.yview)
        self._hist_tree.configure(yscroll=hsb2.set)
        self._hist_tree.pack(side="left", fill="both", expand=True)
        hsb2.pack(side="right", fill="y")
        self._hist_tree.tag_configure("odd",  background=PNL2)
        self._hist_tree.tag_configure("even", background=PNL)

    def _build_predict_form(self):
        """Dynamically build the input form based on the dataset's feature types."""
        a = self.analyzer
        for w in self._form_inner.winfo_children():
            w.destroy()
        self._pv.clear()

        clr_cycle = [ACC, ORG, GRN, PUR, YLW]
        raw = a.X_train_raw

        for i, col in enumerate(a.feature_cols):
            var = tk.StringVar()
            self._pv[col] = var
            row = tk.Frame(self._form_inner, bg=PNL)
            row.pack(fill="x", padx=10, pady=3)
            clr = clr_cycle[i % len(clr_cycle)]

            # colour dot
            tk.Label(row, text="●", font=("Segoe UI", 9),
                     fg=clr, bg=PNL).pack(side="left")

            # column name label
            tk.Label(row, text=col, font=("Segoe UI", 9, "bold"),
                     fg=TX2, bg=PNL, width=20, anchor="w",
                     wraplength=150).pack(side="left", padx=(4, 0))

            if col in a.cat_cols:
                options = a._cat_vals.get(col, [""])
                var.set(options[0] if options else "")
                ttk.Combobox(row, textvariable=var, values=options,
                             state="readonly", font=FK, width=16
                             ).pack(side="left", padx=(4, 0))
            else:
                # default to column mean as hint
                try:
                    mean_val = float(raw[col].mean())
                    var.set(f"{mean_val:.5g}")
                except Exception:
                    var.set("0")
                tk.Entry(row, textvariable=var, font=FM,
                         fg=TXT, bg=PNL2,
                         insertbackground=ACC,
                         relief="flat", bd=4, width=18).pack(side="left", padx=(4, 0))

        self._pred_btn.config(state="normal")
        # reset canvas scroll to top
        self._form_canvas.yview_moveto(0)

    def _run_predict(self):
        if not self.searcher or not self.searcher.best_model:
            messagebox.showwarning("Not Ready", "Please complete training first.")
            return
        feat_dict = {col: v.get() for col, v in self._pv.items()}
        try:
            pred, proba = self.searcher.predict_one(feat_dict)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        task = self.analyzer.task
        if task == "classification":
            decoded = self.analyzer.decode_label(pred)
            result_str = str(decoded[0])
            result_color = PUR
        else:
            result_str = f"{float(pred[0]):.6g}"
            result_color = GRN

        # ── update result card ──
        for w in self._result_inner.winfo_children():
            w.destroy()
        tk.Label(self._result_inner, text="Predicted Value:",
                 font=("Segoe UI", 9, "bold"), fg=TX2, bg=PNL).pack(anchor="w")
        tk.Label(self._result_inner, text=result_str,
                 font=("Segoe UI", 28, "bold"),
                 fg=result_color, bg=PNL, pady=6).pack(anchor="w")

        # ── class probability bars ──
        for w in self._proba_frame.winfo_children():
            w.destroy()
        if task == "classification" and proba is not None:
            p_vals  = proba[0]
            classes = self.analyzer.class_names
            tk.Label(self._proba_frame, text="Class Probabilities:",
                     font=("Segoe UI", 9, "bold"), fg=TX2, bg=PNL).pack(anchor="w")
            clr_list = [ACC, ORG, GRN, PUR, YLW, RED2]
            for j, (cls, pv) in enumerate(zip(classes, p_vals)):
                brow = tk.Frame(self._proba_frame, bg=PNL)
                brow.pack(fill="x", pady=2)
                clr = clr_list[j % len(clr_list)]
                tk.Label(brow, text=str(cls), font=FK,
                         fg=clr, bg=PNL, width=14, anchor="w").pack(side="left")
                bg_bar = tk.Frame(brow, bg=BDR, height=16, width=220)
                bg_bar.pack(side="left", padx=4)
                bg_bar.pack_propagate(False)
                fill_w = max(2, int(220 * pv))
                tk.Frame(bg_bar, bg=clr, height=16, width=fill_w).pack(side="left")
                tk.Label(brow, text=f"{pv * 100:.1f}%", font=FK,
                         fg=TXT, bg=PNL, width=7).pack(side="left")

        # ── add to history ──
        n   = len(self._pred_hist) + 1
        self._pred_hist.append((feat_dict, result_str))
        summary = "  ·  ".join(f"{k}={v}" for k, v in list(feat_dict.items())[:5])
        if len(feat_dict) > 5:
            summary += f"  …+{len(feat_dict)-5} more"
        tag = "odd" if n % 2 else "even"
        iid = self._hist_tree.insert("", 0,
                                      values=(n, result_str, summary),
                                      tags=(tag,))
        self._hist_tree.see(iid)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
