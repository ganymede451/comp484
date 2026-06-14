"""
Pix2Pix Inference App — Sketch to Photo
COMP 484 | Kathmandu University
"""

import sys
import os
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading

tf        = None
generator = None

IMG_SIZE  = 256
APP_TITLE = "Pix2Pix  ·  Sketch → Photo"
ACCENT    = "#7C6EF7"
BG_DARK   = "#0D0D12"
BG_PANEL  = "#16161F"
BG_CARD   = "#1E1E2A"
TEXT_PRI  = "#F0EFF8"
TEXT_SEC  = "#8A8AAB"
SUCCESS   = "#4ECDC4"
ERROR_CLR = "#FF5F6D"
BORDER    = "#2A2A3C"
BTN_HOVER = "#6355E0"


class Pix2PixApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1060x700")
        self.minsize(860, 600)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.model_path    = tk.StringVar()
        self.input_path    = tk.StringVar()
        self.output_image  = None
        self._input_pil    = None
        self._model_loaded = False

        self._build_ui()
        self._log("Ready. Load a model, then open a sketch.")

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        topbar = tk.Frame(self, bg=BG_DARK, height=56)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="pix2pix", font=("Courier New", 18, "bold"),
                 fg=ACCENT, bg=BG_DARK).pack(side="left", padx=20, pady=10)
        tk.Label(topbar, text="sketch → photo", font=("Helvetica", 11),
                 fg=TEXT_SEC, bg=BG_DARK).pack(side="left", pady=10)

        self._status_dot = tk.Label(topbar, text="●", font=("Helvetica", 14),
                                    fg=ERROR_CLR, bg=BG_DARK)
        self._status_dot.pack(side="right", padx=6)
        self._status_lbl = tk.Label(topbar, text="No model loaded",
                                    font=("Helvetica", 10), fg=TEXT_SEC, bg=BG_DARK)
        self._status_lbl.pack(side="right", padx=(0, 4))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=16, pady=12)

        left = tk.Frame(main, bg=BG_DARK, width=260)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        self._build_controls(left)

        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)
        self._build_canvas_area(right)

        log_frame = tk.Frame(self, bg=BG_PANEL, height=80)
        log_frame.pack(fill="x", side="bottom")
        log_frame.pack_propagate(False)
        tk.Label(log_frame, text="LOG", font=("Courier New", 9, "bold"),
                 fg=TEXT_SEC, bg=BG_PANEL).pack(side="left", padx=10, pady=(6, 0), anchor="n")
        self._log_var = tk.StringVar(value="")
        tk.Label(log_frame, textvariable=self._log_var, font=("Courier New", 10),
                 fg=SUCCESS, bg=BG_PANEL, justify="left",
                 wraplength=820).pack(side="left", padx=4, pady=8)

    def _build_controls(self, parent):
        def section(text):
            tk.Label(parent, text=text.upper(), font=("Helvetica", 9, "bold"),
                     fg=TEXT_SEC, bg=BG_DARK).pack(anchor="w", pady=(14, 4))
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(0, 8))

        section("01  Model")
        tk.Entry(parent, textvariable=self.model_path, font=("Courier New", 9),
                 bg=BG_CARD, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(fill="x", ipady=6, pady=(0, 6))
        tk.Label(parent, text="Accepts .keras or SavedModel folder",
                 font=("Helvetica", 8), fg=TEXT_SEC, bg=BG_DARK).pack(anchor="w")
        self._make_button(parent, "Browse model…",  self._browse_model).pack(fill="x", pady=(6, 2))
        self._make_button(parent, "Load model",      self._load_model, primary=True).pack(fill="x")
        self._prog = ttk.Progressbar(parent, mode="indeterminate", length=220)
        style = ttk.Style(); style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BG_CARD, background=ACCENT, bordercolor=BG_DARK)

        section("02  Input sketch")
        self._make_button(parent, "Open sketch image…", self._browse_input).pack(fill="x", pady=(0, 4))
        self._thumb_frame = tk.Frame(parent, bg=BG_CARD, width=220, height=130,
                                     highlightthickness=1, highlightbackground=BORDER)
        self._thumb_frame.pack(pady=(0, 4))
        self._thumb_frame.pack_propagate(False)
        self._thumb_lbl = tk.Label(self._thumb_frame, text="no image",
                                   font=("Helvetica", 9), fg=TEXT_SEC, bg=BG_CARD)
        self._thumb_lbl.place(relx=0.5, rely=0.5, anchor="center")

        section("03  Generate")
        self._run_btn = self._make_button(parent, "▶  Run inference", self._run_inference, primary=True)
        self._run_btn.pack(fill="x", pady=(0, 4))
        self._run_btn.configure(state="disabled")
        self._save_btn = self._make_button(parent, "Save output…", self._save_output)
        self._save_btn.pack(fill="x")
        self._save_btn.configure(state="disabled")

    def _build_canvas_area(self, parent):
        header = tk.Frame(parent, bg=BG_DARK)
        header.pack(fill="x", pady=(0, 8))
        for label in ["Input Sketch", "Generated Photo"]:
            tk.Label(header, text=label, font=("Helvetica", 11, "bold"),
                     fg=TEXT_PRI, bg=BG_DARK).pack(side="left", expand=True)

        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill="both", expand=True)
#works
        def make_card(par):
            card = tk.Frame(par, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
            card.pack(side="left", fill="both", expand=True, padx=6)
            lbl = tk.Label(card, text="—", fg=TEXT_SEC, bg=BG_CARD, font=("Helvetica", 10))
            lbl.pack(expand=True)
            return card, lbl

        self._in_card,  self._in_lbl  = make_card(row)
        self._out_card, self._out_lbl = make_card(row)

    def _make_button(self, parent, text, cmd, primary=False):
        fg  = BG_DARK  if primary else TEXT_PRI
        bg  = ACCENT   if primary else BG_CARD
        hbg = BTN_HOVER if primary else BORDER
        btn = tk.Label(parent, text=text,
                       font=("Helvetica", 10, "bold" if primary else "normal"),
                       fg=fg, bg=bg, cursor="hand2", pady=9, relief="flat")
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>",    lambda e: btn.configure(bg=hbg))
        btn.bind("<Leave>",    lambda e: btn.configure(bg=bg))
        return btn

    # ── Helpers ───────────────────────────────────────────────────────────
    def _log(self, msg):
        self._log_var.set(f"›  {msg}")
        self.update_idletasks()

    def _set_status(self, ready):
        self._status_dot.configure(fg=SUCCESS if ready else ERROR_CLR)
        self._status_lbl.configure(text="Model ready" if ready else "No model loaded")

    def _show_in_canvas(self, card, lbl, pil_img):
        card.update_idletasks()
        w = max(card.winfo_width() - 16, 256)
        h = max(card.winfo_height() - 16, 256)
        display = pil_img.copy()
        display.thumbnail((w, h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(display)
        lbl.configure(image=tk_img, text="", bg=BG_CARD)
        lbl.image = tk_img

    # ── Actions ───────────────────────────────────────────────────────────
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("Keras model", "*.keras"), ("All files", "*.*")])
        if path:
            self.model_path.set(path)

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select sketch image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")])
        if path:
            self.input_path.set(path)
            self._load_input_preview(path)

    def _load_input_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            self._input_pil = img
            thumb = img.copy()
            thumb.thumbnail((218, 126), Image.LANCZOS)
            tk_thumb = ImageTk.PhotoImage(thumb)
            self._thumb_lbl.configure(image=tk_thumb, text="")
            self._thumb_lbl.image = tk_thumb
            self._show_in_canvas(self._in_card, self._in_lbl, img)
            self._log(f"Loaded: {Path(path).name}  ({img.width}×{img.height})")
            if self._model_loaded:
                self._run_btn.configure(state="normal")
        except Exception as exc:
            self._log(f"Error loading image: {exc}")

    def _load_model(self):
        path = self.model_path.get().strip()
        if not path:
            messagebox.showwarning("No path", "Enter or browse for a model path first.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"Path does not exist:\n{path}")
            return
        self._log("Loading model…")
        self._prog.pack(fill="x", pady=4)
        self._prog.start(12)

        def _load():
            global tf, generator
            try:
                import tensorflow as _tf
                tf = _tf
                tf.get_logger().setLevel("ERROR")
                generator = tf.keras.models.load_model(path, compile=False)
                self.after(0, self._on_model_loaded)
            except Exception as exc:
                self.after(0, lambda: self._on_model_error(str(exc)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self):
        self._prog.stop(); self._prog.pack_forget()
        self._model_loaded = True
        self._set_status(True)
        self._log(f"Model loaded  ·  output shape: {generator.output_shape}")
        if self._input_pil:
            self._run_btn.configure(state="normal")

    def _on_model_error(self, msg):
        self._prog.stop(); self._prog.pack_forget()
        self._log(f"Model load failed: {msg}")
        messagebox.showerror("Load error", msg)

    def _run_inference(self):
        if not self._model_loaded or self._input_pil is None:
            return
        self._log("Running inference…")
        self._run_btn.configure(state="disabled")

        def _infer():
            try:
                img = self._input_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGB")
                arr = (np.array(img, dtype=np.float32) / 127.5) - 1.0
                arr = np.expand_dims(arr, 0)
                pred = generator(arr, training=False)
                out  = np.clip(((pred[0].numpy() + 1.0) / 2.0) * 255, 0, 255).astype(np.uint8)
                result = Image.fromarray(out)
                self.after(0, lambda: self._on_inference_done(result))
            except Exception as exc:
                self.after(0, lambda: self._on_inference_error(str(exc)))

        threading.Thread(target=_infer, daemon=True).start()

    def _on_inference_done(self, result_pil):
        self.output_image = result_pil
        self._show_in_canvas(self._out_card, self._out_lbl, result_pil)
        self._log("Done. Output rendered.")
        self._run_btn.configure(state="normal")
        self._save_btn.configure(state="normal")

    def _on_inference_error(self, msg):
        self._log(f"Inference error: {msg}")
        messagebox.showerror("Inference error", msg)
        self._run_btn.configure(state="normal")

    def _save_output(self):
        if self.output_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if path:
            self.output_image.save(path)
            self._log(f"Saved → {Path(path).name}")


if __name__ == "__main__":
    Pix2PixApp().mainloop()
# inference skeleton
# full inference UI
# UI polish
# thumbnail fix
# clean final pass
# inference skeleton
# full inference UI
# UI polish
# thumbnail fix
# clean final pass
# inference skeleton
# full inference UI
# UI polish
# thumbnail fix
# clean final pass
