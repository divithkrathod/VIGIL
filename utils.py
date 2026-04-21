"""UI drawing helpers for VIGIL — video feed + grey canvas + side panel."""

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

import config

# Fonts
_F_VIGIL  = cv2.FONT_HERSHEY_DUPLEX    # VIGIL title — slightly heavier
_F_MONO   = cv2.FONT_HERSHEY_PLAIN     # closest built-in to monospace (Consolas feel)
_F_UI     = cv2.FONT_HERSHEY_SIMPLEX   # general UI fallback


# ── Canvas / background ───────────────────────────────────────────────────────

def build_canvas(feed_h: int, feed_w: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    P        = config.CANVAS_PADDING
    canvas_w = P + feed_w + P + config.SIDE_PANEL_WIDTH
    canvas_h = P + feed_h + P
    canvas   = np.full((canvas_h, canvas_w, 3), config.BG_GREY_BGR, dtype=np.uint8)
    return canvas, (P, P)


def paste_feed(canvas: np.ndarray, frame: np.ndarray, origin: Tuple[int, int]) -> None:
    x, y = origin
    h, w = frame.shape[:2]
    canvas[y:y + h, x:x + w] = frame


# ── Rounded rectangle helper ──────────────────────────────────────────────────

def _rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        # filled
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, -1)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, -1)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, -1)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, -1)
    else:
        # outline only
        cv2.line(img,  (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img,  (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img,  (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img,  (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90,  color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90,  color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90,  color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90,  color, thickness)


# ── Stat box helper ───────────────────────────────────────────────────────────

def _stat_box(canvas, label, value, px, y, pw, val_color):
    """Dark rounded box with a small label and larger value inside."""
    bx1 = px + 14
    bx2 = px + pw - 14
    by1 = y
    by2 = y + 46
    _rounded_rect(canvas, bx1, by1, bx2, by2, 6, (22, 22, 22))
    # label — small, grey, mono
    cv2.putText(canvas, label, (bx1 + 10, by1 + 14),
                _F_MONO, 0.75, (110, 110, 110), 1, cv2.LINE_AA)
    # value — larger, coloured
    cv2.putText(canvas, value, (bx1 + 10, by1 + 36),
                _F_MONO, 1.0, val_color, 1, cv2.LINE_AA)


# ── Segmented progress bar ────────────────────────────────────────────────────

def _segmented_bar(
    canvas, x1, y1, x2, y2,
    secs_until_alert,       # counts DOWN from DROWSY_SECONDS_BEFORE_ALERT
    cancel_secs_remaining,  # counts DOWN from CANCEL_WINDOW_SECONDS
    is_alerting,
    in_cancel_window,
    contact_sent,
):
    """
    Three-segment pipeline bar (Rapido-style):
      [  Checking (5s)  |  Cancel (5s)  |  Sent  ]
    Each segment fills independently as its phase progresses.
    Blue throughout — red only on the Sent segment.
    """
    BH   = y2 - y1          # bar height
    GAP  = 4                 # gap between segments
    r    = BH // 2           # corner radius

    total_w  = x2 - x1
    seg_w    = (total_w - GAP * 2) // 3   # each segment same width

    seg_starts = [x1, x1 + seg_w + GAP, x1 + (seg_w + GAP) * 2]
    seg_ends   = [seg_starts[0] + seg_w,
                  seg_starts[1] + seg_w,
                  x2]

    SEG_COLORS = [
        (200, 140, 40),   # seg 0 — checking   — blue
        (200, 140, 40),   # seg 1 — cancel      — blue
        (76,  76,  255),  # seg 2 — sent        — red
    ]
    TRACK_COLOR = (45, 45, 45)

    # ── compute fill fractions ────────────────────────────────────────────
    check_total = config.DROWSY_SECONDS_BEFORE_ALERT   # 5s
    cancel_total = config.CANCEL_WINDOW_SECONDS         # 5s

    if contact_sent:
        fracs = [1.0, 1.0, 1.0]
    elif in_cancel_window:
        elapsed_cancel = cancel_total - cancel_secs_remaining
        fracs = [1.0, min(1.0, elapsed_cancel / cancel_total), 0.0]
    elif is_alerting:
        fracs = [1.0, 0.0, 0.0]
    elif 0.0 < secs_until_alert < check_total:
        # actively checking — fill proportionally as time elapses
        elapsed_check = check_total - secs_until_alert
        fracs = [min(1.0, elapsed_check / check_total), 0.0, 0.0]
    else:
        # idle — eyes open or no face — keep bar empty
        fracs = [0.0, 0.0, 0.0]

    # ── draw each segment ─────────────────────────────────────────────────
    for i in range(3):
        sx1, sx2 = seg_starts[i], seg_ends[i]
        sw = sx2 - sx1

        # track (empty background)
        _rounded_rect(canvas, sx1, y1, sx2, y2, r, TRACK_COLOR)

        # fill
        if fracs[i] > 0:
            fill_x2 = sx1 + max(int(sw * fracs[i]), r * 2)
            fill_x2 = min(fill_x2, sx2)
            _rounded_rect(canvas, sx1, y1, fill_x2, y2, r, SEG_COLORS[i])

    # ── segment labels below bar ──────────────────────────────────────────
    lbl_y = y2 + 13
    labels = ["CHECKING", "CANCEL", "SENT"]
    label_colors = [
        (180, 180, 180) if fracs[0] < 1.0 else (200, 140, 40),
        (180, 180, 180) if fracs[1] < 1.0 else (200, 140, 40),
        (180, 180, 180) if not contact_sent  else (76, 76, 255),
    ]
    for i in range(3):
        sx1, sx2 = seg_starts[i], seg_ends[i]
        (tw, _), _ = cv2.getTextSize(labels[i], _F_MONO, 0.65, 1)
        lx = sx1 + (sx2 - sx1 - tw) // 2
        cv2.putText(canvas, labels[i], (lx, lbl_y),
                    _F_MONO, 0.65, label_colors[i], 1, cv2.LINE_AA)


# ── Video-feed overlays ───────────────────────────────────────────────────────

def apply_dark_theme_overlay(frame: np.ndarray) -> np.ndarray:
    tint = np.full_like(frame, config.DARK_NAVY_BGR, dtype=np.uint8)
    return cv2.addWeighted(frame, 0.75, tint, 0.25, 0.0)


def draw_eye_landmarks(frame: np.ndarray, eye_points: np.ndarray) -> None:
    for x, y in eye_points.astype(int):
        cv2.circle(frame, (x, y), config.LANDMARK_RADIUS, config.LIME_GREEN_BGR, -1)


def draw_eye_bounding_box(frame: np.ndarray, eye_points: np.ndarray) -> None:
    x, y, w, h = cv2.boundingRect(eye_points.astype(int))
    cv2.rectangle(frame, (x, y), (x + w, y + h),
                  config.ELECTRIC_BLUE_BGR, config.BOX_THICKNESS)


def draw_alert_banner(frame: np.ndarray, text: str = "DROWSINESS ALERT!") -> None:
    overlay = frame.copy()
    fh, fw  = frame.shape[:2]
    top     = int(fh * 0.10)
    bottom  = int(fh * 0.22)
    cv2.rectangle(overlay, (0, top), (fw, bottom), config.ALERT_RED_BGR, -1)
    cv2.addWeighted(overlay, config.ALERT_BANNER_ALPHA,
                    frame, 1 - config.ALERT_BANNER_ALPHA, 0, frame)
    _centered_text(frame, text, int((top + bottom) / 2 + 10),
                   config.WHITE_BGR, _F_UI, 0.85, 2)


def draw_feed_hud(frame: np.ndarray, ear: float,
                  frame_counter: int, is_alerting: bool) -> None:
    status = "ALERT" if is_alerting else "NORMAL"
    cv2.putText(frame, f"EAR: {ear:.3f}", (14, 28),
                _F_MONO, 0.9, config.CYAN_BGR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frames: {frame_counter}/{config.CONSECUTIVE_FRAMES}",
                (14, 50), _F_MONO, 0.85, config.CYAN_BGR, 1, cv2.LINE_AA)
    color = config.ALERT_RED_BGR if is_alerting else config.CYAN_BGR
    _centered_text(frame, status, 48, color, _F_UI,
                   config.STATUS_FONT_SCALE, config.STATUS_FONT_THICKNESS)


# ── Side panel ────────────────────────────────────────────────────────────────

def draw_side_panel(
    canvas: np.ndarray,
    feed_w: int,
    feed_h: int,
    ear: float,
    low_ear_frames: int,
    is_alerting: bool,
    in_cancel_window: bool,
    cancel_secs_remaining: float,
    contact_sent: bool,
    secs_until_alert: float,
) -> Tuple[int, int, int, int]:
    P  = config.CANVAS_PADDING
    px = P + feed_w + P
    pw = config.SIDE_PANEL_WIDTH
    ph = feed_h
    py = P

    # Panel background
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph), config.PANEL_DARK_BGR, -1)
    cv2.line(canvas, (px, py), (px, py + ph), config.PANEL_BORDER_BGR, 2)

    # ── VIGIL title ────────────────────────────────────────────────────────
    _centered_text(canvas, "VIGIL", py + 34, config.WHITE_BGR,
                   _F_VIGIL, 1.0, 2, x_offset=px, width=pw)
    _centered_text(canvas, "DROWSINESS MONITOR", py + 54, (100, 100, 100),
                   _F_MONO, 0.8, 1, x_offset=px, width=pw)

    _hdivider(canvas, px, py + 66, pw)

    # ── Status badge ───────────────────────────────────────────────────────
    status_text  = "ALERT" if is_alerting else "NORMAL"
    status_color = config.ALERT_RED_BGR if is_alerting else config.GREEN_OK_BGR
    bx1, bx2 = px + 20, px + pw - 20
    by1, by2 = py + 76, py + 108
    _rounded_rect(canvas, bx1, by1, bx2, by2, 8, status_color)
    _centered_text(canvas, status_text, by2 - 10, config.WHITE_BGR,
                   _F_UI, 0.60, 2, x_offset=px, width=pw)

    _hdivider(canvas, px, py + 118, pw)

    # ── Stat boxes ─────────────────────────────────────────────────────────
    sy = py + 128
    ear_color = (config.ALERT_RED_BGR if ear < config.EAR_THRESHOLD
                 else config.LIME_GREEN_BGR)
    _stat_box(canvas, "EAR VALUE",
              f"{ear:.4f}", px, sy, pw, ear_color)

    _stat_box(canvas, "THRESHOLD",
              f"{config.EAR_THRESHOLD:.2f}", px, sy + 56, pw, config.CYAN_BGR)

    _stat_box(canvas, "LOW-EAR FRAMES",
              f"{low_ear_frames} / {config.CONSECUTIVE_FRAMES}",
              px, sy + 112, pw, config.WHITE_BGR)

    # ── Segmented pipeline bar ─────────────────────────────────────────────
    bar_top = sy + 172
    bar_h   = 14
    bx1_bar = px + 14
    bx2_bar = px + pw - 14
    _segmented_bar(
        canvas, bx1_bar, bar_top, bx2_bar, bar_top + bar_h,
        secs_until_alert, cancel_secs_remaining,
        is_alerting, in_cancel_window, contact_sent,
    )

    # ── Contact sent notice ────────────────────────────────────────────────
    if contact_sent:
        _centered_text(canvas, "CONTACT ALERT SENT", py + ph - 100,
                       config.ALERT_RED_BGR, _F_MONO, 0.8, 1,
                       x_offset=px, width=pw)

    # ── Cancel button (rounded) ────────────────────────────────────────────
    btn_m  = config.CANCEL_BTN_MARGIN
    btn_h  = config.CANCEL_BTN_HEIGHT
    btn_x1 = px + btn_m
    btn_x2 = px + pw - btn_m
    btn_y2 = py + ph - btn_m
    btn_y1 = btn_y2 - btn_h

    _hdivider(canvas, px, btn_y1 - 12, pw)

    if in_cancel_window:
        _rounded_rect(canvas, btn_x1, btn_y1, btn_x2, btn_y2, 10,
                      config.ALERT_RED_BGR)
        _rounded_rect(canvas, btn_x1, btn_y1, btn_x2, btn_y2, 10,
                      config.WHITE_BGR, 2)
        mid_y = (btn_y1 + btn_y2) // 2
        _centered_text(canvas, "CANCEL ALERT", mid_y - 8, config.WHITE_BGR,
                       _F_MONO, 0.85, 1, x_offset=px, width=pw)
        _centered_text(canvas, f"{cancel_secs_remaining:.1f}s", mid_y + 14,
                       config.WHITE_BGR, _F_MONO, 1.1, 1,
                       x_offset=px, width=pw)

    elif is_alerting:
        _rounded_rect(canvas, btn_x1, btn_y1, btn_x2, btn_y2, 10, (60, 60, 60))
        _centered_text(canvas, "CANCEL EXPIRED",
                       (btn_y1 + btn_y2) // 2 + 6, (110, 110, 110),
                       _F_MONO, 0.75, 1, x_offset=px, width=pw)
    else:
        _rounded_rect(canvas, btn_x1, btn_y1, btn_x2, btn_y2, 10, (48, 48, 48))
        _centered_text(canvas, "monitoring...",
                       (btn_y1 + btn_y2) // 2 + 6, (90, 90, 90),
                       _F_MONO, 0.80, 1, x_offset=px, width=pw)

    return (btn_x1, btn_y1, btn_x2, btn_y2)


# ── Drawing primitives ────────────────────────────────────────────────────────

def _centered_text(img, text, y, color, font, scale, thickness,
                   x_offset=0, width=None):
    """Draw horizontally centred text. Uses img width if width not given."""
    w = width if width is not None else img.shape[1]
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = x_offset + max((w - tw) // 2, 0)
    cv2.putText(img, text, (x, max(y, th + 2)),
                font, scale, color, thickness, cv2.LINE_AA)


def _hdivider(canvas, px, y, pw):
    cv2.line(canvas, (px + 12, y), (px + pw - 12, y),
             config.PANEL_BORDER_BGR, 1)