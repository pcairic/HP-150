#!/usr/bin/env python3
import argparse, sys
import numpy as np
from PIL import Image, ImageFilter

# Authentic phosphor green on black
DEFAULT_INK = "#00FFB4"
DEFAULT_BG  = "#000000"

GLYPH_W, GLYPH_H = 7, 16
ADV_W = GLYPH_W + 1  # 8-pixel advance per cell
DEFAULT_MARGIN_RATIO = 0.05

def hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

class Row:
    __slots__ = ("bits", "phase")
    def __init__(self, bits, phase):
        # bits: left->right columns; 0 = ink (lit), 1 = background
        self.bits = bits
        self.phase = phase  # 0/1 for text rows; 0 for graphics rows

def is_graphics(ch: int) -> bool:
    # HP-150 graphics set is 256..383 inclusive
    return 256 <= ch <= 383

def is_text_with_phase(ch: int) -> bool:
    # Text glyphs that use MSB phase: 0..255 and 384..511
    return (0 <= ch <= 255) or (384 <= ch <= 511)

def extract_glyphs(byte_data, glyph_height=16, glyph_count=512,
                   start_offset=0, stride=16, baseline_shift=0):
    """
    Mapping per manual and your clarification:
    - Text glyphs (0..255, 384..511): MSB (bit 7) = phase; bits 6..0 = 7 columns (left->right), 0=ink, 1=bg.
    - Graphics glyphs (256..383): bits 7..0 = 8 columns (left->right), 0=ink, 1=bg; no phase.
    """
    glyphs = []
    for g in range(glyph_count):
        rows = []
        base = start_offset + g * stride
        for r in range(glyph_height):
            idx = base + r + baseline_shift
            if idx < base or idx >= base + stride:
                # pad with background if out of range
                rows.append(Row([1]*GLYPH_W, 0))
                continue
            b = byte_data[idx]
            if is_graphics(g):
                # Graphics: 8 columns, no phase
                bits8 = [ (b >> (7 - i)) & 1 for i in range(8) ]
                rows.append(Row(bits8, 0))
            else:
                # Text: MSB = phase; bits6..0 = 7 columns
                phase = (b >> 7) & 1
                bits7 = [ (b >> (6 - i)) & 1 for i in range(GLYPH_W) ]
                rows.append(Row(bits7, phase))
        glyphs.append(rows)
    return glyphs

def render_supersampled_grid(glyphs, cols, rows, ink_color, bg_color,
                             ssx=12, ssy=12):
    """
    Supersampled renderer:
    - Text (0..255, 384..511): 7 columns + spacing set once per cell; apply half-pixel row shift when phase=1.
    - Graphics (256..383): full 8 columns, no phase, no spacing; ensures box-drawing connects.
    Polarity: 0=ink, 1=background.
    """
    grid_w_ss = cols * ADV_W * ssx
    grid_h_ss = rows * GLYPH_H * ssy
    arr = np.zeros((grid_h_ss, grid_w_ss, 3), dtype=np.uint8)
    arr[:] = bg_color

    ink = np.array(ink_color, dtype=np.uint8)
    bg = np.array(bg_color, dtype=np.uint8)
    half = ssx // 2

    idx = 0
    for row in range(rows):
        for col in range(cols):
            ch = idx % 512
            idx += 1
            glyph_rows = glyphs[ch]

            if is_graphics(ch):
                glyph_width = ADV_W       # draw all 8 columns
                advance = ADV_W
                use_phase = False
            else:
                glyph_width = GLYPH_W     # draw 7 columns
                advance = ADV_W           # cell width 8 columns
                use_phase = True

            # Supersampled bounds for this cell’s drawable area
            cell_y0 = row * GLYPH_H * ssy
            cell_y1 = cell_y0 + GLYPH_H * ssy
            cell_x0 = col * advance * ssx
            cell_x1 = cell_x0 + glyph_width * ssx  # drawable area excludes spacing for text

            # Paint spacing column ONCE per cell (text only)
            if not is_graphics(ch):
                xs = cell_x0 + GLYPH_W * ssx
                xe = xs + ssx
                arr[cell_y0:cell_y1, xs:xe] = bg

            # Draw all 16 ROM rows exactly (no baseline shift)
            for gy in range(GLYPH_H):
                row_obj = glyph_rows[gy]
                row_bits = row_obj.bits
                phase_shift = (half if (use_phase and row_obj.phase and half > 0) else 0)

                ys = cell_y0 + gy * ssy
                ye = ys + ssy

                # Draw glyph columns (0=ink)
                for gx in range(glyph_width):
                    bit = row_bits[gx] if gx < len(row_bits) else 1  # default background
                    if bit == 0:  # 0 = ink
                        xs = cell_x0 + gx * ssx + phase_shift
                        xe = xs + ssx
                        # Clamp strictly inside drawable area to prevent left-edge bleed
                        x0c = max(cell_x0, min(cell_x1, xs))
                        x1c = max(cell_x0, min(cell_x1, xe))
                        if x1c > x0c:
                            arr[ys:ye, x0c:x1c] = ink

    return arr

def float_downscale(img_ss, target_w, target_h, blur_radius=0.35):
    img_prep = img_ss.filter(ImageFilter.GaussianBlur(radius=blur_radius)) if blur_radius > 0 else img_ss
    return img_prep.resize((target_w, target_h), resample=Image.LANCZOS)

def compose_display(glyphs, cols, rows,
                    ink_color, bg_color,
                    display_w=2048, display_h=1536,
                    margin_ratio=DEFAULT_MARGIN_RATIO,
                    ssx=12, ssy=12,
                    blur_radius=0.35):
    # Active area with margins
    active_w = int(round(display_w * (1.0 - 2.0 * margin_ratio)))
    active_h = int(round(display_h * (1.0 - 2.0 * margin_ratio)))

    # Floating scales per axis
    scale_x = active_w / float(cols * ADV_W)
    scale_y = active_h / float(rows * GLYPH_H)

    arr_ss = render_supersampled_grid(glyphs, cols, rows, ink_color, bg_color,
                                      ssx=ssx, ssy=ssy)
    img_ss = Image.fromarray(arr_ss, "RGB")

    target_w = max(1, int(round(cols * ADV_W * scale_x)))
    target_h = max(1, int(round(rows * GLYPH_H * scale_y)))

    active_img = float_downscale(img_ss, target_w, target_h, blur_radius=blur_radius)

    comp = Image.new("RGB", (display_w, display_h), color=bg_color)
    x_offset = (display_w - target_w) // 2
    y_offset = (display_h - target_h) // 2
    comp.paste(active_img, (x_offset, y_offset))

    print(f"Margins: {int(margin_ratio*100)}% each side | Active area: {active_w}x{active_h}")
    print(f"Float scales: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    print(f"Supersample: ssx={ssx}, ssy={ssy} | Text phase (MSB) on 0..255 & 384..511; Graphics 256..383 full 8 cols")
    print(f"Blur radius={blur_radius}")
    print(f"Placed active {target_w}x{target_h} at offsets ({x_offset},{y_offset}) on {display_w}x{display_h}")

    return comp

def main():
    ap = argparse.ArgumentParser(
        description="HP-150 renderer with correct graphics range (256..383), MSB per-row phase on text (0..255, 384..511), supersampling, margins, and blur."
    )
    ap.add_argument("bin", help="ROM binary file")
    ap.add_argument("--cols", type=int, default=80, help="Columns (80 or 132)")
    ap.add_argument("--rows", type=int, default=27, help="Rows (default 27)")
    ap.add_argument("--ink", default=DEFAULT_INK, help="Ink color (#RRGGBB)")
    ap.add_argument("--bg", default=DEFAULT_BG, help="Background color (#RRGGBB)")
    ap.add_argument("--display-w", type=int, default=2048, help="Display width (pixels)")
    ap.add_argument("--display-h", type=int, default=1536, help="Display height (pixels)")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN_RATIO, help="Margin ratio (0.05 = 5%%)")
    ap.add_argument("--ssx", type=int, default=12, help="Supersample factor X")
    ap.add_argument("--ssy", type=int, default=12, help="Supersample factor Y")
    ap.add_argument("--blur", type=float, default=0.35, help="Gaussian blur radius before downscale")
    args, _ = ap.parse_known_args()

    with open(args.bin, "rb") as f:
        data = f.read()

    glyphs = extract_glyphs(data, glyph_count=512, baseline_shift=0)
    ink = hex_to_rgb(args.ink)
    bg = hex_to_rgb(args.bg)

    comp = compose_display(
        glyphs, args.cols, args.rows,
        ink_color=ink, bg_color=bg,
        display_w=args.display_w, display_h=args.display_h,
        margin_ratio=args.margin,
        ssx=args.ssx, ssy=args.ssy,
        blur_radius=args.blur
    )

    comp.show()

if __name__ == "__main__":
    main()