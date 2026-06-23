from __future__ import annotations

import argparse
import io
import re
import shutil
from pathlib import Path

from lxml import etree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_SVG = REPO_ROOT / "Figures" / "figure_5_cema_tcn_endpoint_estimator.svg"
DEFAULT_TEMPLATE = OUT_SVG
SOURCE_FIGURES = REPO_ROOT / "Figures"
SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}
RECT_PATH_RE = re.compile(
    r"^M(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?) "
    r"(-?\d+(?:\.\d+)?) \2 \3 (-?\d+(?:\.\d+)?) \1 \4Z$"
)
TRANSLATE_RE = re.compile(r"translate\((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.default": "it",
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def strip_metadata(svg_root: etree._Element) -> None:
    for node in svg_root.findall(f"{{{SVG_NS}}}metadata"):
        svg_root.remove(node)
    for node in svg_root.xpath(".//svg:text[@fill-opacity='0']", namespaces=NS):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)


def make_nested_svg(x: float, y: float, width: float, height: float, svg_root: etree._Element) -> etree._Element:
    strip_metadata(svg_root)
    nested = etree.Element(f"{{{SVG_NS}}}svg")
    nested.set("x", f"{x:g}")
    nested.set("y", f"{y:g}")
    nested.set("width", f"{width:g}")
    nested.set("height", f"{height:g}")
    nested.set("viewBox", f"0 0 {width:g} {height:g}")
    nested.set("overflow", "visible")
    for child in list(svg_root):
        nested.append(child)
    return nested


def build_math_svg(formula: str, width: float, height: float, fontsize: float) -> etree._Element:
    set_style()
    fig = plt.figure(figsize=(width / 72.0, height / 72.0), dpi=72)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, formula, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)
    return etree.fromstring(buffer.getvalue())


def parse_cell_center(path_d: str) -> tuple[float, float] | None:
    match = RECT_PATH_RE.match(path_d)
    if not match:
        return None

    x1, y1, x2, y2 = (float(value) for value in match.groups())
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if not (20.0 <= width <= 36.0 and 20.0 <= height <= 36.0):
        return None
    if not (0.0 <= min(x1, x2) <= 580.0 and 0.0 <= min(y1, y2) <= 515.0):
        return None

    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dot_position(text_node: etree._Element) -> tuple[float, float] | None:
    transform = text_node.get("transform", "")
    match = TRANSLATE_RE.search(transform)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def make_dot(cx: float, cy: float, radius: float = 3.2) -> etree._Element:
    dot = etree.Element(f"{{{SVG_NS}}}circle")
    dot.set("cx", f"{cx:.3f}")
    dot.set("cy", f"{cy:.3f}")
    dot.set("r", f"{radius:g}")
    dot.set("fill", "#000000")
    return dot


def center_panel_a_dots(main_group: etree._Element) -> None:
    cell_centers = [
        center
        for path in main_group.findall(f"{{{SVG_NS}}}path")
        if (center := parse_cell_center(path.get("d", ""))) is not None
    ]
    if not cell_centers:
        raise RuntimeError("No panel (a) cell centers were found.")

    dot_text_nodes = main_group.xpath(".//svg:text[text()='●']", namespaces=NS)
    for text_node in dot_text_nodes:
        position = dot_position(text_node)
        parent = text_node.getparent()
        if position is None or parent is None or parent.getparent() is not main_group:
            continue

        tx, ty = position
        cx, cy = min(cell_centers, key=lambda center: (center[0] - tx) ** 2 + (center[1] - ty) ** 2)
        insert_idx = list(main_group).index(parent)
        main_group.remove(parent)
        main_group.insert(insert_idx, make_dot(cx, cy))


def build_silu_plot_svg(width: float, height: float) -> etree._Element:
    set_style()
    x = np.linspace(-6.0, 6.0, 600)
    y = x / (1.0 + np.exp(-x))

    fig = plt.figure(figsize=(width / 72.0, height / 72.0), dpi=72)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.13, 0.17, 0.84, 0.78])
    ax.plot(x, y, color="#111111", linewidth=1.6)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-0.6, 6.2)
    ax.set_xticks(np.arange(-6, 7, 2))
    ax.set_yticks([-0.5, 0.0, 2.0, 4.0, 6.0])
    ax.grid(True, linestyle="--", linewidth=0.55, color="#BFBFBF")
    ax.set_xlabel(r"$x$", fontsize=11, fontweight="bold", labelpad=1.0)
    ax.set_ylabel(r"$SiLU(x)$", fontsize=11, fontweight="bold", labelpad=2.0)
    ax.tick_params(axis="both", labelsize=8.5, width=0.9, length=3.2, pad=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#111111")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)
    return etree.fromstring(buffer.getvalue())


def add_text(x: float, y: float, text: str, *, fontsize: float = 18.6667) -> etree._Element:
    node = etree.Element(f"{{{SVG_NS}}}text")
    node.set("font-family", "Times New Roman,Times New Roman_MSFontService,sans-serif")
    node.set("font-weight", "700")
    node.set("font-size", f"{fontsize:g}")
    node.set("transform", f"translate({x:g} {y:g})")
    node.text = text
    return node


def replace_b_panel(template: Path, out_svg: Path) -> None:
    root = etree.fromstring(template.read_bytes())
    main_group = root.xpath("./svg:g[@clip-path='url(#clip0)']", namespaces=NS)[0]

    original_nodes = list(main_group)
    formula_nodes = [original_nodes[idx] for idx in range(5, 18)]
    insert_idx = list(main_group).index(formula_nodes[0])
    for node in formula_nodes:
        if node.getparent() is main_group:
            main_group.remove(node)

    title = add_text(652, 40, "SiLU activation")
    formula = make_nested_svg(
        665,
        50,
        205,
        38,
        build_math_svg(r"$y=x\sigma(x)=\dfrac{x}{1+e^{-x}}$", 205, 38, 18),
    )
    plot = make_nested_svg(596, 82, 342, 226, build_silu_plot_svg(342, 226))
    main_group.insert(insert_idx, title)
    main_group.insert(insert_idx + 1, formula)
    main_group.insert(insert_idx + 2, plot)

    s_hat_nodes = [original_nodes[idx] for idx in (386, 387, 388, 389)]
    s_hat_insert_idx = list(main_group).index(s_hat_nodes[0])
    for node in s_hat_nodes:
        if node.getparent() is main_group:
            main_group.remove(node)
    s_hat = make_nested_svg(902, 424, 32, 30, build_math_svg(r"$\widehat{\mathbf{s}}_{\mathbf{t}}$", 32, 30, 18))
    main_group.insert(s_hat_insert_idx, s_hat)

    center_panel_a_dots(main_group)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_bytes(etree.tostring(root, xml_declaration=False, encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a vector SiLU activation plot to panel (b) of the slide SVG.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE.as_posix())
    parser.add_argument("--out-svg", default=OUT_SVG.as_posix())
    return parser.parse_args()


def copy_if_distinct(src: Path, dst: Path) -> None:
    if not src.exists():
        if dst.exists():
            return
        raise FileNotFoundError(f"Required figure source not found: {src}")
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    replace_b_panel(Path(args.template), Path(args.out_svg))
    out_svg = Path(args.out_svg)
    out_dir = out_svg.parent
    out_png = out_dir / "figure_5_cema_tcn_endpoint_estimator.png"
    out_pdf = out_dir / "figure_5_cema_tcn_endpoint_estimator.pdf"
    out_tif = out_dir / "figure_5_cema_tcn_endpoint_estimator.tif"
    src_png = SOURCE_FIGURES / "figure_5_cema_tcn_endpoint_estimator.png"
    src_pdf = SOURCE_FIGURES / "figure_5_cema_tcn_endpoint_estimator.pdf"
    copy_if_distinct(src_png, out_png)
    copy_if_distinct(src_pdf, out_pdf)
    Image.open(out_png).convert("RGB").save(out_tif, dpi=(600, 600))
    print(f"Wrote {Path(args.out_svg)}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
