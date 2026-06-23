from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

from lxml import etree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plot_figure7_corrected_voltage_behavior as fig7  # noqa: E402
from plot_figure7_voltage_current_abs_current_behavior import (  # noqa: E402
    REQUIRED_COLUMNS,
    selected_profile_record,
)


OUT_SVG = REPO_ROOT / "Figures" / "figure_4_cema_tcn_workflow.svg"
DEFAULT_TEMPLATE = OUT_SVG
SOURCE_FIGURES = REPO_ROOT / "Figures"
PLOT_BOX = {"x": 334.0, "y": 69.0, "width": 293.0, "height": 253.0}
NORMALIZATION_FORMULA_BOX = {"x": 405.0, "y": 632.0, "width": 150.0, "height": 48.0}
SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}


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
            "mathtext.default": "bf",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_zoom_frame():
    selected = selected_profile_record("DST", 25.0)
    raw_roots = fig7.candidate_raw_roots([])
    if not raw_roots:
        raise FileNotFoundError("No raw NMC roots found. Set G4_RAW_ROOTS or pass --raw-root.")
    fig7.REQUIRED_COLUMNS = REQUIRED_COLUMNS
    frame, raw_root = fig7.load_selected_feature_frame(selected, raw_roots)
    frame, xlabel = fig7.attach_time_axis(frame, raw_root, str(selected["file_name"]))
    start, end, _ = fig7.choose_zoom_interval(frame)
    return frame.iloc[start:end].copy()


def axis_rect(x: float, y: float, w: float, h: float) -> list[float]:
    return [x / PLOT_BOX["width"], 1.0 - (y + h) / PLOT_BOX["height"], w / PLOT_BOX["width"], h / PLOT_BOX["height"]]


def style_axis(ax: plt.Axes, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.04)
        ax.spines[spine].set_color("#000000")
    ax.tick_params(axis="both", labelsize=7.0, width=1.04, length=4.5, pad=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")
        label.set_fontweight("bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.9, fontweight="bold", fontfamily="Times New Roman", labelpad=2.0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.9, fontweight="bold", fontfamily="Times New Roman", labelpad=2.0)


def build_cema_plot_svg() -> etree._Element:
    set_style()
    zoom = load_zoom_frame()
    x = zoom["_x"]

    fig = plt.figure(figsize=(PLOT_BOX["width"] / 72.0, PLOT_BOX["height"] / 72.0), dpi=72)
    fig.patch.set_alpha(0)
    axes = [
        fig.add_axes(axis_rect(41, 6, 193, 60)),
        fig.add_axes(axis_rect(41, 85, 193, 58)),
        fig.add_axes(axis_rect(41, 156, 193, 64)),
    ]
    for ax in axes:
        ax.set_facecolor("none")
        ax.set_xlim(8460, 9360)
        ax.set_xticks([8600, 8800, 9000, 9200])

    blue = "#2F6FAE"
    green = "#4C8C6B"
    orange = "#C27A2C"
    purple = "#6E5A9A"
    gray = "#4A4A4A"

    axes[0].plot(x, zoom["V_corr_raw"], color=blue, linewidth=1.23, label=r"$\mathbf{V}_{\mathbf{t}}^{\mathregular{corr}}$")
    axes[0].plot(x, zoom["V_corr_raw_ema50"], color=green, linewidth=1.23, label=r"$\mathbf{m}^{(50)}$")
    axes[0].plot(x, zoom["V_corr_raw_ema200"], color=orange, linewidth=1.23, label=r"$\mathbf{m}^{(200)}$")
    axes[0].plot(x, zoom["V_corr_raw_ema800"], color=purple, linewidth=1.23, label=r"$\mathbf{m}^{(800)}$")
    axes[0].set_ylim(3.475, 3.565)
    axes[0].set_yticks([3.50, 3.55])

    axes[1].plot(x, zoom["I_raw"], color=gray, linewidth=1.10, label=r"$\mathbf{I}_{\mathbf{t}}$")
    axes[1].plot(x, zoom["I_raw_ema50"], color=green, linewidth=1.23, label=r"$\mathbf{m}^{(50)}$")
    axes[1].plot(x, zoom["I_raw_ema200"], color=orange, linewidth=1.23, label=r"$\mathbf{m}^{(200)}$")
    axes[1].set_ylim(-4.3, 2.3)
    axes[1].set_yticks([-2.5, 0.0])

    axes[2].plot(x, zoom["absI_ema50"], color=green, linewidth=1.23, label=r"$\mathbf{m}^{(50)}$")
    axes[2].plot(x, zoom["absI_ema200"], color=orange, linewidth=1.23, label=r"$\mathbf{m}^{(200)}$")
    axes[2].set_ylim(0.3, 1.65)
    axes[2].set_yticks([0.5, 1.0, 1.5])

    style_axis(axes[0], "Voltage (V)")
    style_axis(axes[1], "Current (A)")
    style_axis(axes[2], "|Current| (A)", "Time (s)")
    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)

    for ax in axes:
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            ncol=1,
            handlelength=1.35,
            labelspacing=0.25,
            borderaxespad=0.1,
            prop={"family": "Times New Roman", "size": 7.4, "weight": "bold"},
        )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", transparent=True)
    plt.close(fig)
    return etree.fromstring(buffer.getvalue())


def strip_matplotlib_metadata(svg_root: etree._Element) -> None:
    for tag in ("metadata",):
        for node in svg_root.findall(f"{{{SVG_NS}}}{tag}"):
            svg_root.remove(node)
    remove_invisible_text(svg_root)


def remove_invisible_text(svg_root: etree._Element) -> None:
    for node in svg_root.xpath(".//svg:text[@fill-opacity='0']", namespaces=NS):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)


def build_math_svg(formula: str, width: float, height: float, fontsize: float) -> etree._Element:
    set_style()
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig = plt.figure(figsize=(width / 72.0, height / 72.0), dpi=72)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            formula,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontfamily="Times New Roman",
            fontweight="bold",
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", transparent=True)
        plt.close(fig)
    return etree.fromstring(buffer.getvalue())


def make_nested_svg(x: float, y: float, width: float, height: float, svg_root: etree._Element) -> etree._Element:
    strip_matplotlib_metadata(svg_root)
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


def build_normalization_formula_svg() -> etree._Element:
    return build_math_svg(
        r"$\mathbf{x}_{t,j}=\dfrac{\mathbf{x}_{t,j}-\boldsymbol{\mu}_{j}^{\mathrm{train}}}{\boldsymbol{\sigma}_{j}^{\mathrm{train}}}$",
        NORMALIZATION_FORMULA_BOX["width"],
        NORMALIZATION_FORMULA_BOX["height"],
        16,
    )


def text_content(node: etree._Element) -> str:
    return "".join(node.itertext())


def replace_nodes_with_math(main_group: etree._Element, original_nodes: list[etree._Element], replacement: dict[str, object]) -> None:
    nodes = [original_nodes[idx] for idx in replacement["indices"]]
    insert_idx = list(main_group).index(nodes[0])
    for node in nodes:
        if node.getparent() is main_group:
            main_group.remove(node)
    nested = make_nested_svg(
        float(replacement["x"]),
        float(replacement["y"]),
        float(replacement["width"]),
        float(replacement["height"]),
        build_math_svg(str(replacement["formula"]), float(replacement["width"]), float(replacement["height"]), float(replacement["fontsize"])),
    )
    main_group.insert(insert_idx, nested)


def replace_powerpoint_math_symbols(main_group: etree._Element) -> None:
    original_nodes = list(main_group)
    replacements = [
        {
            "indices": [17, 18],
            "x": 55,
            "y": 69,
            "width": 58,
            "height": 34,
            "fontsize": 24,
            "formula": r"$\mathbf{V}_{\mathbf{t}}$",
        },
        {
            "indices": [43, 44],
            "x": 214,
            "y": 69,
            "width": 52,
            "height": 34,
            "fontsize": 24,
            "formula": r"$\mathbf{I}_{\mathbf{t}}$",
        },
        {
            "indices": [22, 23, 24, 25, 26, 27],
            "x": 122,
            "y": 170,
            "width": 82,
            "height": 36,
            "fontsize": 24,
            "formula": r"$\widehat{\mathbf{R}}_{\mathbf{0}}(\mathbf{T})$",
        },
        {
            "indices": [30, 31],
            "x": 135,
            "y": 267,
            "width": 58,
            "height": 34,
            "fontsize": 24,
            "formula": r"$\mathbf{U}_{\mathbf{t}}$",
        },
        {
            "indices": [37, 38, 39],
            "x": 125,
            "y": 363,
            "width": 82,
            "height": 38,
            "fontsize": 24,
            "formula": r"$\mathbf{V}_{\mathbf{t}}^{\mathregular{corr}}$",
        },
        {
            "indices": [114, 115, 116, 117, 118],
            "x": 755,
            "y": 60,
            "width": 88,
            "height": 38,
            "fontsize": 24,
            "formula": r"$\widehat{\mathbf{SOC}}{}_{\mathbf{t}}$",
        },
        {
            "indices": [105, 106, 107, 108, 109, 110],
            "x": 752,
            "y": 505,
            "width": 92,
            "height": 32,
            "fontsize": 20,
            "formula": r"$\mathbf{x}_{t-49:t}$",
        },
        {
            "indices": [163, 164, 165, 166, 167, 168],
            "x": 106,
            "y": 512,
            "width": 120,
            "height": 34,
            "fontsize": 21,
            "formula": r"$\mathbf{V}_{\mathbf{t}}^{\mathregular{corr}}, \mathbf{I}_{\mathbf{t}}, \mathbf{T}$",
        },
    ]
    for replacement in sorted(replacements, key=lambda item: min(item["indices"]), reverse=True):
        replace_nodes_with_math(main_group, original_nodes, replacement)


def replace_normalization_formula(main_group: etree._Element) -> None:
    children = list(main_group)
    title_idx = None
    foot_idx = None
    for idx, child in enumerate(children):
        if child.tag == f"{{{SVG_NS}}}text" and text_content(child) == "Normalization":
            title_idx = idx
        if child.tag == f"{{{SVG_NS}}}text" and text_content(child).startswith("standard scaler;"):
            foot_idx = idx
            break
    if title_idx is None or foot_idx is None or foot_idx <= title_idx:
        raise ValueError("Could not locate the original Normalization formula block in the SVG template.")

    insert_idx = title_idx + 1
    for child in children[title_idx + 1 : foot_idx]:
        main_group.remove(child)

    formula_svg = build_normalization_formula_svg()
    nested = make_nested_svg(
        NORMALIZATION_FORMULA_BOX["x"],
        NORMALIZATION_FORMULA_BOX["y"],
        NORMALIZATION_FORMULA_BOX["width"],
        NORMALIZATION_FORMULA_BOX["height"],
        formula_svg,
    )
    main_group.insert(insert_idx, nested)


def svg_path(d: str, *, stroke: str | None = None, fill: str | None = None, width: float = 2.0) -> etree._Element:
    node = etree.Element(f"{{{SVG_NS}}}path")
    node.set("d", d)
    if stroke is not None:
        node.set("stroke", stroke)
        node.set("stroke-width", f"{width:g}")
        node.set("stroke-miterlimit", "8")
    if fill is not None:
        node.set("fill", fill)
    else:
        node.set("fill", "none")
    node.set("fill-rule", "evenodd")
    return node


def replace_tcn_residual_path(main_group: etree._Element) -> None:
    children = list(main_group)
    remove_nodes = []
    insert_idx = None
    residual_top_y = 230.0
    residual_bottom_y = 452.0
    for idx, child in enumerate(children):
        d = child.get("d", "")
        transform = child.get("transform", "")
        if d.startswith("M797.441 452.835 675 452"):
            remove_nodes.append(child)
            insert_idx = idx if insert_idx is None else insert_idx
            residual_top_y = 230.0
            residual_bottom_y = 452.0
        elif d.startswith("M797.441 418.835 675 418"):
            remove_nodes.append(child)
            insert_idx = idx if insert_idx is None else insert_idx
            residual_top_y = 195.0
            residual_bottom_y = 418.0
        elif d.startswith("M0 0 0.000104987 223.653") and "676 452.653" in transform:
            remove_nodes.append(child)
            insert_idx = idx if insert_idx is None else insert_idx
        elif d.startswith("M0 0 0.000104987 223.653") and "676 418.653" in transform:
            remove_nodes.append(child)
            insert_idx = idx if insert_idx is None else insert_idx
        elif d.startswith("M676 229 780.136 229"):
            remove_nodes.append(child)
            insert_idx = idx if insert_idx is None else insert_idx
        elif d.startswith("M676 194 780.136 194"):
            remove_nodes.append(child)
            insert_idx = idx if insert_idx is None else insert_idx

    if not remove_nodes:
        return
    for node in remove_nodes:
        main_group.remove(node)

    if insert_idx is None:
        insert_idx = len(main_group)
    # Make the residual branch visibly continuous: input branch -> vertical skip path -> plus node.
    replacements = [
        svg_path(f"M799 {residual_bottom_y:g} L676 {residual_bottom_y:g} L676 {residual_top_y:g}", stroke="#262626", width=2.0),
        svg_path(f"M676 {residual_top_y:g} L786.8 {residual_top_y:g}", stroke="#262626", width=2.0),
        svg_path(f"M778.8 {residual_top_y - 4:g} L786.8 {residual_top_y:g} L778.8 {residual_top_y + 4:g} Z", fill="#262626"),
    ]
    for offset, node in enumerate(replacements):
        main_group.insert(insert_idx + offset, node)


def replace_cema_group(template: Path, out_svg: Path) -> None:
    root = etree.fromstring(template.read_bytes())
    plot_svg = build_cema_plot_svg()
    strip_matplotlib_metadata(plot_svg)

    main_group = root.xpath("./svg:g[@clip-path='url(#clip0)']", namespaces=NS)[0]
    replace_powerpoint_math_symbols(main_group)
    replace_normalization_formula(main_group)
    replace_tcn_residual_path(main_group)
    old_groups = main_group.xpath("./svg:g[@clip-path='url(#clip1)']", namespaces=NS)
    if not old_groups:
        raise ValueError("Could not find the PowerPoint Causal-EMA plot group using clipPath #clip1.")
    old_group = old_groups[0]
    insert_index = list(main_group).index(old_group)
    main_group.remove(old_group)

    nested = etree.Element(f"{{{SVG_NS}}}svg")
    nested.set("x", f"{PLOT_BOX['x']:g}")
    nested.set("y", f"{PLOT_BOX['y']:g}")
    nested.set("width", f"{PLOT_BOX['width']:g}")
    nested.set("height", f"{PLOT_BOX['height']:g}")
    nested.set("viewBox", f"0 0 {PLOT_BOX['width']:g} {PLOT_BOX['height']:g}")
    nested.set("overflow", "visible")
    for child in list(plot_svg):
        nested.append(child)

    main_group.insert(insert_index, nested)
    remove_invisible_text(root)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_bytes(etree.tostring(root, xml_declaration=False, encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use the PowerPoint SVG as a vector template and replace only the Causal-EMA plot.")
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
    replace_cema_group(Path(args.template), Path(args.out_svg))
    out_svg = Path(args.out_svg)
    out_dir = out_svg.parent
    out_png = out_dir / "figure_4_cema_tcn_workflow.png"
    out_pdf = out_dir / "figure_4_cema_tcn_workflow.pdf"
    out_tif = out_dir / "figure_4_cema_tcn_workflow.tif"
    src_png = SOURCE_FIGURES / "figure_4_cema_tcn_workflow.png"
    src_pdf = SOURCE_FIGURES / "figure_4_cema_tcn_workflow.pdf"
    copy_if_distinct(src_png, out_png)
    copy_if_distinct(src_pdf, out_pdf)
    Image.open(out_png).convert("RGB").save(out_tif, dpi=(600, 600))
    print(f"Wrote {Path(args.out_svg)}")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
