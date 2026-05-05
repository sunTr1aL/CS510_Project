#!/usr/bin/env python3
"""Generate paper figures and tables from local CS510 result artifacts.

The script intentionally reads the canonical Markdown/JSON files instead of
embedding a separate copy of the experimental results.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
CONSOLIDATED = ROOT / "refine-logs" / "CONSOLIDATED_METHOD_RESULTS_2026-04-30.md"
DIAGNOSTICS = ROOT / "refine-logs" / "R030_diagnostics_summary.json"
TOPVENUE = ROOT / "outputs" / "remote" / "rrb_topvenue_20260430_163958" / "analysis" / "R041_R045_results.json"


def clean_cell(text: str) -> str:
    text = text.strip().replace("**", "").replace("`", "")
    text = re.sub(r"<br\s*/?>", " ", text)
    return text


def to_float(text: str):
    text = clean_cell(text)
    if text in {"-", "", "baseline"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def parse_main_results() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    in_table = False
    for line in CONSOLIDATED.read_text().splitlines():
        if line.startswith("| Run | System | Relation |") and "Residual mag." in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and not line.startswith("|"):
            break
        if not in_table:
            continue
        cells = [clean_cell(c) for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        run = cells[0]
        if not run.startswith("R"):
            continue
        rows[run] = {
            "run": run,
            "system": cells[1],
            "relation": to_float(cells[2]),
            "delta_vs_r005": to_float(cells[3]),
            "coco_i2t": to_float(cells[4]),
            "imagenet": to_float(cells[5]),
            "residual_mag": to_float(cells[6]),
            "z0_cosine": to_float(cells[7]),
            "gate": cells[8] if len(cells) > 8 else "",
        }
    if not rows:
        raise RuntimeError(f"No main results parsed from {CONSOLIDATED}")
    return rows


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def summarize_group(rows: dict[str, dict], name: str, run_ids: list[str]) -> dict:
    vals = [rows[r] for r in run_ids]
    return {
        "name": name,
        "runs": run_ids,
        "relation_mean": sum(v["relation"] for v in vals) / len(vals),
        "relation_std": sample_std([v["relation"] for v in vals]),
        "coco_mean": sum(v["coco_i2t"] for v in vals) / len(vals),
        "coco_std": sample_std([v["coco_i2t"] for v in vals]),
        "imagenet_mean": sum(v["imagenet"] for v in vals) / len(vals),
        "imagenet_std": sample_std([v["imagenet"] for v in vals]),
    }


def fmt(x, digits: int = 4) -> str:
    return "--" if x is None else f"{x:.{digits}f}"


def technical_note(note: str) -> str:
    mapping = {
        "baseline": "Frozen OpenCLIP baseline",
        "safe baseline": "Plain adapter baseline",
        "collapse": "Catastrophic guardrail degradation",
        "pass": "Clears pilot guardrails",
        "pass; selected": "Selected configuration",
        "COCO fail": "COCO guardrail below threshold",
        "COCO near miss": "Near COCO threshold",
        "relation fail": "Relation gain below target",
        "guardrail-fragile": "Large retrieval/zero-shot degradation",
    }
    return mapping.get(note, note)


def pct_delta(a: float, b: float) -> str:
    return f"{a - b:+.4f}"


def tikz_point(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float, w: float, h: float):
    px = (x - xmin) / (xmax - xmin) * w
    py = (y - ymin) / (ymax - ymin) * h
    return px, py


def write_fig1(rows: dict[str, dict]) -> None:
    selected = [
        "R003",
        "R005",
        "R011",
        "R014",
        "R019",
        "R023",
        "R024",
        "R028",
        "R029",
        "R031",
        "R032",
        "R033",
        "R035",
        "R036",
        "R037",
        "R038",
        "R039",
        "R040",
        "R043",
    ]
    # R043 lives in the top-venue JSON, not the consolidated table.
    top = json.loads(TOPVENUE.read_text())
    rows = dict(rows)
    rows["R043"] = {
        "run": "R043",
        "system": "ROP ablation",
        "relation": top["R043_R044"]["R043"]["eval"]["accuracy"],
        "coco_i2t": top["R043_R044"]["R043"]["guardrail"]["coco_retrieval"]["image_to_text_r1"],
        "imagenet": top["R043_R044"]["R043"]["guardrail"]["imagenet_zeroshot"]["top1"],
        "residual_mag": top["R043_R044"]["R043"]["train"]["residual_relative_magnitude"],
        "z0_cosine": top["R043_R044"]["R043"]["train"]["z0_cosine_mean"],
        "gate": "relation fail",
    }
    xmin, xmax = 0.64, 0.81
    ymin, ymax = 0.34, 0.82
    w, h = 6.8, 4.0
    lines = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{tikzpicture}[font=\\footnotesize]",
        f"\\draw[->] (0,0) -- ({w + 0.25:.2f},0) node[right] {{Relation accuracy}};",
        f"\\draw[->] (0,0) -- (0,{h + 0.25:.2f}) node[above] {{COCO I2T R@1}};",
    ]
    for tick in [0.65, 0.70, 0.75, 0.80]:
        x, _ = tikz_point(tick, ymin, xmin, xmax, ymin, ymax, w, h)
        lines.append(f"\\draw ({x:.2f},0.04) -- ({x:.2f},-0.04) node[below] {{{tick:.2f}}};")
    for tick in [0.40, 0.50, 0.60, 0.70, 0.80]:
        _, y = tikz_point(xmin, tick, xmin, xmax, ymin, ymax, w, h)
        lines.append(f"\\draw (0.04,{y:.2f}) -- (-0.04,{y:.2f}) node[left] {{{tick:.2f}}};")
    x_r005, _ = tikz_point(rows["R005"]["relation"], ymin, xmin, xmax, ymin, ymax, w, h)
    _, y_gate = tikz_point(xmin, 0.70, xmin, xmax, ymin, ymax, w, h)
    lines.append(f"\\draw[densely dashed,gray] ({x_r005:.2f},0) -- ({x_r005:.2f},{h:.2f});")
    lines.append(f"\\draw[densely dashed,gray] (0,{y_gate:.2f}) -- ({w:.2f},{y_gate:.2f});")
    lines.append(f"\\node[gray,anchor=south west] at ({x_r005 + 0.05:.2f},0.05) {{R005 relation}};")
    lines.append(f"\\node[gray,anchor=south east] at ({w - 0.05:.2f},{y_gate + 0.05:.2f}) {{COCO gate}};")
    styles = {
        "baseline": ("black", "circle"),
        "adapter": ("gray", "circle"),
        "hard": ("red!75!black", "rectangle"),
        "rrb": ("blue!75!black", "circle"),
        "anchor20": ("cyan!60!black", "circle"),
        "noslot": ("orange!85!black", "diamond"),
        "rop": ("purple!75!black", "circle"),
    }

    def category(run: str):
        if run == "R003":
            return "baseline"
        if run in {"R005", "R024"}:
            return "adapter"
        if run in {"R011", "R014"}:
            return "hard"
        if run in {"R033", "R040"}:
            return "noslot"
        if run in {"R035", "R036", "R037", "R038", "R039"}:
            return "anchor20"
        if run == "R043":
            return "rop"
        return "rrb"

    label_runs = {"R003", "R005", "R011", "R028", "R033", "R038", "R043"}
    for run in selected:
        row = rows[run]
        x, y = tikz_point(row["relation"], row["coco_i2t"], xmin, xmax, ymin, ymax, w, h)
        color, shape = styles[category(run)]
        size = "2.3pt" if run == "R028" else "1.8pt"
        lines.append(f"\\node[{shape},draw={color},fill={color},inner sep={size}] ({run}) at ({x:.2f},{y:.2f}) {{}};")
        if run in label_runs:
            anchor = "west" if x < w * 0.75 else "east"
            dx = 0.08 if anchor == "west" else -0.08
            lines.append(f"\\node[anchor={anchor}] at ({x + dx:.2f},{y + 0.10:.2f}) {{{run}}};")
    frontier_runs = ["R024", "R005", "R038", "R023"]
    frontier_points = []
    for run in frontier_runs:
        row = rows[run]
        x, y = tikz_point(row["relation"], row["coco_i2t"], xmin, xmax, ymin, ymax, w, h)
        frontier_points.append(f"({x:.2f},{y:.2f})")
    lines.append("\\draw[dashed,thick,black!55] " + " -- ".join(frontier_points) + ";")
    lines.append("\\node[black!60,anchor=west] at (4.65,2.70) {evaluated frontier};")
    lines.extend(
        [
            f"\\node[anchor=west] at (0.15,{h + 0.05:.2f}) {{\\tikz\\node[circle,draw=blue!75!black,fill=blue!75!black,inner sep=1.5pt]{{}}; RRB}};",
            f"\\node[anchor=west] at (1.55,{h + 0.05:.2f}) {{\\tikz\\node[rectangle,draw=red!75!black,fill=red!75!black,inner sep=1.4pt]{{}}; hard token}};",
            f"\\node[anchor=west] at (3.25,{h + 0.05:.2f}) {{\\tikz\\node[diamond,draw=orange!85!black,fill=orange!85!black,inner sep=1.5pt]{{}}; no-channel}};",
            f"\\node[anchor=west] at (5.25,{h + 0.05:.2f}) {{\\tikz\\node[circle,draw=cyan!60!black,fill=cyan!60!black,inner sep=1.5pt]{{}}; anchor-20}};",
            "\\end{tikzpicture}",
        ]
    )
    (FIG_DIR / "fig1_relation_coco.tikz.tex").write_text("\n".join(lines) + "\n")


def write_fig2(groups: list[dict]) -> None:
    w, h = 6.8, 3.5
    rel_min, rel_max = 0.68, 0.79
    coco_min, coco_max = 0.65, 0.75
    lines = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{tikzpicture}[font=\\footnotesize]",
        f"\\draw[->] (0,0) -- (0,{h:.2f}) node[above] {{Relation}};",
        f"\\draw[->] (3.75,0) -- (3.75,{h:.2f}) node[above] {{COCO I2T}};",
    ]
    for tick in [0.70, 0.75]:
        y = (tick - rel_min) / (rel_max - rel_min) * h
        lines.append(f"\\draw (0.04,{y:.2f}) -- (-0.04,{y:.2f}) node[left] {{{tick:.2f}}};")
    for tick in [0.68, 0.72]:
        y = (tick - coco_min) / (coco_max - coco_min) * h
        lines.append(f"\\draw (3.79,{y:.2f}) -- (3.71,{y:.2f}) node[left] {{{tick:.2f}}};")
    colors = ["blue!65!black", "cyan!60!black", "teal!70!black"]
    labels = ["full", "anchor10", "anchor20"]
    for i, group in enumerate(groups):
        x1 = 0.7 + i * 0.85
        x2 = 4.45 + i * 0.85
        rel_y = (group["relation_mean"] - rel_min) / (rel_max - rel_min) * h
        rel_err = group["relation_std"] / (rel_max - rel_min) * h
        coco_y = (group["coco_mean"] - coco_min) / (coco_max - coco_min) * h
        coco_err = group["coco_std"] / (coco_max - coco_min) * h
        lines.append(f"\\draw[fill={colors[i]},draw={colors[i]}] ({x1 - 0.18:.2f},0) rectangle ({x1 + 0.18:.2f},{rel_y:.2f});")
        lines.append(f"\\draw[{colors[i]},thick] ({x1:.2f},{rel_y - rel_err:.2f}) -- ({x1:.2f},{rel_y + rel_err:.2f});")
        lines.append(f"\\node[rotate=35,anchor=east] at ({x1 + 0.15:.2f},-0.15) {{{labels[i]}}};")
        lines.append(f"\\node[above] at ({x1:.2f},{rel_y + rel_err + 0.05:.2f}) {{{group['relation_mean']:.3f}}};")
        lines.append(f"\\draw[fill={colors[i]},draw={colors[i]}] ({x2 - 0.18:.2f},0) rectangle ({x2 + 0.18:.2f},{coco_y:.2f});")
        lines.append(f"\\draw[{colors[i]},thick] ({x2:.2f},{coco_y - coco_err:.2f}) -- ({x2:.2f},{coco_y + coco_err:.2f});")
        lines.append(f"\\node[rotate=35,anchor=east] at ({x2 + 0.15:.2f},-0.15) {{{labels[i]}}};")
        lines.append(f"\\node[above] at ({x2:.2f},{coco_y + coco_err + 0.05:.2f}) {{{group['coco_mean']:.3f}}};")
    lines.append("\\end{tikzpicture}")
    (FIG_DIR / "fig2_seed_anchor.tikz.tex").write_text("\n".join(lines) + "\n")


def write_fig3() -> None:
    data = json.loads(DIAGNOSTICS.read_text())
    rows = [
        r
        for r in data["failure_taxonomy"]
        if r["failures"] >= 50 and r["total"] >= 200
    ]
    rows.sort(key=lambda r: r["failures"], reverse=True)
    rows = rows[:8]
    max_fail = max(r["failures"] for r in rows)
    w = 6.4
    step = 0.42
    lines = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{tikzpicture}[font=\\footnotesize]",
    ]
    for i, row in enumerate(rows):
        y = (len(rows) - 1 - i) * step
        bar_w = row["failures"] / max_fail * w
        readable = {
            "swap_att": "Attribute swap",
            "swap_obj": "Object swap",
            "replace_rel": "Relation replace",
            "replace_att": "Attribute replace",
            "word_order": "Word order",
            "to the right of": "Spatial: right of",
            "to the left of": "Spatial: left of",
            "aro_visual_attribution": "Visual attribution",
        }
        label = readable.get(row["edit_type"], row["edit_type"])
        if row["source"] == "aro_visual_relation":
            label = f"ARO {label}"
        elif row["source"].startswith("sugarcrepe"):
            label = f"SugarCrepe {label}"
        elif row["source"] == "aro_visual_attribution":
            label = "ARO Visual attribution"
        lines.append(f"\\node[anchor=east] at (-0.10,{y + 0.08:.2f}) {{{label}}};")
        lines.append(f"\\draw[fill=blue!55!black,draw=blue!55!black] (0,{y:.2f}) rectangle ({bar_w:.2f},{y + 0.16:.2f});")
        lines.append(f"\\node[anchor=west] at ({bar_w + 0.08:.2f},{y + 0.08:.2f}) {{{row['failures']}/{row['total']} ({row['failure_rate']:.2f})}};")
    lines.append(f"\\draw[->] (0,-0.15) -- ({w + 0.35:.2f},-0.15) node[right] {{failures}};")
    for tick in [0, 50, 100, 150]:
        x = tick / max_fail * w
        lines.append(f"\\draw ({x:.2f},-0.11) -- ({x:.2f},-0.19) node[below] {{{tick}}};")
    lines.append("\\end{tikzpicture}")
    (FIG_DIR / "fig3_failure_taxonomy.tikz.tex").write_text("\n".join(lines) + "\n")


def write_tables(rows: dict[str, dict], groups: list[dict]) -> None:
    table1_runs = ["R003", "R005", "R011", "R014", "R019", "R028", "R033", "R040"]
    table1 = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main relation/anchor-preservation comparison. Relation accuracy is measured on staged ARO/SugarCrepe-style relation suites. COCO is image-to-text Recall@1 on the 1k-image guardrail split. Residual is the mean relative magnitude of the learned residual branch where available.}",
        "\\label{tab:main_results}",
        "\\small",
        "\\begin{tabular}{llrrrrrp{2.9cm}}",
        "\\toprule",
        "Run & System & Relation & $\\Delta$ vs R005 & COCO R@1 & ImageNet & Residual & Notes \\\\",
        "\\midrule",
    ]
    r005 = rows["R005"]["relation"]
    for run in table1_runs:
        row = rows[run]
        delta = None if run == "R003" else row["relation"] - r005
        table1.append(
            f"{run} & {row['system']} & {fmt(row['relation'])} & {fmt(delta)} & {fmt(row['coco_i2t'])} & {fmt(row['imagenet'])} & {fmt(row['residual_mag'])} & {technical_note(row['gate'])} \\\\"
        )
    table1.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (FIG_DIR / "table1_main_results.tex").write_text("\n".join(table1))

    ablations = [
        ("R023", "yes", "0", "no-targeted, weak anchor"),
        ("R024", "no", "1", "generic hard negatives"),
        ("R028", "yes", "0", "selected stronger-anchor"),
        ("R029", "yes", "1", "matched targeted control"),
    ]
    table2 = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Structured counterfactuals, not the targeted relational loss $L_{targeted}$, explain the retained relation gain.}",
        "\\label{tab:ablation}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llccrr}",
        "\\toprule",
        "Run & Variant & Struct. CF & $L_{targeted}$ & Relation & COCO \\\\",
        "\\midrule",
    ]
    for run, structured, targeted, variant in ablations:
        row = rows[run]
        table2.append(
            f"{run} & {variant} & {structured} & {targeted} & {fmt(row['relation'])} & {fmt(row['coco_i2t'])} \\\\"
        )
    table2.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    (FIG_DIR / "table2_ablation.tex").write_text("\n".join(table2))

    top = json.loads(TOPVENUE.read_text())
    slot = top["R041"]["R028"]
    flickr = top["R045"]
    r043 = top["R043_R044"]["R043"]
    table3 = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{External-validity and mechanism checks bound the claim. These follow-ups strengthen the limitation story rather than changing the main positive result.}",
        "\\label{tab:external_limits}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{p{2.1cm}p{4.1cm}p{7.7cm}}",
        "\\toprule",
        "Check & Result & Interpretation \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Downstream and external retrieval checks}} \\\\",
        "Winoground R034 & R028 text/image/group 0.2400/0.1100/0.0700 vs OpenCLIP 0.2775/0.1075/0.0825 & Text and group scores drop while image score is nearly flat; no broad binding claim. \\\\",
        f"Flickr30k R045 & R028 I2T R@1 {flickr['R028']['image_to_text_r1']:.4f} vs OpenCLIP {flickr['OpenCLIP']['image_to_text_r1']:.4f}; R038 {flickr['R038']['image_to_text_r1']:.4f} & External retrieval tax persists, smaller for anchor-20 R038. \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Mechanistic and preservation probes}} \\\\",
        f"Slot responsibility R041 & entropy {slot['mean_slot_entropy']:.4f}, top-1 concentration {slot['mean_top1_slot_concentration']:.4f}, edited overlap {slot['mean_edited_span_overlap']:.4f} & Diffuse responsibility; no interpretable semantic-channel claim. \\\\",
        f"ROP ablation R043 & relation {r043['eval']['accuracy']:.4f}, COCO {r043['guardrail']['coco_retrieval']['image_to_text_r1']:.4f}, ImageNet {r043['guardrail']['imagenet_zeroshot']['top1']:.4f} & Better preservation but relation gate fails; ROP remains an ablation/future direction. \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    (FIG_DIR / "table3_external_limits.tex").write_text("\n".join(table3))

    group_table = [
        "% Auto-generated by figures/generate_paper_assets.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Seed summaries for RRB variants. Values are mean $\\pm$ sample standard deviation.}",
        "\\label{tab:seed_summary}",
        "\\small",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Group & Relation & COCO R@1 \\\\",
        "\\midrule",
    ]
    for group in groups:
        group_table.append(
            f"{group['name']} & {group['relation_mean']:.4f} $\\pm$ {group['relation_std']:.4f} & {group['coco_mean']:.4f} $\\pm$ {group['coco_std']:.4f} \\\\"
        )
    group_table.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (FIG_DIR / "table4_seed_summary.tex").write_text("\n".join(group_table))


def write_latex_includes() -> None:
    include = r"""% Auto-generated by figures/generate_paper_assets.py

% === Figure 1: Relation/COCO Pareto ===
\begin{figure*}[t]
    \centering
    \input{figures/fig1_relation_coco.tikz.tex}
    \caption{Relation accuracy versus COCO image-to-text retrieval for key evaluated systems. Hard-token controls replace the pretrained embedding path with a bottlenecked representation, while no-channel controls use a residual path without the multi-channel bottleneck. Standard PEFT baselines such as LoRA and text-only tuning were not evaluated in this ablation study, which focuses on the architectural impact of the RRB bottleneck.}
    \label{fig:relation_coco}
\end{figure*}

% === Figure 2: Seed and anchoring summary ===
\begin{figure}[t]
    \centering
    \input{figures/fig2_seed_anchor.tikz.tex}
    \caption{Seed summaries show stable relation gains but imperfect COCO retention. Anchor-10 and anchor-20 no-targeted variants have similar mean COCO R@1 in these runs (0.711 and 0.709), so stronger anchoring does not eliminate retrieval performance degradation. Error bars are sample standard deviations.}
    \label{fig:seed_anchor}
\end{figure}

% === Figure 3: Failure taxonomy ===
\begin{figure}[t]
    \centering
    \input{figures/fig3_failure_taxonomy.tikz.tex}
    \caption{Largest remaining failure groups for the selected no-targeted, strong-anchor RRB model on the R030 diagnostic suite. The model still struggles with left/right spatial predicates, object and attribute swaps, and relation replacement cases.}
    \label{fig:failure_taxonomy}
\end{figure}
"""
    (FIG_DIR / "latex_includes.tex").write_text(include)


def write_manifest(rows: dict[str, dict], groups: list[dict]) -> None:
    payload = {
        "source_files": {
            "consolidated": str(CONSOLIDATED.relative_to(ROOT)),
            "diagnostics": str(DIAGNOSTICS.relative_to(ROOT)),
            "topvenue": str(TOPVENUE.relative_to(ROOT)),
        },
        "parsed_runs": rows,
        "seed_groups": groups,
    }
    (FIG_DIR / "generated_data.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    rows = parse_main_results()
    groups = [
        summarize_group(rows, "full RRB", ["R019", "R025", "R027"]),
        summarize_group(rows, "anchor-10 no-targeted", ["R028", "R031", "R032"]),
        summarize_group(rows, "anchor-20 no-targeted", ["R035", "R036", "R037", "R038", "R039"]),
    ]
    write_fig1(rows)
    write_fig2(groups)
    write_fig3()
    write_tables(rows, groups)
    write_latex_includes()
    write_manifest(rows, groups)
    print("Generated paper assets in figures/")


if __name__ == "__main__":
    main()
