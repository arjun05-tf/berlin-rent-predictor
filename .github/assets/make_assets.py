"""Generate the animated SVGs used by README.md.

The hero cloud is a PCA of the real cleaned listings in
`data/processed/berlin_processed.csv`, one dot per flat, coloured by rent
quintile, so it shows the actual feature space the model fits rather than
decoration. The bar charts carry the measured numbers from `scripts/train.py`
and `scripts/tune.py`.

    python .github/assets/make_assets.py
"""
import csv
import math
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).parent
ROOT = OUT.parent.parent
PROCESSED = ROOT / "data" / "processed" / "berlin_processed.csv"

BG, PANEL, INK, MUTED, DIM = "#0b0f14", "#1b2735", "#e6edf3", "#8b98a6", "#5a6673"
GOLD, RED, CYAN = "#f2b53c", "#e0564f", "#4cc9f0"
# Cheap to expensive. Five bins, because the cloud is coloured by rent quintile.
RENT_COLORS = ["#4cc9f0", "#4ade80", "#f2b53c", "#fb923c", "#e0564f"]
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SANS = "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, Helvetica, sans-serif"

# 10,388 listings will not fit in a README-sized SVG: every point costs four
# keyframe tracks. Sample deterministically and keep the frame count low.
POINTS = 240
FRAMES = 24
SPIN_SECONDS = 32

FEATURES = ["livingSpace", "rooms", "floor", "building_age", "amenity_score"]


def load_listings():
    """Real feature matrix and rents from the cleaned Berlin CSV."""
    rows = list(csv.DictReader(PROCESSED.open(encoding="utf-8")))
    total = len(rows)
    step = max(1, total // POINTS)
    sample = rows[::step][:POINTS]

    def column(name):
        return np.array(
            [float(r[name]) if r[name] not in ("", "NA") else np.nan for r in sample]
        )

    matrix = np.column_stack([column(f) for f in FEATURES])
    # Median impute, then standardise: floor and livingSpace are on different
    # scales and PCA would otherwise just rediscover square metres.
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        values[np.isnan(values)] = np.nanmedian(values)
    matrix = (matrix - matrix.mean(0)) / matrix.std(0)
    rents = column("baseRent")
    districts = {r["geo_bln"] for r in sample}
    return matrix, rents, total, len(districts)


def pca3(matrix):
    centered = matrix - matrix.mean(0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:3].T
    variance = (singular[:3] ** 2).sum() / (singular**2).sum()
    # Scale on the 90th percentile radius, not the max: a single 300 m² outlier
    # would otherwise collapse everything else into the centre of the sphere.
    norms = np.linalg.norm(coords, axis=1)
    coords /= np.percentile(norms, 90)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords *= np.minimum(1.0, 1.1 / norms)
    return coords, variance


def neighbour_edges(matrix, coords, limit=70):
    """Nearest neighbour per listing in feature space, deduped, closest kept.

    Pairs that PCA throws far apart are dropped: the line would cross the whole
    cloud and read as noise rather than as a neighbour link.
    """
    distances = ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(distances, np.inf)
    edges = {}
    for i, j in enumerate(distances.argmin(1)):
        edges[tuple(sorted((i, int(j))))] = float(distances[i, j])
    span = {e: float(np.linalg.norm(coords[e[0]] - coords[e[1]])) for e in edges}
    cutoff = np.percentile(list(span.values()), 55)
    kept = {e: d for e, d in edges.items() if span[e] <= cutoff}
    return [edge for edge, _ in sorted(kept.items(), key=lambda kv: kv[1])[:limit]]


def rotate_frames(coords, tilt=math.radians(17)):
    """Spin about Y, fixed tilt about X, perspective-project. One list per frame."""
    frames = []
    for frame in range(FRAMES + 1):  # +1 closes the loop seamlessly
        angle = 2 * math.pi * (frame % FRAMES) / FRAMES
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cos_t, sin_t = math.cos(tilt), math.sin(tilt)
        projected = []
        for x, y, z in coords:
            rx, rz = x * cos_a + z * sin_a, -x * sin_a + z * cos_a
            ry, rz = y * cos_t - rz * sin_t, y * sin_t + rz * cos_t
            scale = 4.5 / (4.5 - rz)  # weak perspective: depth cue, stays in frame
            projected.append((rx * scale, -ry * scale, scale))
        frames.append(projected)
    return frames


def trim(value, decimals):
    """Shortest exact text for a keyframe number; only a fraction loses zeros."""
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def spin(attr, series, decimals=1):
    """Evenly spaced keyframes, so SMIL's default keyTimes stays out of the file."""
    values = ";".join(trim(v, decimals) for v in series)
    return (
        f'<animate attributeName="{attr}" dur="{SPIN_SECONDS}s" repeatCount="indefinite" '
        f'values="{values}"/>'
    )


def panel(width, height, title=None, subtitle=None):
    parts = [
        f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',
        f'<rect width="{width}" height="{height}" rx="14" fill="url(#grid)"/>',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="14" '
        f'fill="none" stroke="{PANEL}"/>',
    ]
    if title:
        parts.append(
            f'<text x="28" y="36" font-family="{MONO}" font-size="12" fill="{MUTED}" '
            f'letter-spacing="2.4">{title}</text>'
        )
    if subtitle:
        parts.append(
            f'<text x="28" y="58" font-family="{SANS}" font-size="13" fill="{DIM}">{subtitle}</text>'
        )
    return "".join(parts)


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img"><defs>'
        f'<pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse">'
        f'<path d="M36 0H0V36" fill="none" stroke="#131c26" stroke-width="1"/></pattern>'
        f'<radialGradient id="halo" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="{GOLD}" stop-opacity="0.15"/>'
        f'<stop offset="70%" stop-color="{GOLD}" stop-opacity="0"/></radialGradient>'
        f'<linearGradient id="goldfill" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{GOLD}"/><stop offset="50%" stop-color="#fff0c0"/>'
        f'<stop offset="100%" stop-color="{GOLD}"/>'
        f'<animateTransform attributeName="gradientTransform" type="translate" from="-1 0" '
        f'to="1 0" dur="7s" repeatCount="indefinite"/></linearGradient>'
        f'</defs>{body}</svg>'
    )


# ---------------------------------------------------------------- hero


def hero(frames, rents, variance, total, districts, edges):
    width, height = 1200, 430
    cx, cy, radius = 884, 212, 118
    quintiles = np.quantile(rents, [0.2, 0.4, 0.6, 0.8])
    colour = [RENT_COLORS[int(np.searchsorted(quintiles, r))] for r in rents]

    body = [panel(width, height)]
    body.append(
        f'<clipPath id="cloudclip"><rect x="628" y="14" width="{width - 642}" '
        f'height="{height - 56}" rx="12"/></clipPath><g clip-path="url(#cloudclip)">'
    )
    body.append(f'<ellipse cx="{cx}" cy="{cy}" rx="215" ry="215" fill="url(#halo)"/>')

    for i, j in edges:
        body.append(
            f'<line stroke="{GOLD}" stroke-opacity="0.22" stroke-width="1">'
            + spin("x1", [cx + frames[f][i][0] * radius for f in range(FRAMES + 1)], 0)
            + spin("y1", [cy + frames[f][i][1] * radius for f in range(FRAMES + 1)], 0)
            + spin("x2", [cx + frames[f][j][0] * radius for f in range(FRAMES + 1)], 0)
            + spin("y2", [cy + frames[f][j][1] * radius for f in range(FRAMES + 1)], 0)
            + "</line>"
        )

    for i in sorted(range(len(rents)), key=lambda n: frames[0][n][2]):
        body.append(
            f'<circle fill="{colour[i]}">'
            + spin("cx", [cx + frames[f][i][0] * radius for f in range(FRAMES + 1)], 0)
            + spin("cy", [cy + frames[f][i][1] * radius for f in range(FRAMES + 1)], 0)
            + spin("r", [1.3 + (frames[f][i][2] - 0.78) * 6.0 for f in range(FRAMES + 1)], 2)
            + spin(
                "fill-opacity",
                [min(0.95, max(0.25, (frames[f][i][2] - 0.74) * 2.0)) for f in range(FRAMES + 1)],
                2,
            )
            + "</circle>"
        )
    body.append("</g>")

    legend_x = cx - 148
    for i, label in enumerate(["cheap", "", "", "", ""]):
        body.append(f'<rect x="{legend_x + i * 15}" y="{height - 46}" width="11" height="6" '
                    f'rx="2" fill="{RENT_COLORS[i]}"/>')
        if label:
            body.append(
                f'<text x="{legend_x + i * 15 + 5}" y="{height - 52}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="9" fill="{DIM}">{label}</text>'
            )
    body.append(
        f'<text x="{legend_x + 92}" y="{height - 52}" font-family="{MONO}" font-size="9" '
        f'fill="{DIM}">expensive</text>'
    )
    body.append(
        f'<text x="{cx}" y="{height - 26}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="10.5" fill="{DIM}">{POINTS} of {total:,} listings &#183; PCA of five '
        f'features &#183; {variance * 100:.0f}% variance &#183; colour = rent quintile</text>'
    )

    body.append(
        f'<text x="60" y="92" font-family="{MONO}" font-size="12" fill="{CYAN}" '
        f'letter-spacing="3.4">GRADIENT BOOSTING &#183; LEAKAGE-AWARE EVALUATION</text>'
    )
    body.append(
        f'<text x="60" y="152" font-family="{MONO}" font-size="44" font-weight="700" '
        f'fill="url(#goldfill)">BerlinRentML</text>'
    )
    body.append(
        f'<line x1="60" y1="176" x2="60" y2="176" stroke="{GOLD}" stroke-width="2" '
        f'stroke-opacity="0.55"><animate attributeName="x2" from="60" to="500" dur="1.4s" '
        f'fill="freeze"/></line>'
    )
    body.append(
        f'<text x="60" y="206" font-family="{SANS}" font-size="16" fill="{MUTED}">'
        f'Berlin rent estimates, scored on postal codes the model never saw.</text>'
    )

    query = 'curl localhost:8000/predict -d @flat.json  # 60 m², Mitte'
    answer = "1197.21 &#8364;  &#183;  90% interval 837&#8211;1712"
    qw, steps = 442, 32
    tick_times = ";".join(f"{i / 64:.4f}" for i in range(steps))
    widths = ";".join(f"{qw * min(i, len(query)) / len(query):.1f}" for i in range(steps))
    carets = ";".join(f"{94 + qw * min(i, len(query)) / len(query):.1f}" for i in range(steps))

    body.append(
        f'<rect x="60" y="238" width="524" height="94" rx="10" fill="#0e151d" stroke="{PANEL}"/>'
        f'<text x="76" y="270" font-family="{MONO}" font-size="13" fill="{GOLD}">$</text>'
        f'<clipPath id="typeclip"><rect x="94" y="254" height="22" width="0">'
        f'<animate attributeName="width" dur="9s" repeatCount="indefinite" calcMode="discrete" '
        f'keyTimes="{tick_times}" values="{widths}"/></rect></clipPath>'
        f'<g clip-path="url(#typeclip)"><text x="94" y="270" font-family="{MONO}" font-size="12" '
        f'fill="{INK}" textLength="{qw}" lengthAdjust="spacingAndGlyphs">{query}</text></g>'
        f'<rect y="256" width="7" height="16" fill="{GOLD}">'
        f'<animate attributeName="x" dur="9s" repeatCount="indefinite" calcMode="discrete" '
        f'keyTimes="{tick_times}" values="{carets}"/>'
        f'<animate attributeName="opacity" values="1;0;1" dur="0.9s" repeatCount="indefinite"/>'
        f'</rect>'
        f'<text x="94" y="308" font-family="{MONO}" font-size="13" fill="{CYAN}" '
        f'textLength="330" lengthAdjust="spacingAndGlyphs" opacity="0">{answer}'
        f'<animate attributeName="opacity" dur="9s" repeatCount="indefinite" '
        f'keyTimes="0;0.52;0.6;0.93;1" values="0;0;1;1;0"/></text>'
    )

    x = 60
    for i, (value, label) in enumerate(
        [
            (f"{total:,}", "listings"),
            ("&#8364;191", "MAE, unseen PLZ"),
            ("0.836", "R&#178;"),
            ("88.5%", "90% coverage"),
        ]
    ):
        body.append(
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" '
            f'begin="{1.2 + i * 0.14}s" fill="freeze"/>'
            f'<rect x="{x}" y="356" width="124" height="46" rx="9" fill="#0e151d" '
            f'stroke="{PANEL}"/>'
            f'<text x="{x + 14}" y="379" font-family="{MONO}" font-size="17" font-weight="700" '
            f'fill="{INK}">{value}</text>'
            f'<text x="{x + 14}" y="394" font-family="{MONO}" font-size="9.5" fill="{DIM}" '
            f'letter-spacing="1.1">{label}</text></g>'
        )
        x += 134
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- pipeline

STAGES = [
    ("01", "Clean", "10,388 rows kept", "of 268,850 German", CYAN),
    ("02", "Feature", "age, size bins, log m²", "amenity score", "#a78bfa"),
    ("03", "Split", "GroupedSplit by PLZ", "test codes unseen", GOLD),
    ("04", "Fit", "LightGBM, tuned", "40-candidate search", "#4ade80"),
    ("05", "Calibrate", "split conformal", "90% intervals", RED),
]


def pipeline():
    width, height = 1200, 250
    box_w, gap, y, cycle = 196, 30, 96, 7.5
    total = len(STAGES) * box_w + (len(STAGES) - 1) * gap
    x0 = (width - total) / 2

    body = [
        panel(width, height, "PIPELINE", "the split happens before the fit, which is the point")
    ]
    body.append(
        f'<line x1="{x0}" y1="{y + 40}" x2="{x0 + total}" y2="{y + 40}" stroke="{PANEL}" '
        f'stroke-width="2"/>'
    )
    for i, (num, name, tool, out, colour) in enumerate(STAGES):
        x = x0 + i * (box_w + gap)
        begin = f"{i * cycle / len(STAGES):.2f}s"
        body.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="80" rx="11" fill="#0e151d" '
            f'stroke="{PANEL}"><animate attributeName="stroke" values="{PANEL};{colour};{PANEL}" '
            f'dur="{cycle}s" begin="{begin}" repeatCount="indefinite" keyTimes="0;0.08;0.3"/>'
            f'</rect>'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="80" rx="11" fill="{colour}" '
            f'opacity="0"><animate attributeName="opacity" values="0;0.11;0" dur="{cycle}s" '
            f'begin="{begin}" repeatCount="indefinite" keyTimes="0;0.08;0.3"/></rect>'
            f'<text x="{x + 16}" y="{y + 26}" font-family="{MONO}" font-size="11" fill="{colour}" '
            f'letter-spacing="1.6">{num}</text>'
            f'<text x="{x + 16}" y="{y + 48}" font-family="{SANS}" font-size="17" '
            f'font-weight="600" fill="{INK}">{name}</text>'
            f'<text x="{x + 16}" y="{y + 66}" font-family="{MONO}" font-size="10.5" '
            f'fill="{MUTED}">{tool}</text>'
            f'<text x="{x + 16}" y="{y + 108}" font-family="{MONO}" font-size="10.5" '
            f'fill="{DIM}">{out}</text>'
        )
        if i < len(STAGES) - 1:
            mid = x + box_w + gap / 2
            body.append(
                f'<path d="M{mid - 7} {y + 34} l7 6 -7 6" fill="none" stroke="{DIM}" '
                f'stroke-width="1.6"/>'
            )
    body.append(
        f'<circle cy="{y + 40}" r="4.5" fill="{GOLD}"><animate attributeName="cx" from="{x0}" '
        f'to="{x0 + total}" dur="{cycle}s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.04;0.96;1" '
        f'dur="{cycle}s" repeatCount="indefinite"/></circle>'
    )
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- leakage

# Mean absolute error in euro per month, from scripts/train.py.
MODELS = [
    ("Mean baseline", 535, 550),
    ("Median baseline", 509, 507),
    ("Linear regression", 186, 291),
    ("Ridge", 186, 202),
    ("Random forest", 182, 208),
    ("LightGBM", 162, 194),
]


def leakage():
    width, height = 1200, 520
    left, top, bar_w, row_h = 212, 112, 700, 48
    axis_max = 600
    baseline = top + row_h * len(MODELS)

    body = [
        panel(
            width,
            height,
            "GENERALISATION &#183; 10,388 LISTINGS",
            "mean absolute error on a random split against a postal-code grouped split",
        )
    ]
    x = width - 28
    for label, colour in (("unseen postal codes", GOLD), ("random split", DIM)):
        x -= len(label) * 6.6
        body.append(
            f'<text x="{x}" y="36" font-family="{MONO}" font-size="11" fill="{MUTED}">'
            f'{label}</text>'
            f'<rect x="{x - 20}" y="27" width="11" height="11" rx="3" fill="{colour}"/>'
        )
        x -= 44
    for tick in range(0, axis_max + 1, 100):
        x = left + bar_w * tick / axis_max
        body.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{baseline}" stroke="#141d27"/>'
            f'<text x="{x:.1f}" y="{baseline + 20}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="10" fill="{DIM}">{tick}</text>'
        )
    body.append(
        f'<text x="{left + bar_w / 2}" y="{baseline + 38}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="10" fill="{DIM}">MAE, &#8364; per month, lower is '
        f'better</text>'
    )
    for i, (name, random_split, grouped) in enumerate(MODELS):
        y = top + i * row_h
        body.append(
            f'<text x="28" y="{y + 24}" font-family="{SANS}" font-size="14" font-weight="600" '
            f'fill="{INK if name == "LightGBM" else MUTED}">{name}</text>'
        )
        for row, (value, colour) in enumerate(((random_split, DIM), (grouped, GOLD))):
            by = y + row * 15
            body.append(
                f'<rect x="{left}" y="{by}" height="11" rx="3" width="0" fill="{colour}" '
                f'fill-opacity="{0.55 if row == 0 else 1}">'
                f'<animate attributeName="width" from="0" to="{bar_w * value / axis_max:.1f}" '
                f'dur="1.1s" begin="{0.25 + i * 0.12 + row * 0.08}s" fill="freeze" '
                f'calcMode="spline" keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/></rect>'
                f'<text x="{left + bar_w * value / axis_max + 10:.1f}" y="{by + 10}" '
                f'font-family="{MONO}" font-size="11" fill="{INK if row else MUTED}" '
                f'opacity="0">&#8364;{value}'
                f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
                f'begin="{1.3 + i * 0.12 + row * 0.08}s" fill="freeze"/></text>'
            )
    body.append(
        f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="2.4s" '
        f'fill="freeze"/>'
        f'<rect x="{left}" y="{baseline + 52}" width="{bar_w}" height="40" rx="9" fill="#0e151d" '
        f'stroke="{PANEL}"/>'
        f'<text x="{left + 16}" y="{baseline + 77}" font-family="{SANS}" font-size="13" '
        f'fill="{MUTED}">Linear regression loses <tspan fill="{RED}" font-weight="700">'
        f'&#8364;105</tspan> once test postal codes are unseen; LightGBM loses '
        f'<tspan fill="{GOLD}" font-weight="700">&#8364;32</tspan>.</text></g>'
    )
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- districts

# MAE in euro on the random test set, from scripts/analyze_errors.py.
DISTRICTS = [
    ("Mitte", 241),
    ("Prenzlauer Berg", 241),
    ("Wilmersdorf", 240),
    ("Charlottenburg", 220),
    ("Tiergarten", 189),
    ("Friedrichshain", 162),
    ("Kreuzberg", 162),
    ("Neuk&#246;lln", 141),
    ("Wedding", 94),
    ("Spandau", 85),
]


def districts():
    width, height = 1200, 386
    left, top, bar_w, col_h = 150, 96, 54, 150
    axis_max = 260
    gap = (width - left - 60) / len(DISTRICTS)

    body = [
        panel(
            width,
            height,
            "ERROR BY DISTRICT",
            "where the model is wrong, and by how much, on the random test set",
        )
    ]
    baseline = top + col_h
    body.append(
        f'<line x1="{left - 24}" y1="{baseline}" x2="{width - 40}" y2="{baseline}" '
        f'stroke="{PANEL}" stroke-width="2"/>'
    )
    for i, (name, mae) in enumerate(DISTRICTS):
        x = left + i * gap
        bar_h = col_h * mae / axis_max
        # The top four are the expensive central districts; the finding is that
        # price level and error track each other, so colour by rank.
        colour = RED if i < 4 else (GOLD if i < 7 else CYAN)
        body.append(
            f'<rect x="{x:.1f}" y="{baseline}" width="{bar_w}" height="0" rx="4" fill="{colour}" '
            f'fill-opacity="0.85">'
            f'<animate attributeName="height" from="0" to="{bar_h:.1f}" dur="1.0s" '
            f'begin="{0.3 + i * 0.07}s" fill="freeze" calcMode="spline" '
            f'keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/>'
            f'<animate attributeName="y" from="{baseline}" to="{baseline - bar_h:.1f}" dur="1.0s" '
            f'begin="{0.3 + i * 0.07}s" fill="freeze" calcMode="spline" '
            f'keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/></rect>'
            f'<text x="{x + bar_w / 2:.1f}" y="{baseline - bar_h - 10:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" fill="{INK}" opacity="0">&#8364;{mae}'
            f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
            f'begin="{1.1 + i * 0.07}s" fill="freeze"/></text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{baseline + 26}" text-anchor="end" '
            f'font-family="{MONO}" font-size="10" fill="{MUTED}" '
            f'transform="rotate(-28 {x + bar_w / 2:.1f} {baseline + 26})">{name}</text>'
        )
    body.append(
        f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="2.0s" '
        f'fill="freeze"/>'
        f'<rect x="{left - 24}" y="{baseline + 82}" width="{width - left - 36}" height="40" '
        f'rx="9" fill="#0e151d" stroke="{PANEL}"/>'
        f'<text x="{left - 8}" y="{baseline + 107}" font-family="{SANS}" font-size="13" '
        f'fill="{MUTED}">Error tracks price level: the four most expensive central districts '
        f'carry roughly <tspan fill="{RED}" font-weight="700">2.8x</tspan> the error of the '
        f'cheapest ones, and large flats are under-priced.</text></g>'
    )
    return svg(width, height, "".join(body))


if __name__ == "__main__":
    matrix, rents, total, district_count = load_listings()
    coords, variance = pca3(matrix)
    frames = rotate_frames(coords)
    edges = neighbour_edges(matrix, coords)
    for name, markup in (
        ("hero.svg", hero(frames, rents, variance, total, district_count, edges)),
        ("pipeline.svg", pipeline()),
        ("leakage.svg", leakage()),
        ("districts.svg", districts()),
    ):
        (OUT / name).write_text(markup, encoding="utf-8")
        print(f"{name:16} {len(markup) / 1024:6.1f} KB")
