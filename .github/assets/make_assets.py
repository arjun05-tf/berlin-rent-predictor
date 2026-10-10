"""Generate the figures used by README.md.

House style: an architect's drawing. Ink on drafting paper, hairline rules,
hatching instead of solid fill for the weaker series, serif headings and a
single terracotta accent.

The skyline is built from the real cleaned listings in
`data/processed/berlin_processed.csv`: one building per flat, height from its
rent, width from its floor area. The bar charts carry the measured numbers
from `scripts/train.py`, `scripts/tune.py` and `scripts/analyze_errors.py`.

    python .github/assets/make_assets.py
"""
import csv
import pathlib

OUT = pathlib.Path(__file__).parent
ROOT = OUT.parent.parent
PROCESSED = ROOT / "data" / "processed" / "berlin_processed.csv"

PAPER = "#f4f0e6"
PAPER_DEEP = "#eae4d6"
INK = "#1d1b17"
GRAPHITE = "#6c665c"
FAINT = "#c9c1b1"
TERRA = "#b4502e"
SLATE = "#2f5d7c"
SERIF = "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

BUILDINGS = 54


def load_listings():
    rows = list(csv.DictReader(PROCESSED.open(encoding="utf-8")))
    flats = []
    for row in rows:
        try:
            rent = float(row["baseRent"])
            size = float(row["livingSpace"])
        except (TypeError, ValueError):
            continue
        flats.append((rent, size, row["geo_bln"]))
    flats.sort()
    # Even slices through the rent order, so the skyline spans the market
    # instead of showing whichever flats sit at the top of the file.
    step = len(flats) / BUILDINGS
    sample = [flats[min(len(flats) - 1, int(i * step))] for i in range(BUILDINGS)]
    return sample, len(rows)


def frame(width, height, title=None, subtitle=None, sheet=None):
    """Drawing sheet: graph paper, a hairline border and a title block."""
    parts = [
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        f'<rect width="{width}" height="{height}" fill="url(#graph)"/>',
        f'<rect x="10.5" y="10.5" width="{width - 21}" height="{height - 21}" fill="none" '
        f'stroke="{INK}" stroke-width="0.8" stroke-opacity="0.55"/>',
        f'<rect x="16.5" y="16.5" width="{width - 33}" height="{height - 33}" fill="none" '
        f'stroke="{INK}" stroke-width="0.4" stroke-opacity="0.3"/>',
    ]
    if title:
        parts.append(
            f'<text x="40" y="52" font-family="{MONO}" font-size="11" fill="{TERRA}" '
            f'letter-spacing="3.2">{title}</text>'
        )
    if subtitle:
        parts.append(
            f'<text x="40" y="76" font-family="{SERIF}" font-size="15" fill="{GRAPHITE}" '
            f'font-style="italic">{subtitle}</text>'
        )
    if sheet:
        parts.append(
            f'<text x="{width - 40}" y="52" text-anchor="end" font-family="{MONO}" '
            f'font-size="10" fill="{GRAPHITE}" letter-spacing="2">{sheet}</text>'
        )
    return "".join(parts)


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img"><defs>'
        f'<pattern id="graph" width="20" height="20" patternUnits="userSpaceOnUse">'
        f'<path d="M20 0H0V20" fill="none" stroke="{SLATE}" stroke-width="0.4" '
        f'stroke-opacity="0.12"/></pattern>'
        f'<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{PAPER_DEEP}"/>'
        f'<path d="M0 0V6" stroke="{GRAPHITE}" stroke-width="1.1" stroke-opacity="0.55"/>'
        f'</pattern>'
        f'<pattern id="hatchterra" width="5" height="5" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(-45)">'
        f'<rect width="5" height="5" fill="{TERRA}" fill-opacity="0.14"/>'
        f'<path d="M0 0V5" stroke="{TERRA}" stroke-width="1" stroke-opacity="0.5"/>'
        f'</pattern>'
        f'</defs>{body}</svg>'
    )


def dimension(x1, x2, y, text, colour=GRAPHITE):
    """A surveyor's dimension line, ticked at both ends."""
    return (
        f'<path d="M{x1} {y} H{x2}" stroke="{colour}" stroke-width="0.7"/>'
        f'<path d="M{x1} {y - 4} V{y + 4} M{x2} {y - 4} V{y + 4}" stroke="{colour}" '
        f'stroke-width="0.7"/>'
        f'<rect x="{(x1 + x2) / 2 - 190}" y="{y - 9}" width="380" height="14" fill="{PAPER}"/>'
        f'<text x="{(x1 + x2) / 2}" y="{y + 2}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="9.5" fill="{colour}" letter-spacing="1.4">{text}</text>'
    )


# ---------------------------------------------------------------- hero


def hero(sample, total):
    width, height = 1200, 470
    base, roof = 344, 196
    left, span = 52, width - 104
    rents = [rent for rent, _, _ in sample]
    sizes = [size for _, size, _ in sample]
    ceiling = sorted(rents)[int(len(rents) * 0.97)]
    widest = max(sizes)

    body = [frame(width, height)]

    # Title block first, so the elevation below it stays clear.
    body.append(
        f'<text x="52" y="76" font-family="{MONO}" font-size="11" fill="{TERRA}" '
        f'letter-spacing="3.4">BERLIN &#183; RENT &#183; LEAKAGE-AWARE EVALUATION</text>'
        f'<text x="50" y="126" font-family="{SERIF}" font-size="44" font-weight="700" '
        f'fill="{INK}">BerlinRentML</text>'
        f'<path d="M52 142 H52" stroke="{TERRA}" stroke-width="1.6">'
        f'<animate attributeName="d" from="M52 142 H52" to="M52 142 H470" dur="1.2s" '
        f'fill="freeze"/></path>'
        f'<text x="52" y="168" font-family="{SERIF}" font-size="17" fill="{GRAPHITE}" '
        f'font-style="italic">Rent estimates scored on postal codes the model never saw.</text>'
        f'<text x="{width - 52}" y="76" text-anchor="end" font-family="{MONO}" font-size="10" '
        f'fill="{GRAPHITE}" letter-spacing="2">ELEVATION &#183; SHEET 01</text>'
        f'<text x="{width - 52}" y="100" text-anchor="end" font-family="{MONO}" '
        f'font-size="9.5" fill="{GRAPHITE}">height = monthly rent &#183; width = floor area '
        f'&#183; ordered cheapest to dearest</text>'
    )

    slot = span / len(sample)
    body.append(f'<path d="M{left} {base} H{left + span}" stroke="{INK}" stroke-width="1.1"/>')
    for i, (rent, size, _) in enumerate(sample):
        block_w = max(7.0, min(slot - 2.4, slot * 0.5 + 11 * size / widest))
        block_h = roof * min(1.0, rent / ceiling)
        x = left + i * slot + (slot - block_w) / 2
        y = base - block_h
        rise = f"{0.5 + i * 0.035:.2f}s"
        body.append(
            f'<g><animateTransform attributeName="transform" type="translate" '
            f'from="0 {block_h:.0f}" to="0 0" dur="0.9s" begin="{rise}" fill="freeze" '
            f'calcMode="spline" keySplines="0.15 0.85 0.25 1" keyTimes="0;1"/>'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{block_w:.1f}" height="{block_h:.1f}" '
            f'fill="url(#hatchterra)" stroke="{INK}" stroke-width="0.8"/>'
        )
        # Window courses: one storey every 14 px, which is what makes it read
        # as a building rather than a bar.
        storey = y + 11
        while storey < base - 6:
            body.append(
                f'<path d="M{x + 3:.1f} {storey:.1f} H{x + block_w - 3:.1f}" stroke="{INK}" '
                f'stroke-width="0.5" stroke-opacity="0.35"/>'
            )
            storey += 14
        body.append("</g>")

    # A survey line sweeping the elevation, the way the model sweeps the market.
    body.append(
        f'<g opacity="0.85"><line y1="{base - roof - 8}" y2="{base}" stroke="{SLATE}" '
        f'stroke-width="1" stroke-dasharray="5 4">'
        f'<animate attributeName="x1" values="{left};{left + span};{left}" dur="22s" '
        f'repeatCount="indefinite"/>'
        f'<animate attributeName="x2" values="{left};{left + span};{left}" dur="22s" '
        f'repeatCount="indefinite"/></line>'
        f'<circle cy="{base}" r="3.5" fill="{SLATE}">'
        f'<animate attributeName="cx" values="{left};{left + span};{left}" dur="22s" '
        f'repeatCount="indefinite"/></circle></g>'
    )
    body.append(dimension(left, left + span, base + 26,
                          f"{BUILDINGS} FLATS SAMPLED ACROSS {total:,} CLEANED LISTINGS"))

    # Specification strip along the bottom, in four columns.
    specs = [
        ("MODEL", "LightGBM, log-rent target"),
        ("SPLIT", "grouped by postal code"),
        ("ERROR", "191 EUR MAE, unseen PLZ"),
        ("INTERVAL", "90% band, 88.5% covered"),
    ]
    strip_y = base + 56
    column = span / len(specs)
    body.append(f'<path d="M{left} {strip_y} H{left + span}" stroke="{FAINT}" '
                f'stroke-width="0.7"/>')
    for i, (key, value) in enumerate(specs):
        x = left + i * column
        body.append(
            f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur="0.5s" '
            f'begin="{0.9 + i * 0.13}s" fill="freeze"/>'
            f'<text x="{x:.0f}" y="{strip_y + 22}" font-family="{MONO}" font-size="9.5" '
            f'fill="{TERRA}" letter-spacing="2">{key}</text>'
            f'<text x="{x:.0f}" y="{strip_y + 44}" font-family="{SERIF}" font-size="15" '
            f'fill="{INK}">{value}</text></g>'
        )
        if i:
            body.append(f'<path d="M{x - 18:.0f} {strip_y + 6} V{strip_y + 50}" '
                        f'stroke="{FAINT}" stroke-width="0.7"/>')
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- pipeline

STAGES = [
    ("Clean", "10,388 of 268,850 rows kept"),
    ("Feature", "age bands, log area, amenities"),
    ("Split", "GroupedSplit, PLZ held out"),
    ("Fit", "LightGBM, 40-candidate search"),
    ("Calibrate", "split conformal, 90% band"),
]


def pipeline():
    width, height = 1200, 230
    y = 128
    left, span = 60, width - 120
    gap = span / len(STAGES)

    body = [
        frame(width, height, "METHOD", "the split is drawn before the model is fitted",
              "SHEET 02")
    ]
    body.append(f'<path d="M{left} {y} H{left + span}" stroke="{INK}" stroke-width="0.8"/>')
    for i, (name, detail) in enumerate(STAGES):
        cx = left + gap * (i + 0.5)
        begin = f"{i * 1.6:.1f}s"
        body.append(
            f'<circle cx="{cx:.1f}" cy="{y}" r="15" fill="{PAPER}" stroke="{INK}" '
            f'stroke-width="1"/>'
            f'<circle cx="{cx:.1f}" cy="{y}" r="15" fill="{TERRA}" fill-opacity="0">'
            f'<animate attributeName="fill-opacity" values="0;0.22;0" dur="8s" begin="{begin}" '
            f'repeatCount="indefinite" keyTimes="0;0.07;0.26"/></circle>'
            f'<text x="{cx:.1f}" y="{y + 5}" text-anchor="middle" font-family="{SERIF}" '
            f'font-size="14" font-weight="700" fill="{INK}">{i + 1}</text>'
            f'<text x="{cx:.1f}" y="{y - 30}" text-anchor="middle" font-family="{SERIF}" '
            f'font-size="17" fill="{INK}">{name}</text>'
            f'<text x="{cx:.1f}" y="{y + 42}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="9.5" fill="{GRAPHITE}">{detail}</text>'
        )
    body.append(
        f'<circle cy="{y}" r="4" fill="{SLATE}">'
        f'<animate attributeName="cx" from="{left}" to="{left + span}" dur="8s" '
        f'repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.03;0.97;1" dur="8s" '
        f'repeatCount="indefinite"/></circle>'
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
    width, height = 1200, 500
    left, top, bar_w, row_h = 250, 120, 640, 48
    axis_max = 600
    baseline = top + row_h * len(MODELS)

    body = [
        frame(width, height, "FIGURE 1 &#183; GENERALISATION",
              "mean absolute error, random split against postal-code grouped split", "SHEET 03")
    ]
    for tick in range(0, axis_max + 1, 100):
        x = left + bar_w * tick / axis_max
        body.append(
            f'<path d="M{x:.1f} {top - 14} V{baseline + 6}" stroke="{FAINT}" '
            f'stroke-width="0.6"/>'
            f'<text x="{x:.1f}" y="{baseline + 24}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="9.5" fill="{GRAPHITE}">{tick}</text>'
        )
    body.append(
        f'<text x="{left + bar_w / 2}" y="{baseline + 44}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="9.5" fill="{GRAPHITE}" letter-spacing="1.6">'
        f'EUR PER MONTH, LOWER IS BETTER</text>'
    )
    for i, (name, random_split, grouped) in enumerate(MODELS):
        y = top + i * row_h
        body.append(
            f'<text x="{left - 18}" y="{y + 17}" text-anchor="end" font-family="{SERIF}" '
            f'font-size="15" fill="{INK}">{name}</text>'
        )
        for index, (value, fill) in enumerate(((random_split, "url(#hatch)"), (grouped, TERRA))):
            by = y + index * 14
            body.append(
                f'<rect x="{left}" y="{by}" height="11" width="0" fill="{fill}" stroke="{INK}" '
                f'stroke-width="0.6">'
                f'<animate attributeName="width" from="0" to="{bar_w * value / axis_max:.1f}" '
                f'dur="1.1s" begin="{0.3 + i * 0.1 + index * 0.07}s" fill="freeze" '
                f'calcMode="spline" keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/></rect>'
                f'<text x="{left + bar_w * value / axis_max + 9:.1f}" y="{by + 10}" '
                f'font-family="{MONO}" font-size="10.5" fill="{INK}" opacity="0">{value}'
                f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
                f'begin="{1.3 + i * 0.1 + index * 0.07}s" fill="freeze"/></text>'
            )
    legend_x = width - 430
    for label, fill in (("random split", "url(#hatch)"), ("unseen postal codes", TERRA)):
        body.append(
            f'<rect x="{legend_x}" y="{height - 34}" width="22" height="10" fill="{fill}" '
            f'stroke="{INK}" stroke-width="0.6"/>'
            f'<text x="{legend_x + 30}" y="{height - 25}" font-family="{MONO}" font-size="10" '
            f'fill="{GRAPHITE}">{label}</text>'
        )
        legend_x += 190
    body.append(
        f'<text x="52" y="{height - 25}" font-family="{SERIF}" font-size="14" fill="{INK}" '
        f'opacity="0">Linear regression loses 105 EUR to unseen postal codes. LightGBM loses 32.'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="2.2s" '
        f'fill="freeze"/></text>'
    )
    return svg(width, height, "".join(body))


# ---------------------------------------------------------------- districts

# MAE in euro on the random test set, from scripts/analyze_errors.py.
DISTRICTS = [
    ("Mitte", 241), ("Prenzlauer Berg", 241), ("Wilmersdorf", 240),
    ("Charlottenburg", 220), ("Tiergarten", 189), ("Friedrichshain", 162),
    ("Kreuzberg", 162), ("Neuk&#246;lln", 141), ("Wedding", 94), ("Spandau", 85),
]


def districts():
    width, height = 1200, 390
    left, top, col_h = 70, 124, 136
    axis_max = 260
    span = width - left - 70
    gap = span / len(DISTRICTS)
    bar_w = gap - 46
    baseline = top + col_h

    body = [
        frame(width, height, "FIGURE 2 &#183; ERROR BY DISTRICT",
              "where the estimate is worst, and by how much", "SHEET 04")
    ]
    body.append(f'<path d="M{left - 12} {baseline} H{width - 58}" stroke="{INK}" '
                f'stroke-width="0.9"/>')
    for i, (name, mae) in enumerate(DISTRICTS):
        x = left + i * gap
        bar_h = col_h * mae / axis_max
        fill = TERRA if i < 4 else ("url(#hatchterra)" if i < 7 else "url(#hatch)")
        body.append(
            f'<rect x="{x:.1f}" y="{baseline}" width="{bar_w:.1f}" height="0" fill="{fill}" '
            f'stroke="{INK}" stroke-width="0.7">'
            f'<animate attributeName="height" from="0" to="{bar_h:.1f}" dur="0.9s" '
            f'begin="{0.3 + i * 0.07}s" fill="freeze" calcMode="spline" '
            f'keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/>'
            f'<animate attributeName="y" from="{baseline}" to="{baseline - bar_h:.1f}" '
            f'dur="0.9s" begin="{0.3 + i * 0.07}s" fill="freeze" calcMode="spline" '
            f'keySplines="0.2 0.8 0.2 1" keyTimes="0;1"/></rect>'
            f'<text x="{x + bar_w / 2:.1f}" y="{baseline - bar_h - 9:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="10.5" fill="{INK}" opacity="0">{mae}'
            f'<animate attributeName="opacity" from="0" to="1" dur="0.4s" '
            f'begin="{1.1 + i * 0.07}s" fill="freeze"/></text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{baseline + 22}" text-anchor="end" '
            f'font-family="{SERIF}" font-size="12.5" fill="{GRAPHITE}" '
            f'transform="rotate(-30 {x + bar_w / 2:.1f} {baseline + 22})">{name}</text>'
        )
    body.append(
        f'<text x="52" y="{height - 28}" font-family="{SERIF}" font-size="14" fill="{INK}" '
        f'opacity="0">Error tracks price level: the four dearest central districts carry about '
        f'2.8 times the error of the cheapest, and large flats are under-priced.'
        f'<animate attributeName="opacity" from="0" to="1" dur="0.6s" begin="1.9s" '
        f'fill="freeze"/></text>'
    )
    return svg(width, height, "".join(body))


if __name__ == "__main__":
    sample, total = load_listings()
    for name, markup in (
        ("hero.svg", hero(sample, total)),
        ("pipeline.svg", pipeline()),
        ("leakage.svg", leakage()),
        ("districts.svg", districts()),
    ):
        (OUT / name).write_text(markup, encoding="utf-8")
        print(f"{name:16} {len(markup) / 1024:6.1f} KB")
