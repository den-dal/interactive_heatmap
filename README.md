import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# 1. USER SETTINGS
# ============================================================

INPUT_FILE = "allredgenus_reads_samples_for_heatmap.csv"
OUTPUT_HTML = "interactive_red_algae_heatmap_26Oct2026.html"

# Existing threshold used to count occupied localities.
PRESENCE_THRESHOLD = 0

# A sample is positive when its genus read count exceeds this.
SAMPLE_PRESENCE_THRESHOLD = 0

GENUS_FONT_SIZE = 12


# ============================================================
# 2. SUBSPECIES COLORS
# ============================================================

SUBSPECIES_COLORS = {
    "sielmanni": "#E41A1C",
    "hayampi": "#E7298A",
    "nanus": "#FFC0CB",
    "cristatus": "#1B9E77",
    "wikelskii": "#00FFFF",
    "hassi": "#6B8E23",
    "trillmichi": "#FFFF33",
    "godzilla": "#0066CC",
    "mertensi": "#66CCFF",
    "venustissimus": "#FF7F00",
    "jeffreysi": "#999999",
}

ALTERNATIVE_COLORS = [
    "#984EA3", "#4DAF4A", "#A65628", "#F781BF",
    "#377EB8", "#FF1493", "#00A6D6", "#B15928",
    "#8C564B", "#BCBD22", "#17BECF", "#9467BD",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_name(value):
    """Normalize subspecies names for color matching."""
    value = str(value).strip().lower()
    value = re.sub(r"[\s.\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def consecutive_groups(labels):
    """Return (label, start, end), with end excluded."""
    if not labels:
        return []

    groups = []
    start = 0
    current_label = labels[0]

    for position in range(1, len(labels)):
        if labels[position] != current_label:
            groups.append((current_label, start, position))
            start = position
            current_label = labels[position]

    groups.append((current_label, start, len(labels)))
    return groups


# ============================================================
# 4. READ THE CSV
# ============================================================

from pathlib import Path

input_path = Path(INPUT_FILE).resolve()

raw = pd.read_csv(
    input_path,
    sep=None,
    engine="python",
    header=None,
    dtype=str,
    encoding="utf-8-sig",
)

raw = raw.apply(lambda column: column.str.strip())
raw = raw.replace(r"^\s*$", np.nan, regex=True)
raw = raw.dropna(axis=0, how="all")
raw = raw.dropna(axis=1, how="all")

print(f"\nFile being opened: {input_path}")
print(f"Rows: {raw.shape[0]}, columns: {raw.shape[1]}")
print("\nFirst five rows and columns:")
print(raw.iloc[:5, :5].to_string(index=False, header=False))

if raw.shape[0] < 5 or raw.shape[1] < 2:
    raise ValueError(
        "The input was not read as a sample-by-genus table. "
        "Check the file path and CSV separator."
    )
# ============================================================
# 5. CHECK THE HEADER
# ============================================================

expected_labels = ["Subspecies", "Island", "Locality", "Genus"]
actual_labels = raw.iloc[:4, 0].tolist()

if actual_labels != expected_labels:
    raise ValueError(
        "The first four entries in column 1 must be:\n"
        "Subspecies\nIsland\nLocality\nGenus\n\n"
        f"Python found: {actual_labels}"
    )


# ============================================================
# 6. EXTRACT AND CHECK SAMPLE METADATA
# ============================================================

metadata = raw.iloc[:4, 1:].T.copy()
metadata.columns = ["Subspecies", "Island", "Locality", "Sample"]

if metadata.isna().any().any():
    raise ValueError(
        "Every sample must have a subspecies, island, "
        "locality, and sample ID."
    )

duplicated_samples = metadata.loc[
    metadata["Sample"].duplicated(keep=False), "Sample"
].unique()

if len(duplicated_samples):
    raise ValueError(
        "Every sample ID must be unique. "
        f"Duplicated IDs: {duplicated_samples.tolist()}"
    )

metadata = metadata.set_index("Sample")

# Each locality must belong to one island and one subspecies.
metadata_counts = metadata.groupby(
    "Locality", sort=False
)[["Island", "Subspecies"]].nunique()

inconsistent_localities = metadata_counts.index[
    metadata_counts.gt(1).any(axis=1)
].tolist()

if inconsistent_localities:
    raise ValueError(
        "Each locality name must identify one island and "
        "one subspecies. Inconsistent localities: "
        f"{inconsistent_localities}. Use distinct locality "
        "names if different sites share the same name."
    )

# Preserve locality order from the input file.
locality_metadata = (
    metadata.drop_duplicates(subset="Locality")
    .set_index("Locality")
)

localities = locality_metadata.index.tolist()
islands = locality_metadata["Island"].tolist()
subspecies = locality_metadata["Subspecies"].tolist()

samples_per_locality = (
    metadata.groupby("Locality", sort=False)
    .size()
    .reindex(localities)
)


# ============================================================
# 7. ASSIGN SUBSPECIES COLORS
# ============================================================

normalized_colors = {
    normalize_name(name): color
    for name, color in SUBSPECIES_COLORS.items()
}

actual_subspecies_colors = {}
unknown_subspecies = []

for subspecies_name in dict.fromkeys(subspecies):
    normalized_name = normalize_name(subspecies_name)

    if normalized_name in normalized_colors:
        color = normalized_colors[normalized_name]
    else:
        color = ALTERNATIVE_COLORS[
            len(unknown_subspecies) % len(ALTERNATIVE_COLORS)
        ]
        unknown_subspecies.append(subspecies_name)

    actual_subspecies_colors[subspecies_name] = color

print("\nSubspecies colors used:\n")
for name, color in actual_subspecies_colors.items():
    print(f"{name}: {color}")

if unknown_subspecies:
    print(
        "\nWARNING: These subspecies did not match "
        "the provided color keys:"
    )
    for name in unknown_subspecies:
        print(f"  {name}: {actual_subspecies_colors[name]}")


# ============================================================
# 8. EXTRACT SAMPLE READS AND AGGREGATE BY LOCALITY
# ============================================================

genus_names = raw.iloc[4:, 0]

if genus_names.isna().any():
    raise ValueError("Every read-count row must have a genus name.")

read_cells = raw.iloc[4:, 1:].copy()
numeric_reads = read_cells.apply(pd.to_numeric, errors="coerce")
values = numeric_reads.to_numpy(dtype=float)

# Reject missing or invalid counts instead of treating them
# as biological absences.
invalid = ~np.isfinite(values)
if invalid.any():
    row_position, column_position = np.argwhere(invalid)[0]
    raise ValueError(
        "Missing or invalid read count for genus "
        f"'{genus_names.iloc[row_position]}', sample "
        f"'{metadata.index[column_position]}': "
        f"{read_cells.iloc[row_position, column_position]!r}. "
        "Use a nonnegative integer, or 0 for absence."
    )

if (values < 0).any() or (values != np.floor(values)).any():
    raise ValueError(
        "Read counts must be nonnegative whole numbers."
    )

sample_reads = numeric_reads.astype("int64")
sample_reads.index = genus_names.tolist()
sample_reads.index.name = "Genus"
sample_reads.columns = metadata.index

# Combine duplicate genus rows before counting positive samples,
# so each sample is counted only once per genus.
sample_reads = sample_reads.groupby(level=0, sort=False).sum()
sample_reads = sample_reads.loc[sample_reads.sum(axis=1) > 0]

if sample_reads.empty:
    raise ValueError("No positive read counts were found.")

sample_localities = metadata["Locality"]

# Sum reads across samples from the same locality.
reads = (
    sample_reads.T
    .groupby(sample_localities, sort=False)
    .sum()
    .T
    .reindex(columns=localities)
)

# Count positive samples independently for each genus/locality.
positive_samples = (
    sample_reads.gt(SAMPLE_PRESENCE_THRESHOLD)
    .T
    .groupby(sample_localities, sort=False)
    .sum()
    .T
    .reindex(columns=localities)
    .astype("int64")
)


# ============================================================
# 9. SORT GENERA
# ============================================================

localities_present = reads.gt(PRESENCE_THRESHOLD).sum(axis=1)
total_reads = reads.sum(axis=1)

sorting_table = pd.DataFrame(
    {
        "localities_present": localities_present,
        "total_reads": total_reads,
        "genus_name": reads.index,
    },
    index=reads.index,
)

sorting_table = sorting_table.sort_values(
    by=["localities_present", "total_reads", "genus_name"],
    ascending=[False, False, True],
)

reads_sorted = reads.loc[sorting_table.index]
positive_samples_sorted = positive_samples.loc[sorting_table.index]

print("\nOrder of genera in the heatmap:\n")
print(
    sorting_table[
        ["localities_present", "total_reads"]
    ].to_string()
)


# ============================================================
# 10. APPLY LOG(READS + 1)
# ============================================================

plot_values = np.log1p(reads_sorted)

maximum_reads = float(reads_sorted.to_numpy().max())
maximum_log_value = float(plot_values.to_numpy().max())


# ============================================================
# 11. IDENTIFY CONSECUTIVE GROUPS
# ============================================================

subspecies_groups = consecutive_groups(subspecies)
island_groups = consecutive_groups(islands)

island_block_counts = {}
for island_name, start, end in island_groups:
    island_block_counts[island_name] = (
        island_block_counts.get(island_name, 0) + 1
    )

for island_name, block_count in island_block_counts.items():
    if block_count > 1:
        print(
            f"Warning: island '{island_name}' appears in "
            f"{block_count} separate column blocks."
        )


# ============================================================
# 12. BUILD HOVER INFORMATION
# ============================================================

number_of_genera, number_of_localities = reads_sorted.shape

# Fields:
# 0 = genus
# 1 = locality
# 2 = island
# 3 = subspecies
# 4 = locality read total
# 5 = number of occupied localities
# 6 = total reads for genus across all localities
# 7 = positive samples for this genus at this locality
# 8 = total input samples at this locality

customdata = np.empty(
    (number_of_genera, number_of_localities, 9),
    dtype=object,
)

# Broadcasting avoids repeated per-cell DataFrame lookups.
customdata[:, :, 0] = np.asarray(reads_sorted.index)[:, None]
customdata[:, :, 1] = np.asarray(localities)[None, :]
customdata[:, :, 2] = np.asarray(islands)[None, :]
customdata[:, :, 3] = np.asarray(subspecies)[None, :]
customdata[:, :, 4] = reads_sorted.to_numpy()
customdata[:, :, 5] = sorting_table[
    "localities_present"
].to_numpy()[:, None]
customdata[:, :, 6] = sorting_table[
    "total_reads"
].to_numpy()[:, None]
customdata[:, :, 7] = positive_samples_sorted.to_numpy()
customdata[:, :, 8] = samples_per_locality.to_numpy()[None, :]


# ============================================================
# 13. CREATE LOG-SCALE COLOR-BAR TICKS
# ============================================================

regular_log_ticks = np.arange(
    0, maximum_log_value, 2.0
).tolist()

log_colorbar_ticks = sorted(
    {
        round(value, 10)
        for value in regular_log_ticks + [maximum_log_value]
    }
)

colorbar_tick_labels = [
    f"{maximum_log_value:.2f} max"
    if np.isclose(value, maximum_log_value)
    else f"{value:g}"
    for value in log_colorbar_ticks
]


# ============================================================
# 14. CALCULATE FIGURE SIZE AND SPACING
# ============================================================

figure_width = max(1500, number_of_localities * 32)
figure_height = max(850, number_of_genera * 23 + 300)

TOP_MARGIN = 45
BOTTOM_MARGIN = 45

paper_height = figure_height - TOP_MARGIN - BOTTOM_MARGIN

# Space below the heatmap.
ISLAND_LABEL_FROM_BOTTOM_PX = 12
ISLAND_BRACKET_FROM_BOTTOM_PX = 30
ISLAND_BRACKET_HEIGHT_PX = 14
HEATMAP_BOTTOM_FROM_BOTTOM_PX = 155

# Space above the heatmap.
HEATMAP_TOP_FROM_TOP_PX = 82
BAR_GAP_FROM_HEATMAP_PX = 8
SUBSPECIES_BAR_HEIGHT_PX = 20
LABEL_GAP_ABOVE_BAR_PX = 2

ISLAND_LABEL_Y = ISLAND_LABEL_FROM_BOTTOM_PX / paper_height
ISLAND_BRACKET_Y = ISLAND_BRACKET_FROM_BOTTOM_PX / paper_height

ISLAND_BRACKET_END_Y = (
    ISLAND_BRACKET_FROM_BOTTOM_PX + ISLAND_BRACKET_HEIGHT_PX
) / paper_height

HEATMAP_BOTTOM = HEATMAP_BOTTOM_FROM_BOTTOM_PX / paper_height
HEATMAP_TOP = 1 - HEATMAP_TOP_FROM_TOP_PX / paper_height

SUBSPECIES_BAR_BOTTOM = (
    HEATMAP_TOP + BAR_GAP_FROM_HEATMAP_PX / paper_height
)
SUBSPECIES_BAR_TOP = (
    SUBSPECIES_BAR_BOTTOM + SUBSPECIES_BAR_HEIGHT_PX / paper_height
)
SUBSPECIES_LABEL_Y = (
    SUBSPECIES_BAR_TOP + LABEL_GAP_ABOVE_BAR_PX / paper_height
)


# ============================================================
# 15. CREATE THE INTERACTIVE HEATMAP
# ============================================================

colorbar_center = (HEATMAP_BOTTOM + HEATMAP_TOP) / 2
colorbar_length = HEATMAP_TOP - HEATMAP_BOTTOM

figure = go.Figure(
    data=go.Heatmap(
        z=plot_values.to_numpy(),
        x=localities,
        y=reads_sorted.index.tolist(),
        customdata=customdata,
        colorscale="Magma",
        zmin=0,
        zmax=maximum_log_value,
        colorbar=dict(
            title=dict(
                text="log(reads + 1)",
                side="right",
            ),
            tickmode="array",
            tickvals=log_colorbar_ticks,
            ticktext=colorbar_tick_labels,
            y=colorbar_center,
            yanchor="middle",
            len=colorbar_length,
            thickness=18,
        ),
        hovertemplate=(
            "<b>Genus:</b> %{customdata[0]}<br>"
            "<b>Locality:</b> %{customdata[1]}<br>"
            "<b>Island:</b> %{customdata[2]}<br>"
            "<b>Subspecies:</b> <i>%{customdata[3]}</i><br>"
            "<b>Reads:</b> %{customdata[4]:,.0f}<br>"
            "<b>Positive samples at locality:</b> "
            "%{customdata[7]:,.0f} / %{customdata[8]:,.0f}<br>"
            "<b>log(reads + 1):</b> %{z:.3f}<br>"
            "<b>Occupied localities:</b> %{customdata[5]:,.0f}<br>"
            "<b>Total genus reads:</b> %{customdata[6]:,.0f}"
            "<extra></extra>"
        ),
        hoverongaps=False,
        xgap=0.5,
        ygap=0.5,
    )
)


# ============================================================
# 16. FORMAT THE FIGURE
# ============================================================

figure.update_layout(
    title=dict(
        text=(
            "Red-algae read abundance across localities"
            "<br><sup>"
            "Genera ordered by occupied localities, "
            "followed by total reads"
            "</sup>"
        ),
        x=0.5,
        xanchor="center",
        y=0.993,
        yanchor="top",
        font=dict(size=20),
    ),
    width=figure_width,
    height=figure_height,
    xaxis=dict(
        title="",
        tickangle=-55,
        tickfont=dict(size=10),
        categoryorder="array",
        categoryarray=localities,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="white",
        spikethickness=1,
        fixedrange=False,
    ),
    yaxis=dict(
        title="Red-algal taxa",
        domain=[HEATMAP_BOTTOM, HEATMAP_TOP],
        autorange="reversed",
        categoryorder="array",
        categoryarray=reads_sorted.index.tolist(),
        tickmode="array",
        tickvals=reads_sorted.index.tolist(),
        ticktext=[f"<i>{genus}</i>" for genus in reads_sorted.index],
        tickfont=dict(
            size=GENUS_FONT_SIZE,
            family="Arial",
        ),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="white",
        spikethickness=1,
        fixedrange=False,
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="black",
        font=dict(color="black", size=13),
    ),
    hovermode="closest",
    margin=dict(
        l=190,
        r=150,
        t=TOP_MARGIN,
        b=BOTTOM_MARGIN,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    dragmode="zoom",
    showlegend=False,
)

# ============================================================
# 17. ADD SUBSPECIES BARS, LABELS, AND CORNER HEADINGS
# ============================================================

for subspecies_name, start, end in subspecies_groups:
    group_center = (start + end - 1) / 2

    figure.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=start - 0.5,
        x1=end - 0.5,
        y0=SUBSPECIES_BAR_BOTTOM,
        y1=SUBSPECIES_BAR_TOP,
        fillcolor=actual_subspecies_colors[subspecies_name],
        line=dict(color="white", width=1),
        layer="above",
    )

    display_subspecies = subspecies_name.replace("_", " ")

    figure.add_annotation(
        x=group_center,
        y=SUBSPECIES_LABEL_Y,
        xref="x",
        yref="paper",
        text=f"<i>{display_subspecies}</i>",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=12, color="black"),
    )

# Remove the vertical axis title; use the corner heading instead.
figure.update_yaxes(title_text="")

# Diagonal divider anchored to the upper-left heatmap corner.
figure.add_shape(
    type="line",
    xref="paper",
    yref="paper",
    xsizemode="pixel",
    ysizemode="pixel",
    xanchor=0,
    yanchor=HEATMAP_TOP,
    x0=-175,
    y0=78,
    x1=-5,
    y1=0,
    line=dict(color="#2a3f5f", width=1.5),
    layer="above",
)

# Heading above/right of the diagonal.
figure.add_annotation(
    x=0,
    y=HEATMAP_TOP,
    xref="paper",
    yref="paper",
    xshift=-8,
    yshift=64,
    text="<b>Marine iguana<br>subspecies</b>",
    showarrow=False,
    xanchor="right",
    yanchor="middle",
    font=dict(size=12, color="#2a3f5f"),
)

# Heading below/left of the diagonal.
figure.add_annotation(
    x=0,
    y=HEATMAP_TOP,
    xref="paper",
    yref="paper",
    xshift=-175,
    yshift=23,
    text="<b>Red algal<br>taxa</b>",
    showarrow=False,
    xanchor="left",
    yanchor="middle",
    align="left",
    font=dict(size=12, color="#2a3f5f"),
)


# ============================================================
# 18. ADD BLACK ISLAND BRACKETS
# ============================================================

for island_name, start, end in island_groups:
    group_center = (start + end - 1) / 2

    # Horizontal part of the bracket.
    figure.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=start - 0.43,
        x1=end - 0.57,
        y0=ISLAND_BRACKET_Y,
        y1=ISLAND_BRACKET_Y,
        line=dict(color="black", width=2),
        layer="above",
    )

    # Left and right ends of the bracket.
    for x_position in (start - 0.43, end - 0.57):
        figure.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x_position,
            x1=x_position,
            y0=ISLAND_BRACKET_Y,
            y1=ISLAND_BRACKET_END_Y,
            line=dict(color="black", width=2),
            layer="above",
        )

    display_island = island_name.replace("_", " ")

    figure.add_annotation(
        x=group_center,
        y=ISLAND_LABEL_Y,
        xref="x",
        yref="paper",
        text=f"<b>{display_island}</b>",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=11, color="black"),
    )

figure.add_annotation(
    x=-0.012,
    y=ISLAND_LABEL_Y,
    xref="paper",
    yref="paper",
    text="<b>Island</b>",
    showarrow=False,
    xanchor="right",
    yanchor="top",
    font=dict(size=12, color="black"),
)


# ============================================================
# 19. ADD ISLAND SEPARATION LINES
# ============================================================

for column_number in range(1, number_of_localities):
    if islands[column_number] != islands[column_number - 1]:
        figure.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=column_number - 0.5,
            x1=column_number - 0.5,
            y0=HEATMAP_BOTTOM,
            y1=HEATMAP_TOP,
            line=dict(color="white", width=3),
            layer="above",
        )

# ============================================================
# 20. SAVE AND DISPLAY
# ============================================================

figure.write_html(
    OUTPUT_HTML,
    include_plotlyjs=True,
    full_html=True,
    config={
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "red_algae_heatmap",
            "height": figure_height,
            "width": figure_width,
            "scale": 2,
        },
    },
)

figure.show()

print(f"\nInteractive heatmap saved as: {OUTPUT_HTML}")
print(f"Maximum read count: {maximum_reads:,.0f}")
print(
    "Maximum log(reads + 1) value shown on color bar: "
    f"{maximum_log_value:.2f}"
)

# Export a high-resolution PNG and a scalable PDF.
figure.write_image(
    "red_algae_heatmap_high_resolution.png",
    width=figure_width,
    height=figure_height,
    scale=4,
)

figure.write_image(
    "red_algae_heatmap.pdf",
    width=figure_width,
    height=figure_height,
)

print("High-resolution PNG and PDF saved.")
