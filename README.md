# interactive_heatmap
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# 1. USER SETTINGS
# ============================================================

INPUT_FILE = "allredgenus_reads_localities_for_heatmap.csv"
OUTPUT_HTML = "interactive_red_algae_heatmap_final.html"

PRESENCE_THRESHOLD = 0


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
    "jeffreysi": "#999999"
}

# Used only if an unexpected subspecies name occurs.
ALTERNATIVE_COLORS = [
    "#984EA3",
    "#4DAF4A",
    "#A65628",
    "#F781BF",
    "#377EB8",
    "#FF1493",
    "#00A6D6",
    "#B15928",
    "#8C564B",
    "#BCBD22",
    "#17BECF",
    "#9467BD"
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_name(value):
    """
    Normalize subspecies names so differences in capitalization,
    spaces, periods, or hyphens do not affect color matching.
    """

    value = str(value).strip().lower()
    value = re.sub(r"[\s.\-]+", "_", value)
    value = re.sub(r"_+", "_", value)

    return value.strip("_")


def consecutive_groups(labels):
    """
    Return consecutive groups as:
    (label, start_position, end_position).

    The end position is excluded.
    """

    if not labels:
        return []

    groups = []
    start = 0
    current_label = labels[0]

    for position in range(1, len(labels)):

        if labels[position] != current_label:

            groups.append(
                (current_label, start, position)
            )

            start = position
            current_label = labels[position]

    groups.append(
        (current_label, start, len(labels))
    )

    return groups


# ============================================================
# 4. READ THE CSV
# ============================================================

raw = pd.read_csv(
    INPUT_FILE,
    sep=";",
    header=None,
    dtype=str,
    encoding="utf-8-sig"
)

# Remove completely empty rows and columns.
raw = raw.dropna(axis=0, how="all")
raw = raw.dropna(axis=1, how="all")

# Remove unnecessary spaces from cells.
raw = raw.apply(
    lambda column: column.str.strip()
    if column.dtype == "object"
    else column
)


# ============================================================
# 5. CHECK THE HEADER
# ============================================================

expected_labels = [
    "Subspecies",
    "Island",
    "Genus"
]

actual_labels = raw.iloc[:3, 0].tolist()

if actual_labels != expected_labels:

    raise ValueError(
        "The first three entries in column 1 must be:\n"
        "Subspecies\n"
        "Island\n"
        "Genus\n\n"
        f"Python found: {actual_labels}"
    )


# ============================================================
# 6. EXTRACT COLUMN METADATA
# ============================================================

subspecies = (
    raw.iloc[0, 1:]
    .astype(str)
    .str.strip()
    .tolist()
)

islands = (
    raw.iloc[1, 1:]
    .astype(str)
    .str.strip()
    .tolist()
)

localities = (
    raw.iloc[2, 1:]
    .astype(str)
    .str.strip()
    .tolist()
)

if not (
    len(subspecies)
    == len(islands)
    == len(localities)
):

    raise ValueError(
        "The Subspecies, Island, and locality rows "
        "must contain the same number of columns."
    )


if len(set(localities)) != len(localities):

    locality_series = pd.Series(localities)

    duplicated = locality_series[
        locality_series.duplicated(keep=False)
    ].unique()

    raise ValueError(
        "Every locality name must be unique. "
        f"Duplicated localities: {list(duplicated)}"
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

    normalized_name = normalize_name(
        subspecies_name
    )

    if normalized_name in normalized_colors:

        actual_subspecies_colors[subspecies_name] = (
            normalized_colors[normalized_name]
        )

    else:

        color_index = (
            len(unknown_subspecies)
            % len(ALTERNATIVE_COLORS)
        )

        actual_subspecies_colors[subspecies_name] = (
            ALTERNATIVE_COLORS[color_index]
        )

        unknown_subspecies.append(
            subspecies_name
        )


print("\nSubspecies colors used:\n")

for name, color in actual_subspecies_colors.items():
    print(f"{name}: {color}")


if unknown_subspecies:

    print(
        "\nWARNING: These subspecies did not match the "
        "provided color keys:"
    )

    for name in unknown_subspecies:

        print(
            f"  {name}: "
            f"{actual_subspecies_colors[name]}"
        )


# ============================================================
# 8. EXTRACT READ COUNTS
# ============================================================

genus_names = (
    raw.iloc[3:, 0]
    .astype(str)
    .str.strip()
)

reads = raw.iloc[3:, 1:].copy()

reads = reads.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

reads.index = genus_names
reads.index.name = "Genus"
reads.columns = localities

# Remove rows without valid genus names.
reads = reads[
    reads.index.notna()
    & (reads.index != "")
    & (reads.index.str.lower() != "nan")
]

if (reads < 0).any().any():

    raise ValueError(
        "Negative read counts were found. "
        "Read counts must be zero or positive."
    )

# Combine duplicated genera.
reads = reads.groupby(
    level=0,
    sort=False
).sum()

# Remove genera with no reads.
reads = reads.loc[
    reads.sum(axis=1) > 0
]

if reads.empty:

    raise ValueError(
        "No positive read counts were found."
    )


# ============================================================
# 9. SORT GENERA
# ============================================================

localities_present = (
    reads > PRESENCE_THRESHOLD
).sum(axis=1)

total_reads = reads.sum(axis=1)

sorting_table = pd.DataFrame(
    {
        "localities_present": localities_present,
        "total_reads": total_reads,
        "genus_name": reads.index
    },
    index=reads.index
)

# Genera found at the most localities appear first.
# Total reads are used as the second sorting criterion.
sorting_table = sorting_table.sort_values(
    by=[
        "localities_present",
        "total_reads",
        "genus_name"
    ],
    ascending=[
        False,
        False,
        True
    ]
)

reads_sorted = reads.loc[
    sorting_table.index
]

print("\nOrder of genera in the heatmap:\n")

print(
    sorting_table[
        [
            "localities_present",
            "total_reads"
        ]
    ].to_string()
)


# ============================================================
# 10. APPLY LOG(READS + 1)
# ============================================================

plot_values = np.log1p(
    reads_sorted
)

maximum_reads = float(
    reads_sorted.to_numpy().max()
)

maximum_log_value = float(
    plot_values.to_numpy().max()
)


# ============================================================
# 11. IDENTIFY CONSECUTIVE GROUPS
# ============================================================

subspecies_groups = consecutive_groups(
    subspecies
)

island_groups = consecutive_groups(
    islands
)


# Warn if an island occurs in separate column blocks.
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

number_of_genera = reads_sorted.shape[0]
number_of_localities = reads_sorted.shape[1]

# Fields:
# 0 = genus
# 1 = locality
# 2 = island
# 3 = subspecies
# 4 = reads
# 5 = number of occupied localities
# 6 = total reads for genus

customdata = np.empty(
    (
        number_of_genera,
        number_of_localities,
        7
    ),
    dtype=object
)

for row_number, genus in enumerate(
    reads_sorted.index
):

    for column_number, locality in enumerate(
        localities
    ):

        customdata[
            row_number,
            column_number
        ] = [
            genus,
            locality,
            islands[column_number],
            subspecies[column_number],
            reads_sorted.iloc[
                row_number,
                column_number
            ],
            sorting_table.loc[
                genus,
                "localities_present"
            ],
            sorting_table.loc[
                genus,
                "total_reads"
            ]
        ]


# ============================================================
# 13. CREATE LOG-SCALE COLOR-BAR TICKS
# ============================================================

# Use intervals of two log units.
regular_log_ticks = np.arange(
    0,
    maximum_log_value,
    2.0
).tolist()

# Add the exact maximum transformed value.
log_colorbar_ticks = regular_log_ticks + [
    maximum_log_value
]

# Remove possible duplicates.
log_colorbar_ticks = sorted(
    set(
        round(value, 10)
        for value in log_colorbar_ticks
    )
)

# Show normal log values, including the exact maximum.
colorbar_tick_labels = []

for value in log_colorbar_ticks:

    if np.isclose(
        value,
        maximum_log_value
    ):

        colorbar_tick_labels.append(
            f"{maximum_log_value:.2f} max"
        )

    else:

        colorbar_tick_labels.append(
            f"{value:g}"
        )


# ============================================================
# 14. CALCULATE FIGURE SIZE AND SPACING
# ============================================================

figure_width = max(
    1500,
    number_of_localities * 32
)

figure_height = max(
    850,
    number_of_genera * 23 + 300
)

# The plotting area excludes the outer margins.
TOP_MARGIN = 45
BOTTOM_MARGIN = 45

paper_height = (
    figure_height
    - TOP_MARGIN
    - BOTTOM_MARGIN
)

# Fixed pixel spacing converted into Plotly paper coordinates.
#
# This prevents spaces from becoming enormous when the
# heatmap contains many genera and the figure becomes tall.

# Space below the heatmap:
ISLAND_LABEL_FROM_BOTTOM_PX = 12
ISLAND_BRACKET_FROM_BOTTOM_PX = 30
ISLAND_BRACKET_HEIGHT_PX = 14
HEATMAP_BOTTOM_FROM_BOTTOM_PX = 155

# Space above the heatmap:
HEATMAP_TOP_FROM_TOP_PX = 82
BAR_GAP_FROM_HEATMAP_PX = 8
SUBSPECIES_BAR_HEIGHT_PX = 20
LABEL_GAP_ABOVE_BAR_PX = 2


ISLAND_LABEL_Y = (
    ISLAND_LABEL_FROM_BOTTOM_PX
    / paper_height
)

ISLAND_BRACKET_Y = (
    ISLAND_BRACKET_FROM_BOTTOM_PX
    / paper_height
)

ISLAND_BRACKET_END_Y = (
    (
        ISLAND_BRACKET_FROM_BOTTOM_PX
        + ISLAND_BRACKET_HEIGHT_PX
    )
    / paper_height
)

HEATMAP_BOTTOM = (
    HEATMAP_BOTTOM_FROM_BOTTOM_PX
    / paper_height
)

HEATMAP_TOP = (
    1
    - HEATMAP_TOP_FROM_TOP_PX
    / paper_height
)

SUBSPECIES_BAR_BOTTOM = (
    HEATMAP_TOP
    + BAR_GAP_FROM_HEATMAP_PX
    / paper_height
)

SUBSPECIES_BAR_TOP = (
    SUBSPECIES_BAR_BOTTOM
    + SUBSPECIES_BAR_HEIGHT_PX
    / paper_height
)

SUBSPECIES_LABEL_Y = (
    SUBSPECIES_BAR_TOP
    + LABEL_GAP_ABOVE_BAR_PX
    / paper_height
)


# ============================================================
# 15. CREATE THE INTERACTIVE HEATMAP
# ============================================================

colorbar_center = (
    HEATMAP_BOTTOM
    + HEATMAP_TOP
) / 2

colorbar_length = (
    HEATMAP_TOP
    - HEATMAP_BOTTOM
)


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
                side="right"
            ),

            tickmode="array",
            tickvals=log_colorbar_ticks,
            ticktext=colorbar_tick_labels,

            y=colorbar_center,
            yanchor="middle",
            len=colorbar_length,

            thickness=18
        ),

        hovertemplate=(
            "<b>Genus:</b> %{customdata[0]}<br>"
            "<b>Locality:</b> %{customdata[1]}<br>"
            "<b>Island:</b> %{customdata[2]}<br>"
            "<b>Subspecies:</b> "
            "<i>%{customdata[3]}</i><br>"
            "<b>Reads:</b> %{customdata[4]:,.0f}<br>"
            "<b>log(reads + 1):</b> %{z:.3f}<br>"
            "<b>Occupied localities:</b> "
            "%{customdata[5]:,.0f}<br>"
            "<b>Total genus reads:</b> "
            "%{customdata[6]:,.0f}"
            "<extra></extra>"
        ),

        hoverongaps=False,

        xgap=0.5,
        ygap=0.5
    )
)


# ============================================================
# 16. FORMAT THE FIGURE
# ============================================================

figure.update_layout(
    title=dict(
        text=(
            "Red-algae read abundance across localities"
            "<br>"
            "<sup>"
            "Genera ordered by occupied localities, "
            "followed by total reads"
            "</sup>"
        ),

        x=0.5,
        xanchor="center",

        y=0.993,
        yanchor="top",

        font=dict(
            size=20
        )
    ),

    width=figure_width,
    height=figure_height,

    xaxis=dict(
        title="",

        tickangle=-55,

        tickfont=dict(
            size=10
        ),

        categoryorder="array",
        categoryarray=localities,

        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="white",
        spikethickness=1,

        fixedrange=False
    ),

    yaxis=dict(
        title="Red-algae genus",

        domain=[
            HEATMAP_BOTTOM,
            HEATMAP_TOP
        ],

        autorange="reversed",

        categoryorder="array",
        categoryarray=reads_sorted.index.tolist(),

        tickfont=dict(
            size=10,
            family="Arial"
        ),

        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="white",
        spikethickness=1,

        fixedrange=False
    ),

    hoverlabel=dict(
        bgcolor="white",
        bordercolor="black",

        font=dict(
            color="black",
            size=13
        )
    ),

    hovermode="closest",

    margin=dict(
        l=190,
        r=150,
        t=TOP_MARGIN,
        b=BOTTOM_MARGIN
    ),

    plot_bgcolor="white",
    paper_bgcolor="white",

    dragmode="zoom",

    showlegend=False
)


# ============================================================
# 17. ADD SUBSPECIES BARS AND LABELS
# ============================================================

for subspecies_name, start, end in subspecies_groups:

    subspecies_color = actual_subspecies_colors[
        subspecies_name
    ]

    group_center = (
        start + end - 1
    ) / 2

    # Thin colored subspecies bar.
    figure.add_shape(
        type="rect",

        xref="x",
        yref="paper",

        x0=start - 0.5,
        x1=end - 0.5,

        y0=SUBSPECIES_BAR_BOTTOM,
        y1=SUBSPECIES_BAR_TOP,

        fillcolor=subspecies_color,

        line=dict(
            color="white",
            width=1
        ),

        layer="above"
    )

    display_subspecies = subspecies_name.replace(
        "_",
        " "
    )

    # Subspecies name immediately above the bar.
    figure.add_annotation(
        x=group_center,
        y=SUBSPECIES_LABEL_Y,

        xref="x",
        yref="paper",

        text=f"<i>{display_subspecies}</i>",

        showarrow=False,

        xanchor="center",
        yanchor="bottom",

        font=dict(
            size=11,
            color="black"
        )
    )


figure.add_annotation(
    x=-0.012,

    y=(
        SUBSPECIES_BAR_BOTTOM
        + SUBSPECIES_BAR_TOP
    ) / 2,

    xref="paper",
    yref="paper",

    text="<b>Subspecies</b>",

    showarrow=False,

    xanchor="right",
    yanchor="middle",

    font=dict(
        size=12,
        color="black"
    )
)


# ============================================================
# 18. ADD BLACK ISLAND BRACKETS
# ============================================================

for island_name, start, end in island_groups:

    group_center = (
        start + end - 1
    ) / 2

    # Horizontal part of the bracket.
    figure.add_shape(
        type="line",

        xref="x",
        yref="paper",

        x0=start - 0.43,
        x1=end - 0.57,

        y0=ISLAND_BRACKET_Y,
        y1=ISLAND_BRACKET_Y,

        line=dict(
            color="black",
            width=2
        ),

        layer="above"
    )

    # Left end of the bracket.
    figure.add_shape(
        type="line",

        xref="x",
        yref="paper",

        x0=start - 0.43,
        x1=start - 0.43,

        y0=ISLAND_BRACKET_Y,
        y1=ISLAND_BRACKET_END_Y,

        line=dict(
            color="black",
            width=2
        ),

        layer="above"
    )

    # Right end of the bracket.
    figure.add_shape(
        type="line",

        xref="x",
        yref="paper",

        x0=end - 0.57,
        x1=end - 0.57,

        y0=ISLAND_BRACKET_Y,
        y1=ISLAND_BRACKET_END_Y,

        line=dict(
            color="black",
            width=2
        ),

        layer="above"
    )

    display_island = island_name.replace(
        "_",
        " "
    )

    figure.add_annotation(
        x=group_center,
        y=ISLAND_LABEL_Y,

        xref="x",
        yref="paper",

        text=f"<b>{display_island}</b>",

        showarrow=False,

        xanchor="center",
        yanchor="top",

        font=dict(
            size=11,
            color="black"
        )
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

    font=dict(
        size=12,
        color="black"
    )
)


# ============================================================
# 19. ADD ISLAND SEPARATION LINES
# ============================================================

for column_number in range(
    1,
    number_of_localities
):

    if (
        islands[column_number]
        != islands[column_number - 1]
    ):

        figure.add_shape(
            type="line",

            xref="x",
            yref="paper",

            x0=column_number - 0.5,
            x1=column_number - 0.5,

            y0=HEATMAP_BOTTOM,
            y1=HEATMAP_TOP,

            line=dict(
                color="white",
                width=3
            ),

            layer="above"
        )


# ============================================================
# 20. SAVE AND DISPLAY
# ============================================================

figure.write_html(
    OUTPUT_HTML,

    # Keep the file usable without an internet connection.
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
            "scale": 2
        }
    }
)

figure.show()

print(
    f"\nInteractive heatmap saved as: {OUTPUT_HTML}"
)

print(
    f"Maximum read count: {maximum_reads:,.0f}"
)

print(
    "Maximum log(reads + 1) value shown on color bar: "
    f"{maximum_log_value:.2f}"
)
