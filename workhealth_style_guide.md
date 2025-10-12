# The Work Health World of Data Visualization 🎨

This style guide provides a precise framework for creating clear, professional, and consistent data visualizations. The core philosophy is **clarity over clutter**, using **meaningful color**, and providing **structured annotation** to guide the reader.

**Table of Contents**

* [Ⅰ. Canvas and Axes](#Ⅰ-canvas-and-axes)
* [Ⅱ. Color Theory: The Generative Palette](#Ⅱ-color-theory-the-generative-palette)
    * [Core Principle](#core-principle)
    * [Color Application Strategy](#color-application-strategy)
* [Ⅲ. Typography and Titling](#Ⅲ-typography-and-titling)
    * [Text Hierarchy](#text-hierarchy)
* [Ⅳ. Data Element Styling](#Ⅳ-data-element-styling)
* [Ⅴ. Legends and Annotation](#Ⅴ-legends-and-annotation)
* [VI. Examples](#vi-examples)
    * [Example 1: Grouped Bar Chart](#example-1-grouped-bar-chart)
    * [Example 2: Scatter Plot](#example-2-scatter-plot)
    * [Example 3: Impact Waterfall Chart](#example-3-impact-waterfall-chart)
    * [Example 4: UMAP Visualization](#example-4-umap-visualization)

## Ⅰ. Canvas and Axes

This section defines the foundational layout of the graph, creating a clean and unobtrusive stage for the data.

  * **Background Color**: The figure and axis backgrounds must be **white** (`#FFFFFF`). This provides maximum contrast for data elements.
  * **Spines (Axis Borders)**: To create a clean, open look, only the essential axes should be visible.
      * The **top and right spines must be removed**.
      * The remaining **left and bottom spines** should be a muted **'grey'**. They define the data space without being visually dominant.
  * **Grid Lines**: Grid lines should be subtle and serve only to guide the eye on the primary quantitative axis (typically the Y-axis).
      * **Axis**: Grid lines are applied to the **Y-axis only**.
      * **Style**: **Dotted** (`:`) or dashed linestyle.
      * **Color**: A light **'grey'**.
      * **Weight**: Thin, with a linewidth of approximately $0.7$ pt.
      * **Position**: The grid must always be drawn **behind** the data elements to avoid visual interference.

-----

## Ⅱ. Color Theory: The Generative Palette

Color should not be merely decorative; it should be an integral part of the data's story. This guide uses a **semantic and generative** approach rather than a fixed palette.

### **Core Principle**

Categories are first grouped by a meaningful quantitative or qualitative metric. Each group is then assigned a distinct, perceptually uniform colormap, creating an intuitive visual language.

### **Color Application Strategy**

1.  **Group Categories Semantically**: Before plotting, group your categorical variables based on a relevant metric (e.g., average value, sentiment score, growth rate). The example script uses three tiers: Positive, Neutral, and Negative.
2.  **Assign Colormaps to Groups**: Assign a distinct `matplotlib` colormap to each semantic group.
      * 👍 **Positive/High-Value Group**: Use a perceptually uniform sequential colormap like `viridis`, `cividis`, or `plasma`.
      * 😐 **Neutral/Baseline Group**: Use the `Greys` colormap.
      * 👎 **Negative/Low-Value Group**: Use a warm-toned sequential colormap like `autumn`, `inferno`, or `magma`. Use the reversed version (e.g., `autumn_r`) if the default map goes from light-to-dark for low-to-high values.
3.  **Sample Colors from Colormaps**: To ensure good visibility and avoid colors that are too light or too dark, sample from the mid-range of each colormap. Do not use the full `0.0` to `1.0` range.
      * **Positive (`viridis`)**: Sample from the $40%$ to $90%$ range: `cm.viridis(np.linspace(0.4, 0.9, num_colors))`
      * **Neutral (`Greys`)**: Sample from the $40%$ to $70%$ range: `cm.Greys(np.linspace(0.4, 0.7, num_colors))`
      * **Negative (`autumn_r`)**: Sample from the $20%$ to $70%$ range: `cm.autumn_r(np.linspace(0.2, 0.7, num_colors))`

This generative method ensures that even with new data, the color scheme remains consistent, meaningful, and accessible.

-----

## Ⅲ. Typography and Titling

Typography establishes a clear visual hierarchy for information, guiding the reader from the main finding to the finer details.

  * **Font Family**: The primary font is **Arial**, with Helvetica and other standard sans-serif fonts as fallbacks.
      * `plt.rcParams['font.family'] = 'sans-serif'`
      * `plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']`

### **Text Hierarchy**

| Element | Font Size | Font Weight | Color | Horizontal Alignment | Placement |
| :--- |:----------| :--- | :--- | :--- | :--- |
| **Main Title** | 20 pt     | **Bold** | Dark Grey (`#333333`) | Left | Top-left of the figure. Include sample size (e.g., `(n=...)`) if relevant. |
| **Subtitle** | 16 pt     | Normal | Grey | Left | Directly below the main title. Used for context or to explain encoding. |
| **Axis Labels** | 12 pt     | **Bold** | Grey | Center | Standard axis label position. |
| **Tick Labels** | 10 pt     | Normal | Grey | Center | On the axes. Y-axis tick *marks* should have a length of 0. |
| **Legend Title** | 12 pt     | **Bold** | Grey | Left | Above the legend items. |
| **Legend Group** | 11 pt     | **Bold** | Dark Grey (`#333333`) | Left | As subheadings within the legend. |
| **Legend Items** | 10 pt     | Normal | Dark Grey (`#333333`) | Left | The individual category labels. |

-----

## Ⅳ. Data Element Styling

This refers to the visual representation of the data itself (e.g., bars, lines, points).

  * **Separation**: When plotting adjacent colored shapes (e.g., bars in a stacked histogram, sections of a treemap), use a **thin, white border** to create clear separation and improve readability.
      * **`edgecolor`**: 'white'
      * **`linewidth`**: `0.5` pt
  * **Ordering**: For stacked or layered plots, data should be ordered logically. The example script orders data by sentiment group (`positive_vars + neutral_vars + negative_vars`), ensuring a consistent visual flow across graphs.

-----

## Ⅴ. Legends and Annotation

For complex visualizations with many categories, a default legend is insufficient. An external, structured legend is required.

  * **Location**: The legend should be placed **outside the plot area** to the right, with subplot parameters adjusted to make room (e.g., `plt.subplots_adjust(right=0.70)`).
  * **Structure**: The legend must be manually built to reflect the semantic grouping established in the Color Theory section.
    1.  **Main Title**: A single title describing the categorical variable.
    2.  **Group Subheadings**: Bolded titles for each semantic group (e.g., "Positive Sentiment").
    3.  **Legend Items**: Each item consists of a colored rectangle and its corresponding label.
  * **Multi-Column Layout**: For a large number of categories, organize the legend into multiple columns to maintain a compact and readable layout.

-----

## VI. Examples

This section provides complete code examples for generating different types of plots that adhere strictly to this style guide.

### Example 1: Grouped Bar Chart

This example shows project performance scores, with projects colored based on their performance tier.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

# --- 1. Adhere to Global Style Settings ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 2. Manual Data Preparation ---
data = {
    'Project': ['Phoenix', 'Orion', 'Pegasus', 'Andromeda', 'Cassiopeia', 'Lynx', 'Draco'],
    'Performance Score': [92, 85, 78, 65, 51, 42, 25],
    'Team': ['Alpha', 'Alpha', 'Bravo', 'Charlie', 'Bravo', 'Alpha', 'Charlie']
}
df = pd.DataFrame(data)

# --- 3. Group Categories Semantically ---
high_perf = df[df['Performance Score'] >= 80]['Project'].tolist()
mid_perf = df[(df['Performance Score'] < 80) & (df['Performance Score'] >= 50)]['Project'].tolist()
low_perf = df[df['Performance Score'] < 50]['Project'].tolist()

# --- 4. Generate Generative Color Palette ---
palette = {}
# Positive/High-Value Group
high_colors = cm.viridis(np.linspace(0.4, 0.9, len(high_perf)))
for project, color in zip(high_perf, high_colors):
    palette[project] = color

# Neutral/Baseline Group
mid_colors = cm.Greys(np.linspace(0.4, 0.7, len(mid_perf)))
for project, color in zip(mid_perf, mid_colors):
    palette[project] = color
    
# Negative/Low-Value Group
low_colors = cm.autumn_r(np.linspace(0.2, 0.7, len(low_perf)))
for project, color in zip(low_perf, low_colors):
    palette[project] = color

# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))
fig.set_facecolor('white')
ax.set_facecolor('white')

# Order data for plotting
df = df.set_index('Project')
ordered_projects = high_perf + mid_perf + low_perf
df = df.loc[ordered_projects]

# Create bars
ax.bar(
    df.index, 
    df['Performance Score'], 
    color=[palette[p] for p in df.index],
    edgecolor='white',
    linewidth=0.5
)

# --- 6. Apply Aesthetics and Titling ---
# Canvas and Axes Styling
ax.grid(axis='y', color='grey', linestyle=':', linewidth=0.7)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_color('grey')
ax.tick_params(axis='x', colors='grey', rotation=45)
ax.tick_params(axis='y', colors='grey', length=0)

# Typography and Text
ax.set_ylabel('Performance Score', fontsize=12, weight='bold', color='grey')
ax.set_xlabel('Project Name', fontsize=12, weight='bold', color='grey')
ax.set_ylim(0, 100) # Set a logical y-limit

fig.text(0.01, 0.98, f'Quarterly Project Performance Review (n={len(df)})',
         fontsize=20, fontweight='bold', ha='left', va='top', color='#333333')
fig.text(0.01, 0.93, 'Projects are colored by performance tier',
         fontsize=14, ha='left', va='top', color='grey')

# --- 7. Build Custom Structured Legend ---
def draw_legend_group(ax, title, var_list, palette, start_x, start_y):
    line_height = 0.04
    group_spacing = 0.04
    y = start_y
    # Group subheading
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight='bold', color='#333333', ha='left', va='top')
    y -= line_height
    # List each variable
    for var in var_list:
        if var in palette:
            rect_height = 0.025
            # ✨ CORRECTED LINE: Center the rectangle vertically with the text
            rect_y = y - (rect_height / 2)

            rect = Rectangle((start_x, rect_y), 0.015, rect_height,
                             facecolor=palette[var], transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            
            fig.text(start_x + 0.02, y, var, transform=ax.transAxes,
                     fontsize=10, color='#333333', ha='left', va='center')
            y -= line_height
    return y

fig.text(1.02, 0.95, 'Projects', transform=ax.transAxes, fontsize=12, weight='bold', color='grey', ha='left', va='top')
y_pos = draw_legend_group(ax, 'High-Performing', high_perf, palette, 1.02, 0.90)
y_pos = draw_legend_group(ax, 'Average-Performing', mid_perf, palette, 1.02, y_pos - 0.02)
draw_legend_group(ax, 'Low-Performing', low_perf, palette, 1.02, y_pos - 0.02)

plt.subplots_adjust(right=0.82, bottom=0.2)
plt.show()

```

### Example 2: Scatter Plot

This example visualizes the relationship between customer acquisition cost and lifetime value, with points colored by the effectiveness of the marketing channel.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D # Needed for scatter plot legend

# --- 1. Adhere to Global Style Settings ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# --- 2. Manual Data Preparation ---
np.random.seed(42)
data = {
    'Channel': ['SEO', 'PPC', 'Referral', 'Social', 'Email', 'Affiliate', 'Display', 'Organic Social'] * 10,
    'Acquisition Cost': np.random.rand(80) * 100 + 20,
    'Lifetime Value': np.random.rand(80) * 500 + 50,
}
df = pd.DataFrame(data)
# Adjust LTV based on channel to create semantic meaning
channel_multiplier = {'SEO': 4.5, 'Email': 4.2, 'Referral': 4.0, 'PPC': 2.5, 'Affiliate': 2.2, 'Social': 1.5, 'Organic Social': 1.8, 'Display': 1.1}
df['Lifetime Value'] = df['Lifetime Value'] * df['Channel'].map(channel_multiplier)
df['Acquisition Cost'] += df['Channel'].map(channel_multiplier) * np.random.randint(-5, 5)


# --- 3. Group Categories Semantically ---
high_roi_channels = ['SEO', 'Email', 'Referral']
mid_roi_channels = ['PPC', 'Affiliate', 'Organic Social']
low_roi_channels = ['Social', 'Display']

# --- 4. Generate Generative Color Palette ---
palette = {}
high_colors = cm.viridis(np.linspace(0.4, 0.9, len(high_roi_channels)))
for channel, color in zip(high_roi_channels, high_colors):
    palette[channel] = color

mid_colors = cm.Greys(np.linspace(0.4, 0.7, len(mid_roi_channels)))
for channel, color in zip(mid_roi_channels, mid_colors):
    palette[channel] = color

low_colors = cm.autumn_r(np.linspace(0.2, 0.7, len(low_roi_channels)))
for channel, color in zip(low_roi_channels, low_colors):
    palette[channel] = color

# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))
fig.set_facecolor('white')
ax.set_facecolor('white')

# Plot each group to ensure correct coloring
for channel, group_df in df.groupby('Channel'):
    ax.scatter(
        group_df['Acquisition Cost'],
        group_df['Lifetime Value'],
        color=palette[channel],
        alpha=0.8,
        s=50, # size of points
        edgecolor='white',
        linewidth=0.5
    )


# --- 6. Apply Aesthetics and Titling ---
ax.grid(axis='y', color='grey', linestyle=':', linewidth=0.7)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_color('grey')
ax.tick_params(axis='x', colors='grey')
ax.tick_params(axis='y', colors='grey', length=0)

ax.set_xlabel('Acquisition Cost ($)', fontsize=12, weight='bold', color='grey')
ax.set_ylabel('Customer Lifetime Value ($)', fontsize=12, weight='bold', color='grey')

fig.text(0.01, 0.98, f'Marketing Channel Effectiveness (n={len(df)})',
         fontsize=20, fontweight='bold', ha='left', va='top', color='#333333')
fig.text(0.01, 0.93, 'Channels colored by general return on investment (ROI)',
         fontsize=14, ha='left', va='top', color='grey')


# --- 7. Build Custom Structured Legend for Scatter ---
def draw_scatter_legend(ax, title, var_list, palette, start_x, start_y):
    line_height = 0.04
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes, fontsize=11, weight='bold', color='#333333', ha='left', va='top')
    y -= line_height
    for var in var_list:
        if var in palette:
            # Create a proxy artist for the legend item
            legend_artist = Line2D([0], [0], marker='o', color='w', label=var,
                                   markerfacecolor=palette[var], markersize=8, markeredgecolor='grey', markeredgewidth=0.5)
            # Manually place the artist and text
            ax.add_line(Line2D([start_x], [y-0.01], transform=ax.transAxes, marker='o', color='w', markerfacecolor=palette[var], markersize=8, markeredgecolor='grey'))
            fig.text(start_x + 0.02, y, var, transform=ax.transAxes, fontsize=10, color='#333333', ha='left', va='center')
            y -= line_height
    return y

fig.text(1.02, 0.95, 'Marketing Channel', transform=ax.transAxes, fontsize=12, weight='bold', color='grey', ha='left', va='top')
y_pos = draw_scatter_legend(ax, 'High ROI', high_roi_channels, palette, 1.02, 0.90)
y_pos = draw_scatter_legend(ax, 'Medium ROI', mid_roi_channels, palette, 1.02, y_pos - 0.02)
draw_scatter_legend(ax, 'Low ROI', low_roi_channels, palette, 1.02, y_pos - 0.02)


plt.subplots_adjust(right=0.82)
plt.show()

```

### Example 3: Impact Waterfall Chart

This example visualizes the impact of shifts in category for similar variables on some outcome of interest.

```python
# ─────────────────────────────  WATERFALL EXAMPLE  ─────────────────────────────
# Creates a 3-panel waterfall chart (mastery / mattering / esteem) with fake data
# --------------------------------------------------------------------------------

import matplotlib.pyplot as plt, matplotlib.cm as cm
import pandas as pd, numpy as np, textwrap
from matplotlib.ticker import FormatStrFormatter

# ─── SETTINGS ─────────────────────────────────────────────────────────────
CONSTRUCTS = ['mastery', 'mattering', 'esteem']
LABEL_PAD_FRAC, BUFFER_FRAC = 0.04, 0.20
COLOR_POS, COLOR_NEG = '#006400', '#00008B'
MIN_BAR_HT = 1e-4

# Fake categories and sample data
categories = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

# Randomly generated "mean scores" for demonstration
np.random.seed(42)
mean_scores = pd.DataFrame({
    "mastery":   np.linspace(2.0, 4.2, len(categories)) + np.random.normal(0, 0.1, len(categories)),
    "mattering": np.linspace(1.8, 4.0, len(categories)) + np.random.normal(0, 0.1, len(categories)),
    "esteem":    np.linspace(2.3, 4.4, len(categories)) + np.random.normal(0, 0.1, len(categories))
}, index=categories)

# Marginal change between adjacent categories
marginal_change = mean_scores.diff().dropna()

# ─── HELPER: Waterfall Plot ───────────────────────────────────────────────
def plot_waterfall(ax, labels, deltas, start_value,
                   v_max, v_min, pad_units,
                   draw_trend=True, smooth=True):
    cumulative = start_value
    y_tops = [start_value]
    pos_cmap, neg_cmap = cm.get_cmap('Greens'), cm.get_cmap('Blues_r')

    for i, d in enumerate(deltas):
        is_pos = d >= 0
        norm_val = (d / (v_max if is_pos else v_min)) if (v_max or v_min) else 1
        bar_col = (pos_cmap if is_pos else neg_cmap)(0.3 + 0.6 * abs(norm_val))

        ax.bar(i, d, bottom=cumulative, color=bar_col,
               edgecolor='black', linewidth=1.0, width=0.7, zorder=10)

        # % change label
        pct = 0 if abs(cumulative) < MIN_BAR_HT else 100 * d / cumulative
        lbl_txt = f"(+{pct:.0f}%)" if d >= 0 else f"({pct:.0f}%)"
        lbl_y = cumulative + d + pad_units * (1 if d >= 0 else -1)

        ax.text(i, lbl_y, lbl_txt,
                ha='center',
                va='bottom' if d >= 0 else 'top',
                fontsize=9,
                color=COLOR_POS if d >= 0 else COLOR_NEG,
                fontweight='bold', zorder=11)

        cumulative += d
        y_tops.append(cumulative)

    # Add horizontal caps
    for i in range(len(y_tops) - 1):
        ax.plot([i-0.35, i+0.35], [y_tops[i], y_tops[i]],
                color='grey', linestyle=':', linewidth=1, zorder=5)

    # Add smoothed trend line
    if draw_trend:
        mid_x = np.arange(len(deltas))
        mid_y = (np.array(y_tops[:-1]) + np.array(y_tops[1:])) / 2
        try:
            from scipy.interpolate import make_interp_spline
            x_s = np.linspace(mid_x.min(), mid_x.max(), 200) if smooth else mid_x
            y_s = make_interp_spline(mid_x, mid_y, k=2)(x_s) if smooth else mid_y
        except ModuleNotFoundError:
            x_s, y_s = mid_x, mid_y
        ax.plot(x_s, y_s, color='black', linewidth=2, zorder=12)
        ax.annotate("", xy=(x_s[-1], y_s[-1]), xytext=(x_s[-2], y_s[-2]),
                    arrowprops=dict(arrowstyle='-|>', color='black',
                                    linewidth=2, shrinkA=0, shrinkB=0),
                    zorder=13)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right',
                       fontsize=10, color='grey')
    ax.grid(axis='y', color='grey', linestyle=':', linewidth=0.7, zorder=0)
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color('black')
    ax.tick_params(axis='y', length=0, labelsize=10, colors='grey')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# ─── PLOT MAIN WATERFALL DASHBOARD ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.set_facecolor("white")

fig.text(0.02, 0.98, 'Impact Waterfall Example (Fake Data)',
         fontsize=20, fontweight="bold", ha="left", va="top")
fig.text(0.02, 0.93, textwrap.fill("This is a sample waterfall impact chart with mastery, mattering, and esteem constructs.", width=100),
         fontsize=18, color="grey", style="italic", ha="left", va="top")

# Determine global span for y-axis padding
spans = {}
for cons in CONSTRUCTS:
    cum = mean_scores.iloc[0][cons]
    lo, hi = cum, cum
    for d in marginal_change[cons]:
        cum += d
        lo, hi = min(lo, cum), max(hi, cum)
    spans[cons] = (lo, hi)

widest_span = max(hi - lo for lo, hi in spans.values())
buffer = BUFFER_FRAC * widest_span
pad_units = LABEL_PAD_FRAC * (widest_span + buffer)

v_max = marginal_change[marginal_change > 0].max().max() or 0
v_min = marginal_change[marginal_change < 0].min().min() or 0
transition_lbls = [f"{p} → {n}" for p, n in zip(categories[:-1], categories[1:])]

# Draw panels
for ax, cons in zip(axes, CONSTRUCTS):
    start_val = mean_scores.loc[categories[0], cons]
    deltas = marginal_change[cons].loc[categories[1:]].values

    plot_waterfall(ax, transition_lbls, deltas, start_val,
                   v_max, v_min, pad_units)

    lo, hi = spans[cons]
    mid = 0.5 * (lo + hi)
    ax.set_ylim(mid - widest_span/2 - buffer,
                mid + widest_span/2 + buffer)
    ax.set_title(cons.capitalize(), fontsize=14, fontweight="bold")

axes[0].set_ylabel("Cumulative Mean Score", fontsize=12,
                   weight="bold", color="grey")
plt.tight_layout(rect=[0.02, 0.02, 1, 0.88])
plt.show()
```
### Example 4: UMAP Visualization

This example visualizes UMAP embeddings of survey data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap as umap
from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import itertools

# ### 1. Global Styling & Visualization Settings ###
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# --- ⭐️ Key User Setting for Clarity ---
# Set the max number of active variable labels to show to prevent clutter
N_ACTIVE_LABELS_TO_SHOW = 15

# ### 2. Synthetic Data Generation ###
def create_synthetic_survey_data(n_samples=1500, n_active=25, n_sup=4):
    """Generates a synthetic DataFrame mimicking survey data."""
    data = {}
    active_vars = [f"active_var_{i+1}" for i in range(n_active)]
    for i, var in enumerate(active_vars):
        p = np.random.dirichlet(np.ones(5) * (i % 4 + 1))
        data[var] = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=p)

    sup_vars = [f"construct_{i+1}" for i in range(n_sup)]
    for var in sup_vars:
        data[var] = np.random.randn(n_samples) * 10 + 50

    df = pd.DataFrame(data)
    print(f"✅ Generated synthetic data with {n_samples} samples.")
    return df, active_vars, sup_vars

# ### 3. Data Preparation and UMAP Processing ###

# --- 1. GENERATE DATA ---
df, active_vars, sup_vars = create_synthetic_survey_data()
n_cleaned = len(df)

# --- 2. PREPARE SUPPLEMENTARY VARIABLES ---
sup_labels = {}
for con in sup_vars:
    tertile_col = f"{con}_tertile"
    df[tertile_col] = pd.qcut(
        df[con].rank(method='first'), q=3,
        labels=[f"Low {con.replace('_', ' ').title()}", f"Avg {con.replace('_', ' ').title()}", f"High {con.replace('_', ' ').title()}"]
    )
    sup_labels[con] = list(df[tertile_col].cat.categories)

# --- 3. ONE-HOT ENCODE DATA ---
active_df_encoded = pd.get_dummies(df[active_vars].astype('category'), dummy_na=False, dtype=float)
sup_df_encoded_list = [pd.get_dummies(df[f"{con}_tertile"], dtype=float) for con in sup_vars]
sup_df_encoded = pd.concat(sup_df_encoded_list, axis=1)

# --- 4. FIT UMAP & CALCULATE COORDINATES ---
print("🔬 Fitting UMAP with parameters optimized for visual spread...")
# ✨ IMPROVEMENT: Adjust min_dist and spread to create a less clumped layout
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    metric='jaccard',
    n_neighbors=15,
    min_dist=0.4, # Pushes points further apart
    spread=1.5    # Expands the embedding space
)
row_coords = pd.DataFrame(reducer.fit_transform(active_df_encoded), columns=['x', 'y'], index=active_df_encoded.index)

# Calculate coordinates for supplementary & active variables
sup_coords_list = []
for col in sup_df_encoded.columns:
    cat_coords = row_coords[sup_df_encoded[col].astype(bool)].mean()
    cat_coords.name = col
    sup_coords_list.append(cat_coords)
sup_coords = pd.DataFrame(sup_coords_list)

active_coords_list = []
for col in active_vars:
    weighted_coords = row_coords.multiply(df[col], axis=0).sum() / df[col].sum()
    weighted_coords.name = col
    active_coords_list.append(weighted_coords)
active_coords = pd.DataFrame(active_coords_list)

# ### 4. Visualization with Custom Legend ###

# --- 1. SETUP PLOT & COLORS ---
fig, ax = plt.subplots(figsize=(16, 14))

# ✨ IMPROVEMENT: Use a distinct, colorblind-friendly color palette
distinct_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
color_cycle = itertools.cycle(distinct_colors)
construct_colors = {con: next(color_cycle) for con in sup_vars}
legend_palette = {}

# --- 2. PLOT DATA POINTS ---
ax.scatter(row_coords['x'], row_coords['y'], s=5, color='grey', alpha=0.1, zorder=1)
ax.scatter(active_coords['x'], active_coords['y'], s=60, color='#6C757D', alpha=0.5, zorder=2) # Muted active color

for con in sup_vars:
    labels_for_construct = sup_labels[con]
    coords_for_construct = sup_coords.loc[labels_for_construct]
    color = construct_colors[con]
    ax.scatter(
        coords_for_construct['x'], coords_for_construct['y'], s=220,
        color=color, ec='black', lw=0.7, zorder=3
    )
    for label in labels_for_construct:
        legend_palette[label] = color

# --- 3. ADD TEXT LABELS ---
texts = []

# ✨ IMPROVEMENT: Select only the most peripheral active variables to label
active_coords['radius'] = np.sqrt(active_coords['x']**2 + active_coords['y']**2)
active_labels_to_plot = set(active_coords.nlargest(N_ACTIVE_LABELS_TO_SHOW, 'radius').index)

all_sup_labels_flat = [lbl for sublist in sup_labels.values() for lbl in sublist]
all_coords = pd.concat([active_coords, sup_coords])

for label_name, row in all_coords.iterrows():
    is_supplementary = label_name in all_sup_labels_flat
    is_active_to_plot = label_name in active_labels_to_plot

    if is_supplementary or is_active_to_plot:
        display_text = str(label_name).replace('_', ' ').title()
        color = '#343A40' # Default text color
        if is_supplementary:
            color = legend_palette.get(label_name, color)

        fontsize = 12 if is_supplementary else 9.5
        txt = ax.text(row['x'], row['y'], display_text, weight='bold', color=color, fontsize=fontsize, zorder=4)
        txt.set_path_effects([patheffects.Stroke(linewidth=4, foreground='white', alpha=0.8), patheffects.Normal()])
        texts.append(txt)

# ✨ IMPROVEMENT: Tweak arrowprops as suggested by the `adjust_text` warning
adjust_text(texts, ax=ax,
            force_points=(1.5, 1.5), force_text=(1.5, 1.5),
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.7, shrinkA=5, shrinkB=5))

# --- 4. AESTHETIC STYLING ---
ax.set_xlabel("UMAP Dimension 1", weight="bold", fontsize=12, color='grey')
ax.set_ylabel("UMAP Dimension 2", weight="bold", fontsize=12, color='grey')
ax.axhline(0, color="grey", lw=0.5, linestyle="--")
ax.axvline(0, color="grey", lw=0.5, linestyle="--")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_color('grey')
ax.tick_params(axis='both', colors='grey', length=0)
ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='lightgrey')

# --- 5. TITLES ---
fig.text(0.01, 0.98, f'UMAP of Synthetic Survey Responses (n={n_cleaned})',
         fontsize=22, fontweight='bold', ha='left', va='top', color='#333333')
fig.text(0.01, 0.94, "With Psychological Constructs as Supplementary Variables",
         fontsize=16, ha='left', va='top', color='grey')

# ### 5. Build Custom Structured Legend ###
def draw_legend_group(ax, title, var_list, palette, start_x, start_y):
    line_height = 0.04
    group_spacing = 0.04
    y = start_y
    # Group subheading
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight='bold', color='#333333', ha='left', va='top')
    y -= line_height
    # List each variable
    for var in var_list:
        if var in palette:
            rect_height = 0.025
            # ✨ CORRECTED LINE: Center the rectangle vertically with the text
            rect_y = y - (rect_height / 2)

            rect = Rectangle((start_x, rect_y), 0.015, rect_height,
                             facecolor=palette[var], transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)
            
            fig.text(start_x + 0.02, y, var, transform=ax.transAxes,
                     fontsize=10, color='#333333', ha='left', va='center')
            y -= line_height
    return y

fig.text(1.01, 0.90, 'Supplementary Constructs', transform=ax.transAxes,
         fontsize=12, weight='bold', color='grey', ha='left', va='top')

y_pos = 0.86
for con in sup_vars:
    title = con.replace('_', ' ').title()
    var_list = sup_labels[con]
    y_pos = draw_legend_group(ax, title, var_list, legend_palette, 1.01, y_pos)
    # ✨ IMPROVEMENT: Increased spacing between legend groups
    y_pos -= 0.04

plt.subplots_adjust(right=0.82, top=0.92)
plt.show()
```
