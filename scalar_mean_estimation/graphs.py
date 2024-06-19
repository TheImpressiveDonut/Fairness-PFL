import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, rgb2hex


def plot_corr(x, y, x_label, y_label, title, save_path=None, hue_control=None, hue_label=None):
    runs = [f"Run {i + 1}" for i in range(len(x))]

    # Combine the data into a single DataFrame
    if hue_control is not None:
        data = {'run': [], 'x': [], 'y': []}
        for i, (x_vals, y_vals, h_vals) in enumerate(zip(x, y, hue_control)):
            data['x'].extend(x_vals)
            data['y'].extend(y_vals)
            data['run'].extend(h_vals)

        if hue_label == "Task Difficulty":
            colors = plt.cm.Blues(np.linspace(0.2, 1, 256))  # Avoid the lightest colors
            custom_palette = LinearSegmentedColormap.from_list("CustomBlues", colors)
        elif hue_label == "True Mean Absolute Value":
            colors = plt.cm.Reds(np.linspace(0.3, 1, 256))  # Avoid the lightest and brightest colors
            custom_palette = LinearSegmentedColormap.from_list("CustomReds", colors)
        else:
            raise NotImplementedError(hue_label)
    else:
        data = {'run': [], 'x': [], 'y': []}
        for i, (x_vals, y_vals) in enumerate(zip(x, y)):
            data['run'].extend([runs[i]] * x_vals.shape[0])
            data['x'].extend(x_vals)
            data['y'].extend(y_vals)

        custom_palette = [
            '#21618C', '#2E86C1',
            '#5DADE2', '#AED6F1',
            '#E74C3C', '#A93226',
            '#7B241C', '#CB4335',
            '#F5B7B1', '#2C3E50'
        ]



        custom_palette = sns.color_palette("Set2", len(x))

        custom_palette = [
            "#729EA1", "#486A7A", "#A1665E", "#2C3E50", "#6C757D",
            "#495057", "#A9A9A9", "#B0C4DE", "#4682B4", "#5F9EA0",
            "#556B2F", "#6B8E23", "#808000", "#B8860B", "#D2691E",
            "#CD5C5C", "#E9967A", "#8B4513", "#A0522D", "#BC8F8F"
        ]

        greens = plt.cm.Greens(np.linspace(0.2, 0.5, 7))  # Avoid the lightest and brightest colors
        blues = plt.cm.Blues(np.linspace(0.2, 0.5, 7))  # Avoid the lightest and brightest colors
        reds = plt.cm.Reds(np.linspace(0.2, 0.5, 6))  # Avoid the lightest and brightest colors

        # Combine these palettes into a single custom palette
        colors = np.vstack([greens, blues, reds])
        custom_palette = [rgb2hex(color) for color in colors]

        custom_palette = [
            "#6C757D", "#495057", "#343A40",  # Greys
            "#729EA1", "#486A7A", "#2C3E50",  # Blues
            "#556B2F", "#6B8E23", "#808000",  # Greens
            "#A0522D", "#BC8F8F", "#8B4513"  # Reds/Browns
        ]

        custom_palette = [
            '#1B4F72', '#21618C', '#2874A6', '#2E86C1', '#3498DB',
            '#5DADE2', '#85C1E9', '#AED6F1', '#D6EAF8', '#E74C3C',
            '#C0392B', '#A93226', '#922B21', '#7B241C', '#641E16',
            '#CB4335', '#F1948A', '#F5B7B1', '#FADBD8', '#2C3E50'
        ]

        custom_palette = [
            "#6C757D", "#495057", "#343A40", "#ADB5BD", "#CED4DA",  # Greys
            "#729EA1", "#486A7A", "#2C3E50", "#8FA8B9", "#B0C4DE",  # Blues
            "#708090", "#778899", "#A9A9A9", "#B7C3A7", "#A9B7A0",  # Muted Greens
            "#A0522D", "#BC8F8F", "#8B4513", "#A6806A", "#B59386"  # Reds/Browns
        ]

    df = pd.DataFrame(data)
    sns.set(style="white")

    plt.figure(figsize=(12, 8))
    scatter_plot = sns.scatterplot(x='x', y='y', hue='run', palette=custom_palette, data=df, s=100, legend=False)

    m, b = np.polyfit(df['x'], df['y'], 1)
    plt.plot(df['x'], m * df['x'] + b, color='#E74C3C', linewidth=2, label='Global Trend Line')

    plt.xlabel(x_label, fontsize=12, color='#2C3E50')
    plt.ylabel(y_label, fontsize=12, color='#2C3E50')
    plt.title(title, fontsize=14, color='#2C3E50')
    if hue_control is not None:
        plt.legend(loc="best")
        norm = plt.Normalize(df['run'].min(), df['run'].max())
        sm = plt.cm.ScalarMappable(cmap=custom_palette, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=scatter_plot.axes)
        cbar.set_label(hue_label, fontsize=12, color='#2C3E50')
    else:
        plt.legend(loc="best")
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
