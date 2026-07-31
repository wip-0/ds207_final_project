"""Function Helpers for Tensroflow/Keras architecture visualizations.

The functions in this module keep notebooks small. A notebook only needs to
provide model paths and a few model-specific choices such as titles, selected
layer indices, graph stages, and legend categories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import keras
import pandas as pd
import visualkeras
from PIL import Image, ImageDraw, ImageFont


# One shared palette keeps all model figures visually consistent.
DEFAULT_PALETTE = {
    "input": "#4E79A7",
    "encoding": "#76B7B2",
    "merge": "#EDC948",
    "convolution": "#E15759",
    "transform": "#B07AA1",
    "regularization": "#9C755F",
    "pooling": "#59A14F",
    "dense": "#F28E2B",
    "outline": "#243B53",
    "text": "#1F2937",
    "connector": "#6B7280",
}

LEGEND_LABELS = {
    "input": "Model inputs",
    "encoding": "Categorical encoding",
    "merge": "Merge / reshape",
    "convolution": "Convolution",
    "transform": "Normalization / activation",
    "regularization": "Dropout",
    "pooling": "Pooling",
    "dense": "Dense / output",
}


def find_project_root(start: str | Path | None = None) -> Path:
    """
    Finds the root directory of a project containing a ``pyproject.toml`` file.

    This function traverses the directory tree upwards from the given starting
    directory or the current working directory until it encounters a directory
    containing a ``pyproject.toml`` file. If no such directory is found, a
    ``FileNotFoundError`` will be raised.

    :param start: Optional starting path. If ``None``, the current working directory
                  is used.
    :type start: str | Path | None
    :return: The path to the root directory containing the ``pyproject.toml`` file.
    :rtype: Path
    :raises FileNotFoundError: If no directory containing a ``pyproject.toml`` file
                               is found.
    """
    current = Path.cwd() if start is None else Path(start)
    for directory in (current.resolve(), *current.resolve().parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError("Could not find a project root containing pyproject.toml")


def load_keras_models(
    model_paths: Mapping[str, str | Path],
    *,
    compile_models: bool = False,
    safe_mode: bool = True,
) -> dict[str, keras.Model]:
    """
    Loads Keras models from the specified file paths and returns them as a dictionary.

    This function resolves the file paths provided in `model_paths`, checks whether all
    specified model files exist, and raises an error if any are missing. It then loads each
    Keras model from its file path using the `keras.models.load_model` method, applying
    optional compilation and safe mode settings.

    :param model_paths: A mapping of model names to their corresponding file paths. Keys
        represent the names of the models, and values are the file paths where the models
        are stored.
    :param compile_models: Determines whether the models should be compiled after loading.
        Defaults to False.
    :param safe_mode: Indicates whether to enable safe loading mode, which restricts
        loading certain custom objects or configurations. Defaults to True.
    :return: A dictionary where the keys are the model names and the values are the loaded
        Keras model instances.
    :raises FileNotFoundError: If any of the provided file paths do not exist.
    """
    resolved_paths = {name: Path(path) for name, path in model_paths.items()}
    missing = [str(path) for path in resolved_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))

    return {
        name: keras.models.load_model(
            path,
            compile=compile_models,
            safe_mode=safe_mode,
        )
        for name, path in resolved_paths.items()
    }


def semantic_legend(
    categories: Sequence[str] | None = None,
    *,
    palette: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """
    Generates a semantic legend from given categories and their palette mappings.

    This function produces a list of tuples, where each tuple represents a legend
    entry consisting of a label and its corresponding color code. If no categories
    are provided, a default set of legend labels is used. If a palette is not
    provided, the default palette is utilized. An exception is raised if any of
    the specified categories are not present in the predefined legend labels.

    :param categories:
        A sequence of category names for which the semantic legend should be
        generated. If None, a predefined set of categories is used.
    :param palette:
        A mapping between category names and their respective color codes.
        If None, a default palette is used.
    :return:
        A list of tuples, where each tuple contains a legend label and its
        corresponding color.
    :rtype:
        list[tuple[str, str]]
    :raises ValueError:
        If any provided category names are not present in the predefined
        legend labels.
    """
    colors = DEFAULT_PALETTE if palette is None else palette
    selected = tuple(LEGEND_LABELS) if categories is None else tuple(categories)
    unknown = [category for category in selected if category not in LEGEND_LABELS]
    if unknown:
        raise ValueError(f"Unknown legend categories: {unknown}")
    return [(LEGEND_LABELS[category], colors[category]) for category in selected]


def build_layer_color_map(
    palette: Mapping[str, str] | None = None,
) -> dict[type, dict[str, str]]:
    """Map Keras layer classes to the shared semantic colors."""
    colors = DEFAULT_PALETTE if palette is None else palette

    def style(fill: str) -> dict[str, str]:
        return {"fill": fill, "outline": colors["outline"]}

    return {
        keras.layers.InputLayer: style(colors["input"]),
        keras.layers.StringLookup: style(colors["encoding"]),
        keras.layers.Embedding: style(colors["encoding"]),
        keras.layers.Flatten: style(colors["encoding"]),
        keras.layers.Concatenate: style(colors["merge"]),
        keras.layers.Reshape: style(colors["merge"]),
        keras.layers.Conv1D: style(colors["convolution"]),
        keras.layers.BatchNormalization: style(colors["transform"]),
        keras.layers.LeakyReLU: style(colors["transform"]),
        keras.layers.Dropout: style(colors["regularization"]),
        keras.layers.AveragePooling1D: style(colors["pooling"]),
        keras.layers.GlobalAveragePooling1D: style(colors["pooling"]),
        keras.layers.GlobalMaxPooling1D: style(colors["pooling"]),
        keras.layers.Dense: style(colors["dense"]),
    }


def publication_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load a readable font on macOS, Linux, or Windows."""
    candidates = [
        Path(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def add_visualkeras_output_shapes(model: keras.Model) -> keras.Model:
    """
    Add output shape information to layers in a Keras model for improved visualization.

    This function iterates over each layer in the provided Keras model and ensures
    that layers without an `output_shape` attribute, but with an `output` attribute,
    are assigned an `output_shape`. If the `output` is a sequence (list or tuple),
    only the first element is considered when determining the output shape.

    :param model: A Keras model whose layers' output shapes should be updated.
    :type model: keras.Model
    :return: The Keras model with updated output shape information added to its layers.
    :rtype: keras.Model
    """
    for layer in model.layers:
        if not hasattr(layer, "output_shape") and hasattr(layer, "output"):
            output = layer.output
            if isinstance(output, (list, tuple)):
                output = output[0]
            layer.output_shape = tuple(output.shape)
    return model


def compact_shape(layer: keras.layers.Layer) -> str:
    """
    Generates a compact string representation of the tensor shape for a given Keras layer.

    This function extracts the output shape of the provided Keras layer, removes the
    batch size dimension, and concatenates the remaining dimensions with the symbol '×'.
    If the shape is undefined or incomplete, a placeholder is returned.

    :param layer: A Keras layer whose output shape is to be processed.
    :type layer: keras.layers.Layer
    :return: A string representing the compact tensor shape, or "?" if the shape is
        undefined or incomplete.
    :rtype: str
    """
    dimensions = [
        dimension
        for dimension in getattr(layer, "output_shape", ())[1:]
        if dimension is not None
    ]
    return " × ".join(map(str, dimensions)) or "?"


def make_layer_labeler(
    label_overrides: Mapping[str, str] | None = None,
):
    """
    Creates a labeling function for Keras layers with optional overrides for specific layer
    names. The returned function generates descriptive text labels for each Keras layer,
    including information such as layer type, configuration, and shape details. Alternating
    label positioning is applied to reduce visual overlap in plots or visualizations.

    :param label_overrides: A mapping of layer names to custom label strings. If provided,
        these custom labels will override the default labeling behavior for corresponding
        layer names. Defaults to None.
    :type label_overrides: Mapping[str, str] | None

    :return: A function that takes an index and a Keras layer, and returns a tuple containing
        a label string and a boolean indicating whether the label needs alternating positioning.
    :rtype: Callable[[int, keras.layers.Layer], tuple[str, bool]]
    """
    overrides = {} if label_overrides is None else dict(label_overrides)

    def label_layer(index: int, layer: keras.layers.Layer) -> tuple[str, bool]:
        if layer.name in overrides:
            label = overrides[layer.name]
        elif isinstance(layer, keras.layers.InputLayer):
            label = f"Input features\n{compact_shape(layer)}"
        elif isinstance(layer, keras.layers.Conv1D):
            label = (
                f"Conv1D\n{layer.filters} filters, k={layer.kernel_size[0]}\n"
                f"{compact_shape(layer)}"
            )
        elif isinstance(layer, keras.layers.AveragePooling1D):
            label = f"Average pool\npool={layer.pool_size[0]}\n{compact_shape(layer)}"
        elif isinstance(layer, keras.layers.GlobalAveragePooling1D):
            label = f"Global average pool\n{compact_shape(layer)}"
        elif isinstance(layer, keras.layers.GlobalMaxPooling1D):
            label = f"Global max pool\n{compact_shape(layer)}"
        elif isinstance(layer, keras.layers.Dense):
            activation = getattr(layer.activation, "__name__", "linear")
            label = (
                f"Dense ({layer.units})\nactivation={activation}\n"
                f"{compact_shape(layer)}"
            )
        elif isinstance(layer, keras.layers.BatchNormalization):
            label = f"Batch normalization\n{compact_shape(layer)}"
        elif isinstance(layer, keras.layers.LeakyReLU):
            label = f"LeakyReLU\n{compact_shape(layer)}"
        elif isinstance(layer, keras.layers.Dropout):
            label = f"Dropout ({layer.rate:g})\n{compact_shape(layer)}"
        else:
            label = f"{type(layer).__name__}\n{compact_shape(layer)}"

        # Alternating labels avoid collisions between neighboring layers.
        return label, index % 2 == 0

    return label_layer


def _display_model(
    model: keras.Model,
    layer_indices: Sequence[int] | None,
    include_input_layer: bool,
) -> SimpleNamespace:
    """Create a lightweight layer list used only by VisualKeras."""
    add_visualkeras_output_shapes(model)
    layers = (
        list(model.layers)
        if layer_indices is None
        else [model.layers[index] for index in layer_indices]
    )

    has_input = any(isinstance(layer, keras.layers.InputLayer) for layer in layers)
    if include_input_layer and not has_input:
        input_layer = keras.layers.InputLayer(
            shape=model.input_shape[1:],
            name="input_features",
        )
        input_layer.output_shape = model.input_shape
        layers.insert(0, input_layer)

    return SimpleNamespace(layers=layers)


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    """Wrap text using rendered pixel width instead of character count."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if current and width > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def compose_publication_figure(
    image: Image.Image,
    *,
    title: str,
    subtitle: str,
    legend_items: Sequence[tuple[str, str]],
    palette: Mapping[str, str] | None = None,
) -> Image.Image:
    """
    Compose a publication-style figure by combining an input image with additional
    elements such as a title, subtitle, and an optional legend. The function adjusts
    the layout dynamically based on the size of the input image, scales text and
    margins for proportionality, and provides customization through a color palette.

    :param image: The main image to be included in the output figure.
    :type image: PIL.Image.Image
    :param title: The title text to display at the top of the figure.
    :type title: str
    :param subtitle: The subtitle text to display below the title.
    :type subtitle: str
    :param legend_items: A sequence of (label, color) tuples representing legend
        entries. Each tuple contains a descriptive label and its associated color.
    :type legend_items: Sequence[tuple[str, str]]
    :param palette: An optional dictionary defining the color palette for the figure.
        Accepted keys include "text", "connector", and "outline". If not provided,
        defaults to `DEFAULT_PALETTE`.
    :type palette: Mapping[str, str] | None
    :return: A new image object that includes the combined input image, text headers,
        and legend if specified.
    :rtype: PIL.Image.Image
    """
    colors = DEFAULT_PALETTE if palette is None else palette
    scale = max(1.0, min(2.0, image.width / 1800))
    margin = round(38 * scale)
    title_font = publication_font(round(32 * scale), bold=True)
    body_font = publication_font(round(18 * scale))
    legend_font = publication_font(round(17 * scale))
    canvas_width = max(image.width + 2 * margin, 1200)

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    wrapped_subtitle = _wrap_text_to_width(
        measure,
        subtitle,
        body_font,
        canvas_width - 2 * margin,
    )
    title_height = measure.textbbox((0, 0), title, font=title_font)[3]
    subtitle_box = measure.multiline_textbbox(
        (0, 0),
        wrapped_subtitle,
        font=body_font,
        spacing=4,
    )
    subtitle_height = subtitle_box[3] - subtitle_box[1]
    header_height = (
        margin
        + title_height
        + round(12 * scale)
        + subtitle_height
        + round(24 * scale)
    )

    columns = min(4, len(legend_items))
    rows = ceil(len(legend_items) / columns) if columns else 0
    row_height = round(34 * scale)
    legend_height = margin + rows * row_height if legend_items else margin
    canvas_height = header_height + image.height + legend_height
    canvas = Image.new("RGBA", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), title, font=title_font, fill=colors["text"])
    draw.multiline_text(
        (margin, margin + title_height + round(12 * scale)),
        wrapped_subtitle,
        font=body_font,
        fill=colors["connector"],
        spacing=4,
    )
    canvas.paste(image, ((canvas_width - image.width) // 2, header_height), image)

    if legend_items:
        legend_y = header_height + image.height + round(18 * scale)
        column_width = (canvas_width - 2 * margin) / columns
        swatch = round(18 * scale)
        for index, (label, color) in enumerate(legend_items):
            row, column = divmod(index, columns)
            x = margin + column * column_width
            y = legend_y + row * row_height
            draw.rounded_rectangle(
                (x, y, x + swatch, y + swatch),
                radius=max(2, swatch // 5),
                fill=color,
                outline=colors["outline"],
                width=max(1, round(scale)),
            )
            draw.text(
                (x + swatch + round(9 * scale), y - round(2 * scale)),
                label,
                font=legend_font,
                fill=colors["text"],
            )
    return canvas


def render_layered_overview(
    model: keras.Model,
    *,
    title: str,
    subtitle: str,
    layer_indices: Sequence[int] | None = None,
    include_input_layer: bool = False,
    label_overrides: Mapping[str, str] | None = None,
    legend_categories: Sequence[str] | None = None,
    palette: Mapping[str, str] | None = None,
    spacing: int = 110,
    padding: int = 180,
    font_size: int = 20,
    draw_volume: bool = True,
) -> Image.Image:
    """
    Generate an image visualization that provides a layered overview of a Keras model
    using customizable layout, colors, and labels.

    This function creates a diagrammatic representation of the model's architecture,
    allowing users to visualize the structure in an aesthetically pleasing and clearly
    labeled format. The generated overview can include color-coded layers, titles,
    subtitles, legends, and various layout customizations for publication-ready visuals.

    :param model: The Keras model to visualize.
    :type model: keras.Model
    :param title: The title to display above the rendered visualization.
    :type title: str
    :param subtitle: A subtitle to display below the title in the rendered visualization.
    :type subtitle: str
    :param layer_indices: A sequence of layer indices specifying which layers to
        include in the visualization. If None, all layers are included.
    :type layer_indices: Sequence[int] | None
    :param include_input_layer: Set to True to include the input layer in the
        rendered visualization. Defaults to False.
    :type include_input_layer: bool
    :param label_overrides: A dictionary mapping layer names to custom labels. If None,
        default layer names are used.
    :type label_overrides: Mapping[str, str] | None
    :param legend_categories: A sequence of legend categories to display in the
        visualization's legend. If None, the legend will not include any categories.
    :type legend_categories: Sequence[str] | None
    :param palette: A dictionary mapping category names to their corresponding colors.
        If None, a default palette will be used.
    :type palette: Mapping[str, str] | None
    :param spacing: The spacing between elements in the rendered visualization.
        Defaults to 110.
    :type spacing: int
    :param padding: The padding around the entire rendered visualization. Defaults to 180.
    :type padding: int
    :param font_size: The font size to use for any text displayed in the rendered visualization.
        Defaults to 20.
    :type font_size: int
    :param draw_volume: Whether to render volumetric layers. Defaults to True.
    :type draw_volume: bool
    :return: A PIL Image object containing the rendered layered overview of the model.
    :rtype: Image.Image
    """
    colors = DEFAULT_PALETTE if palette is None else palette
    display_model = _display_model(model, layer_indices, include_input_layer)
    base = visualkeras.layered_view(
        display_model,
        color_map=build_layer_color_map(colors),
        legend=False,
        show_dimension=False,
        text_callable=make_layer_labeler(label_overrides),
        font=publication_font(font_size),
        font_color=colors["text"],
        text_vspacing=4,
        draw_volume=draw_volume,
        draw_funnel=True,
        min_xy=26,
        min_z=26,
        max_xy=260,
        max_z=220,
        scale_xy=2.0,
        scale_z=1.2,
        sizing_mode="balanced",
        dimension_caps={"channels": 200, "sequence": 240, "general": 240},
        spacing=spacing,
        padding=padding,
        shade_step=8,
        legend_text_spacing_offset=0,
    )
    return compose_publication_figure(
        base,
        title=title,
        subtitle=subtitle,
        legend_items=semantic_legend(legend_categories, palette=colors),
        palette=colors,
    )


def add_stage_axis(
    image: Image.Image,
    stages: Sequence[tuple[int, int, str]],
    *,
    node_size: int,
    layer_spacing: int,
    graph_padding: int,
    palette: Mapping[str, str] | None = None,
) -> Image.Image:
    """
    Adds a horizontal stage axis to an existing image, representing various stages and their
    corresponding ranges in a visual graph.

    This function modifies an input image by appending a new horizontal segment below it and
    rendering a stage axis with labels and markers corresponding to the provided stages. Each stage
    is defined with a starting column, ending column, and a label description. The visual appearance
    of the axis, including line spacing, node sizes, and colors, is configurable through parameters.

    :param image: The input image to which the stage axis will be appended.
    :type image: Image.Image

    :param stages: A sequence of tuples representing the stages. Each tuple contains the starting
        column, the ending column, and the stage label.
    :type stages: Sequence[tuple[int, int, str]]

    :param node_size: The size of each node in the graph.
    :type node_size: int

    :param layer_spacing: The spacing between layers in the graph.
    :type layer_spacing: int

    :param graph_padding: The padding applied to the graph on either side.
    :type graph_padding: int

    :param palette: A mapping defining color values used for visual elements. If None, a default
        color palette is used.
    :type palette: Mapping[str, str] | None

    :return: A modified image with the appended stage axis.
    :rtype: Image.Image
    """
    colors = DEFAULT_PALETTE if palette is None else palette
    band_height = 92
    canvas = Image.new("RGBA", (image.width, image.height + band_height), "white")
    canvas.paste(image, (0, 0), image)
    draw = ImageDraw.Draw(canvas)
    font = publication_font(20, bold=True)
    y = image.height + 25

    def column_center(column: int) -> float:
        return graph_padding + column * (node_size + layer_spacing) + node_size / 2

    for first_column, last_column, label in stages:
        x1, x2 = column_center(first_column), column_center(last_column)
        draw.line((x1, y, x2, y), fill=colors["outline"], width=3)
        draw.line((x1, y - 7, x1, y + 7), fill=colors["outline"], width=3)
        draw.line((x2, y - 7, x2, y + 7), fill=colors["outline"], width=3)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            ((x1 + x2 - text_width) / 2, y + 14),
            label,
            font=font,
            fill=colors["text"],
        )
    return canvas


def render_graph_view(
    model: keras.Model,
    *,
    title: str,
    subtitle: str,
    stages: Sequence[tuple[int, int, str]],
    legend_categories: Sequence[str] | None = None,
    palette: Mapping[str, str] | None = None,
    node_size: int = 46,
    layer_spacing: int = 30,
    node_spacing: int = 10,
    padding: int = 40,
) -> Image.Image:
    """Render an exact graph view with semantic colors and stage labels."""
    colors = DEFAULT_PALETTE if palette is None else palette
    base = visualkeras.graph_view(
        model,
        color_map=build_layer_color_map(colors),
        node_size=node_size,
        layer_spacing=layer_spacing,
        node_spacing=node_spacing,
        connector_fill=colors["connector"],
        connector_width=2,
        show_neurons=False,
        inout_as_tensor=True,
        padding=padding,
        background_fill="white",
    )
    staged = add_stage_axis(
        base,
        stages,
        node_size=node_size,
        layer_spacing=layer_spacing,
        graph_padding=padding,
        palette=colors,
    )
    return compose_publication_figure(
        staged,
        title=title,
        subtitle=subtitle,
        legend_items=semantic_legend(legend_categories, palette=colors),
        palette=colors,
    )


def notebook_preview(
    image: Image.Image,
    max_size: tuple[int, int] = (1600, 900),
) -> Image.Image:
    """
    Generate a resized preview of an image, preserving the aspect ratio, suitable for
    display in a notebook environment. The function creates a copy of the input image
    and resizes it to fit within the maximum size dimensions while maintaining its
    original proportions.

    :param image: An instance of the input image to generate a preview for.
    :type image: PIL.Image.Image
    :param max_size: A tuple representing the maximum allowed width and height of the
        generated preview. The default value is (1600, 900).
    :type max_size: tuple[int, int]
    :return: A resized copy of the input image that fits within the specified dimensions.
    :rtype: PIL.Image.Image
    """
    preview = image.copy()
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    return preview


def presentation_canvas(
    image: Image.Image,
    size: tuple[int, int] = (1920, 1080),
    margin: int = 60,
) -> Image.Image:
    """
    Create a presentation canvas by resizing an image and centering it on a white background.

    This function takes an input image, resizes it to fit within the defined canvas size while maintaining
    its aspect ratio, and places it centered on a plain white background. Margins are applied to define
    the spacing between the image and the edges of the canvas.

    :param image: The input image to be processed.
    :type image: Image.Image
    :param size: The dimensions of the canvas in pixels as a tuple (width, height). Defaults to (1920, 1080).
    :type size: tuple[int, int], optional
    :param margin: The margin (in pixels) applied to the canvas, defining an inset for the image. Defaults to 60.
    :type margin: int, optional
    :return: A new PIL Image object with the input image centered on a white background canvas.
    :rtype: Image.Image
    """
    available = (size[0] - 2 * margin, size[1] - 2 * margin)
    slide_image = image.copy()
    slide_image.thumbnail(available, Image.Resampling.LANCZOS)
    slide = Image.new("RGB", size, "white")
    position = (
        (size[0] - slide_image.width) // 2,
        (size[1] - slide_image.height) // 2,
    )
    slide.paste(slide_image.convert("RGB"), position)
    return slide


def export_publication_figures(
    figures: Mapping[str, Image.Image],
    output_root: str | Path,
    *,
    report_dpi: int = 300,
) -> dict[str, dict[str, Path]]:
    """
    Export publication-quality figures to specified directories in appropriate formats.

    This function processes a dictionary of figure images and exports them into
    separate directories for report and presentation use. The report images are saved
    in PNG format with the specified DPI, and presentation images are adjusted to
    a resolution of 1920x1080 pixels before saving in PNG format.

    :param figures:
        A mapping where the keys are filenames (without extensions) and the values
        are instances of PIL.Image.Image representing the figures to be exported.
    :param output_root:
        The root directory where the exported images will be saved. This directory
        will contain subdirectories for "report" and "presentation" images.
    :param report_dpi:
        The DPI (dots per inch) setting for report images. Defaults to 300.

    :return:
        A dictionary where the keys are filenames (from the input dictionary) and the
        values are dictionaries with keys "report" and "presentation". The values for
        these keys are Path objects pointing to the locations of the exported images.
    """
    output_root = Path(output_root)
    report_dir = output_root / "report"
    presentation_dir = output_root / "presentation"
    report_dir.mkdir(parents=True, exist_ok=True)
    presentation_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, dict[str, Path]] = {}
    for filename, figure in figures.items():
        report_path = report_dir / f"{filename}.png"
        presentation_path = presentation_dir / f"{filename}_1920x1080.png"
        figure.convert("RGB").save(
            report_path,
            dpi=(report_dpi, report_dpi),
            optimize=True,
        )
        presentation_canvas(figure).save(presentation_path, optimize=True)
        exported[filename] = {
            "report": report_path,
            "presentation": presentation_path,
        }
    return exported


def layer_configuration(layer: keras.layers.Layer) -> str:
    """Return the most useful configuration values for an architecture table."""
    if isinstance(layer, keras.layers.Conv1D):
        return f"filters={layer.filters}, kernel={layer.kernel_size[0]}"
    if isinstance(layer, keras.layers.AveragePooling1D):
        return f"pool={layer.pool_size[0]}"
    if isinstance(layer, keras.layers.Dense):
        activation = getattr(layer.activation, "__name__", str(layer.activation))
        return f"units={layer.units}, activation={activation}"
    if isinstance(layer, keras.layers.Dropout):
        return f"rate={layer.rate:g}"
    return ""




__all__ = [
    "DEFAULT_PALETTE",
    "LEGEND_LABELS",
    "add_stage_axis",
    "add_visualkeras_output_shapes",
    "build_layer_color_map",
    "compact_shape",
    "compose_publication_figure",
    "export_publication_figures",
    "find_project_root",
    "layer_configuration",
    "load_keras_models",
    "make_layer_labeler",
    "notebook_preview",
    "presentation_canvas",
    "publication_font",
    "render_graph_view",
    "render_layered_overview",
    "semantic_legend",
]
