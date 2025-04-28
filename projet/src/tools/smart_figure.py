from __future__ import annotations
from shutil import which
from typing import Literal, Optional, Any
from warnings import warn
from string import ascii_lowercase
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from matplotlib.figure import SubFigure

from graphinglib.file_manager import (
    FileLoader,
    FileUpdater,
    get_default_style,
)
from graphinglib.graph_elements import GraphingException, Plottable
from graphinglib.legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)

import numpy as np
from numpy.typing import ArrayLike

class SmartFigure:
    def __init__(
        self,
        num_rows: int = 1,
        num_cols: int = 1,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size: tuple[float, float] = None,
        title: Optional[str] = None,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool = False,
        log_scale_y: bool = False,
        remove_axes: bool = False,
        aspect_ratio: float | str = "auto",
        remove_x_ticks: bool = False,
        remove_y_ticks: bool = False,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        projection: str | Any = None,
        figure_style: str = "default",
        elements: Optional[list[Plottable]] = [],
        show_legend: bool = False,                      # NEW
        legend_location: str = "best",                  # NEW
    ) -> None:
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._x_label = x_label
        self._y_label = y_label
        self._size = size
        self._title = title
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._log_scale_x = log_scale_x
        self._log_scale_y = log_scale_y
        self._remove_axes = remove_axes
        self._aspect_ratio = aspect_ratio
        self._remove_x_ticks = remove_x_ticks
        self._remove_y_ticks = remove_y_ticks
        self._reference_labels = reference_labels
        self._reflabel_loc = reflabel_loc
        self._width_padding = width_padding
        self._height_padding = height_padding
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._share_x = share_x
        self._share_y = share_y
        self._projection = projection
        self._figure_style = figure_style
        self._show_legend = show_legend                  # NEW
        self._legend_location = legend_location          # NEW

        self._elements = {}
        for i, element in enumerate(elements):
            self._elements[self._keys_to_slices(divmod(i, self._num_cols))] = element

        self._figure = None
        self._reference_label_i = None
        self._rc_dict = {}
        self._user_rc_dict = {}

    def __len__(self) -> int:
        return len(self._elements)
    
    def __setitem__(self, key: tuple[slice | int], element: Plottable | list[Plottable] | SmartFigure) -> None:
        key_ = self._keys_to_slices(key)
        if element is None:
            if key_ in self._elements.keys():
                self._elements.pop(key_)                # NEW
        else:
            self._elements[key_] = element

    def __getitem__(self, key: tuple[slice | int]) -> Plottable | list[Plottable] | SmartFigure:
        key_ = self._keys_to_slices(key)
        return self._elements.get(key_, None)

    @property
    def _ordered_elements(self) -> OrderedDict:
        return OrderedDict(sorted(self._elements.items(), key=lambda item: (item[0][0].start, item[0][1].start)))

    def show(
        self,
    ) -> None:
        self._initialize_parent_smart_figure()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    def save(
            self,
            file_name: str,
            dpi: Optional[int] = None,
            transparent: bool = False,
    ) -> None:
        self._initialize_parent_smart_figure()
        plt.savefig(
            file_name,
            bbox_inches="tight",
            dpi=dpi if dpi is not None else "figure",
            transparent=transparent,
        )
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    def _initialize_parent_smart_figure(
            self,
    ) -> None:
        if self._figure_style == "default":
            self._figure_style = get_default_style()
        try:
            file_loader = FileLoader(self._figure_style)
            self._default_params = file_loader.load()
            is_matplotlib_style = False
        except FileNotFoundError:
            is_matplotlib_style = True
            try:
                if self._figure_style == "matplotlib":
                    plt.style.use("default")
                else:
                    plt.style.use(self._figure_style)
                file_loader = FileLoader("plain")
                self._default_params = file_loader.load()
            except OSError:
                raise GraphingException(
                    f"The figure style {self._figure_style} was not found. Please choose a different style."
                )

        multi_figure_params_to_reset = self._fill_in_missing_params(self)

        self._fill_in_rc_params(is_matplotlib_style)

        self._figure = plt.figure(constrained_layout=True, figsize=self._size)
        self._figure.set_constrained_layout_pads(w_pad=0, h_pad=0)
        self._reference_label_i = 0
        main_gridspec = self._prepare_figure()
        if self._show_legend:       # NEW
            plt.legend(loc=self._legend_location, fontsize=plt.rcParams["font.size"], frameon=False)      # NEW

        self._reset_params_to_default(self, multi_figure_params_to_reset)

        # Create an artificial axis to add padding around the figure
        ax_dummy = self._figure.add_subplot(main_gridspec[:, :])
        ax_dummy.set_zorder(-1)
        ax_dummy.set_navigate(False)
        ax_dummy.tick_params(colors=(0,0,0,0), axis="both", direction="in", labelright=True, labeltop=True, labelsize=0)
        ax_dummy.spines["top"].set_visible(False)
        ax_dummy.spines["right"].set_visible(False)
        ax_dummy.spines["left"].set_visible(False)
        ax_dummy.spines["bottom"].set_visible(False)
        ax_dummy.set_xticks([0.5])
        ax_dummy.set_yticks([0.5])
        ax_dummy.set_xticklabels([" "])
        ax_dummy.set_yticklabels([" "])

    def _prepare_figure(
        self,
    ) -> GridSpec:
        gridspec = self._figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding,
            hspace=self._height_padding,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )
        sub_rcs = self._user_rc_dict
        plt.rcParams.update(sub_rcs)

        # Plottable and subfigure plotting
        ax = None   # keep track of the last Axis object, needed for sharing axes
        for (rows, cols), element in self._ordered_elements.items():
            if isinstance(element, SmartFigure):
                subfig = self._figure.add_subfigure(gridspec[rows, cols])
                element._figure = subfig        # associates the current subfigure with the nested SmartFigure
                element._reference_label_i = self._reference_label_i
                element._prepare_figure()
                self._reference_label_i = element._reference_label_i
                if self._show_legend:       # NEW
                    plt.legend(loc=self._legend_location, fontsize=plt.rcParams["font.size"], frameon=False)      # NEW

            else:
                current_elements = element if isinstance(element, list) else [element]
                subfig = self._figure.add_subfigure(gridspec[rows, cols])
                ax = subfig.add_subplot(
                    sharex=ax if self._share_x else None,
                    sharey=ax if self._share_y else None,
                    projection=self._projection,
                )

                # Plotting loop
                for current_element in current_elements:
                    current_element._plot_element(
                        ax,
                        2,
                    )

                # Add reference label
                if self._reference_labels and (len(self) > 1 or isinstance(self._figure, SubFigure)):
                    ax.text(
                        0,
                        1,
                        ascii_lowercase[self._reference_label_i] + ")",
                        transform=ax.transAxes + self._get_reflabel_translation(),
                    )
                    self._reference_label_i += 1
                
                # If axes are shared, manually remove ticklabels from unnecessary plots
                if self._share_x and rows.start != (self._num_rows - 1):
                    ax.tick_params(labelbottom=False)
                if self._share_y and cols.start != 0:
                    ax.tick_params(labelleft=False)

                # Remove ticks
                if self._remove_x_ticks:
                    ax.get_xaxis().set_visible(False)
                if self._remove_y_ticks:
                    ax.get_yaxis().set_visible(False)
                
                # Axes limits
                if self._x_lim:
                    ax.set_xlim(*self._x_lim)
                if self._y_lim:
                    ax.set_ylim(*self._y_lim)

                # Logarithmic scale
                if self._log_scale_x:
                    ax.set_xscale("log")
                if self._log_scale_y:
                    ax.set_yscale("log")

                # Remove axes
                if self._remove_axes:
                    ax.axis("off")

                ax.set_aspect(self._aspect_ratio)

        # Axes labels
        if self._num_cols == 1 and self._num_rows == 1:
            if ax is not None:  # makes sure an element was plotted and that an axis was created
                ax.set_xlabel(self._x_label)
                ax.set_ylabel(self._y_label)
        else:
            self._figure.supxlabel(self._x_label, fontsize=plt.rcParams["font.size"])
            self._figure.supylabel(self._y_label, fontsize=plt.rcParams["font.size"])

        # Title
        if self._title:
            self._figure.suptitle(self._title, fontdict={"fontsize": "medium"})

        return gridspec

    def _fill_in_missing_params(self, element: Plottable) -> list[str]:
        """
        Fills in the missing parameters from the specified ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        for property, value in vars(element).items():
            if (type(value) == str) and (value == "default"):
                params_to_reset.append(property)
                if self._default_params[object_type][property] == "same as curve":
                    element.__dict__["_errorbars_color"] = self._default_params[
                        object_type
                    ]["_color"]
                    element.__dict__["_errorbars_line_width"] = self._default_params[
                        object_type
                    ]["_line_width"]
                    element.__dict__["_cap_thickness"] = self._default_params[
                        object_type
                    ]["_line_width"]
                elif self._default_params[object_type][property] == "same as scatter":
                    element.__dict__["_errorbars_color"] = self._default_params[
                        object_type
                    ]["_face_color"]
                else:
                    element.__dict__[property] = self._default_params[object_type][
                        property
                    ]
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the _fill_in_missing_params method.
        """
        for param in params_to_reset:
            setattr(element, param, "default")

    def _fill_in_rc_params(self, is_matplotlib_style: bool = False) -> None:
        """
        Fills in and sets the missing rc parameters from the specified ``figure_style``.
        If ``is_matplotlib_style`` is ``True``, the rc parameters are reset to the default values for the specified ``figure_style``.
        If ``is_matplotlib_style`` is ``False``, the rc parameters are updated with the missing parameters from the specified ``figure_style``.
        In both cases, the rc parameters are then updated with the user-specified parameters.
        """
        if is_matplotlib_style:
            if self._figure_style == "matplotlib":
                plt.style.use("default")
            else:
                plt.style.use(self._figure_style)
            plt.rcParams.update(self._user_rc_dict)
        else:
            params = self._default_params["rc_params"]
            for property, value in params.items():
                # add to rc_dict if not already in there
                if (property not in self._rc_dict) and (
                    property not in self._user_rc_dict
                ):
                    self._rc_dict[property] = value
            all_rc_params = {**self._rc_dict, **self._user_rc_dict}
            try:
                if all_rc_params["text.usetex"] and which("latex") is None:
                    all_rc_params["text.usetex"] = False
            except KeyError:
                pass
            plt.rcParams.update(all_rc_params)

    @staticmethod
    def _keys_to_slices(keys: tuple[slice | int]) -> tuple[slice]:
        new_slices = [k if isinstance(k, slice) else slice(k, k+1, None) for k in keys]   # convert int -> slice
        new_slices = [s if s.start is not None else slice(0, s.stop) for s in new_slices] # convert starting None -> 0
        return tuple(new_slices)

    def _get_reflabel_translation(
            self,
    ) -> ScaledTranslation:
        if self._reflabel_loc == "outside":
            return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
        elif self._reflabel_loc == "inside":
            return ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Invalid reference label location. Please specify either 'inside' or 'outside'.")
