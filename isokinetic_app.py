import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.drawing.image import Image as XLImage


EXPECTED_COLS = [
    "Time(Seconds)",
    "Position(Degrees)",
    "Torque(Newton-Meters)",
    "Speed(d/s)",
]

DEFAULT_ANGLE_CHECK_LOWER = 20.0
DEFAULT_ANGLE_CHECK_UPPER = 60.0


def normalise_header_text(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def looks_like_header_row(row_values):
    row_norm = [normalise_header_text(v) for v in row_values[:4]]
    expected_norm = [normalise_header_text(v) for v in EXPECTED_COLS]

    matches = 0

    for observed, expected in zip(row_norm, expected_norm):
        if observed == expected or observed in expected or expected in observed:
            matches += 1

    return matches >= 3


def standardise_dataframe(df):
    df = df.copy()

    if df.shape[1] < 4:
        raise ValueError("The selected file has fewer than 4 columns.")

    df = df.iloc[:, :4].copy()

    if len(df) > 0 and looks_like_header_row(df.iloc[0].tolist()):
        df = df.iloc[1:].reset_index(drop=True)

    df.columns = EXPECTED_COLS

    for col in EXPECTED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "Time(Seconds)",
            "Position(Degrees)",
            "Torque(Newton-Meters)",
            "Speed(d/s)",
        ]
    ).reset_index(drop=True)

    return df


def get_excel_sheets(uploaded_file):
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()

    if name.endswith(".xlsx"):
        engine = "openpyxl"
    else:
        engine = "xlrd"

    xls = pd.ExcelFile(uploaded_file, engine=engine)
    return list(xls.sheet_names)


def read_uploaded_to_df(uploaded_file, selected_sheet=None):
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif name.endswith(".txt"):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

    elif name.endswith((".xlsx", ".xls")):
        if selected_sheet is None:
            raise ValueError("A sheet must be selected for Excel files.")

        uploaded_file.seek(0)

        if name.endswith(".xlsx"):
            engine = "openpyxl"
        else:
            engine = "xlrd"

        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet,
            engine=engine,
        )

    else:
        raise ValueError("Unsupported file type. Please upload CSV, TXT, XLSX, or XLS.")

    return standardise_dataframe(df)


def get_trial_label(rep_number, trial_type):
    odd = rep_number % 2 == 1

    if trial_type == "Con/Con":
        return "Concentric Extensors" if odd else "Concentric Flexors"

    if trial_type == "Ecc/Ecc":
        return "Eccentric Extensors" if odd else "Eccentric Flexors"

    return "Concentric" if odd else "Eccentric"


def get_action_group(rep_type):
    rep_type_text = str(rep_type)

    if "Extensor" in rep_type_text:
        return "Extensors"

    if "Flexor" in rep_type_text:
        return "Flexors"

    return rep_type_text


def get_smoothed_position_direction(working, smoothing_window=7):
    position = working["Position(Degrees)"].copy()

    smooth_position = position.rolling(
        window=smoothing_window,
        center=True,
        min_periods=1,
    ).median()

    position_change = smooth_position.diff()

    direction = np.sign(position_change)
    direction = pd.Series(direction, index=working.index)

    direction = direction.replace(0, np.nan)
    direction = direction.ffill().bfill()
    direction = direction.fillna(0)

    return smooth_position, direction


def find_position_direction_boundaries(
    working,
    smoothing_window=7,
    min_position_change_fraction=0.08,
    min_position_change_degrees=5.0,
    min_samples_between_boundaries=5,
):
    smooth_position, direction = get_smoothed_position_direction(
        working=working,
        smoothing_window=smoothing_window,
    )

    n = len(working)

    if n < 3:
        return [0, n - 1]

    position_range = float(
        working["Position(Degrees)"].max()
        - working["Position(Degrees)"].min()
    )

    min_position_change = max(
        min_position_change_degrees,
        position_range * min_position_change_fraction,
    )

    raw_change_points = []

    for i in range(1, n):
        previous_direction = direction.iloc[i - 1]
        current_direction = direction.iloc[i]

        if previous_direction == 0 or current_direction == 0:
            continue

        if current_direction != previous_direction:
            raw_change_points.append(i)

    raw_boundaries = [0] + raw_change_points + [n - 1]

    accepted_boundaries = [raw_boundaries[0]]
    last_accepted_idx = raw_boundaries[0]
    last_accepted_position = float(smooth_position.iloc[last_accepted_idx])

    for candidate_idx in raw_boundaries[1:-1]:
        if candidate_idx - last_accepted_idx < min_samples_between_boundaries:
            continue

        candidate_position = float(smooth_position.iloc[candidate_idx])
        position_change = abs(candidate_position - last_accepted_position)

        if position_change >= min_position_change:
            accepted_boundaries.append(candidate_idx)
            last_accepted_idx = candidate_idx
            last_accepted_position = candidate_position

    if accepted_boundaries[-1] != n - 1:
        accepted_boundaries.append(n - 1)

    accepted_boundaries = sorted(list(set(accepted_boundaries)))

    return accepted_boundaries


def get_velocity_valid_mask(df, target_velocity, tolerance_fraction=0.02):
    lower = abs(target_velocity) * (1 - tolerance_fraction)

    abs_speed = df["Speed(d/s)"].abs()
    valid_velocity = abs_speed >= lower

    return abs_speed, valid_velocity, lower


def get_velocity_based_rep_window(segment, start_idx, valid_velocity):
    segment_valid = valid_velocity.iloc[start_idx:start_idx + len(segment)]
    true_positions = np.where(segment_valid.to_numpy())[0]

    if len(true_positions) == 0:
        return None, None

    shade_start_idx = start_idx + int(true_positions[0])
    shade_end_idx = start_idx + int(true_positions[-1])

    return shade_start_idx, shade_end_idx


def check_velocity_maintained_between_angles(
    rep_df,
    target_velocity,
    tolerance_fraction=0.02,
    angle_lower=20.0,
    angle_upper=60.0,
    required_valid_fraction=1.00,
):
    if angle_upper < angle_lower:
        angle_lower, angle_upper = angle_upper, angle_lower

    abs_speed, valid_velocity, lower = get_velocity_valid_mask(
        rep_df,
        target_velocity=target_velocity,
        tolerance_fraction=tolerance_fraction,
    )

    if not bool(valid_velocity.any()):
        return False, 0.0, f"Speed never reached lower threshold of {lower:.2f} d/s"

    angle_mask = rep_df["Position(Degrees)"].between(
        angle_lower,
        angle_upper,
        inclusive="both",
    )

    if not bool(angle_mask.any()):
        return (
            False,
            0.0,
            f"No data between {angle_lower:.1f} and {angle_upper:.1f} degrees",
        )

    valid_between_angles = valid_velocity[angle_mask]
    valid_fraction = float(valid_between_angles.mean())

    if valid_fraction < required_valid_fraction:
        required_percent = required_valid_fraction * 100
        actual_percent = valid_fraction * 100

        reason = (
            f"Speed dropped below lower threshold between "
            f"{angle_lower:.1f} and {angle_upper:.1f} degrees "
            f"({actual_percent:.1f}% valid; required {required_percent:.1f}%)"
        )

        return False, valid_fraction, reason

    return True, valid_fraction, "Valid"


def identify_reps_with_velocity_rule(
    df,
    target_velocity,
    trial_type,
    tolerance_fraction=0.02,
    consecutive_points=10,
    angle_lower=20.0,
    angle_upper=60.0,
    required_valid_fraction=1.00,
):
    working = df.copy().reset_index(drop=True)

    if angle_upper < angle_lower:
        angle_lower, angle_upper = angle_upper, angle_lower

    abs_speed, valid_velocity, lower_threshold = get_velocity_valid_mask(
        working,
        target_velocity=target_velocity,
        tolerance_fraction=tolerance_fraction,
    )

    working["Abs Speed(d/s)"] = abs_speed
    working["Above Lower Velocity Threshold"] = valid_velocity
    working["Lower Velocity Threshold (d/s)"] = lower_threshold

    boundaries = find_position_direction_boundaries(
        working=working,
        smoothing_window=7,
        min_position_change_fraction=0.08,
        min_position_change_degrees=5.0,
        min_samples_between_boundaries=consecutive_points,
    )

    reps = []
    rep_number = 0

    position_range = float(
        working["Position(Degrees)"].max()
        - working["Position(Degrees)"].min()
    )

    min_rep_position_change = max(5.0, position_range * 0.08)
    min_rep_samples = max(5, int(consecutive_points / 2))

    for boundary_number in range(len(boundaries) - 1):
        segment_start_idx = int(boundaries[boundary_number])
        segment_end_idx = int(boundaries[boundary_number + 1])

        if segment_end_idx <= segment_start_idx:
            continue

        if segment_end_idx - segment_start_idx + 1 < min_rep_samples:
            continue

        segment = working.iloc[segment_start_idx:segment_end_idx + 1].copy()

        position_change = abs(
            float(segment["Position(Degrees)"].iloc[-1])
            - float(segment["Position(Degrees)"].iloc[0])
        )

        if position_change < min_rep_position_change:
            continue

        shade_start_idx, shade_end_idx = get_velocity_based_rep_window(
            segment=segment,
            start_idx=segment_start_idx,
            valid_velocity=valid_velocity,
        )

        if shade_start_idx is None or shade_end_idx is None:
            export_start_idx = segment_start_idx
            export_end_idx = segment_end_idx
            speed_reached_threshold = False
        else:
            export_start_idx = shade_start_idx
            export_end_idx = shade_end_idx
            speed_reached_threshold = True

        if export_end_idx <= export_start_idx:
            continue

        rep_df = working.iloc[export_start_idx:export_end_idx + 1].copy()

        if len(rep_df) < min_rep_samples:
            rep_df = working.iloc[segment_start_idx:segment_end_idx + 1].copy()
            export_start_idx = segment_start_idx
            export_end_idx = segment_end_idx

        rep_valid, valid_fraction_angle_range, invalid_reason = (
            check_velocity_maintained_between_angles(
                rep_df=rep_df,
                target_velocity=target_velocity,
                tolerance_fraction=tolerance_fraction,
                angle_lower=angle_lower,
                angle_upper=angle_upper,
                required_valid_fraction=required_valid_fraction,
            )
        )

        if not speed_reached_threshold:
            rep_valid = False
            valid_fraction_angle_range = 0.0
            invalid_reason = (
                f"Speed never reached lower threshold of {lower_threshold:.2f} d/s"
            )

        rep_number += 1

        rep_df["Rep"] = rep_number
        rep_df["Rep Type"] = get_trial_label(rep_number, trial_type)
        rep_df["Action Group"] = get_action_group(rep_df["Rep Type"].iloc[0])
        rep_df["Velocity Valid Rep"] = rep_valid
        rep_df["Velocity Valid Fraction Angle Range"] = valid_fraction_angle_range
        rep_df["Velocity Validity Comment"] = invalid_reason
        rep_df["Segment Start Index"] = segment_start_idx
        rep_df["Segment End Index"] = segment_end_idx
        rep_df["Shade Start Index"] = export_start_idx
        rep_df["Shade End Index"] = export_end_idx
        rep_df["Lower Velocity Threshold (d/s)"] = lower_threshold
        rep_df["Velocity Check Lower Angle (deg)"] = angle_lower
        rep_df["Velocity Check Upper Angle (deg)"] = angle_upper
        rep_df["Required Valid Fraction Angle Range"] = required_valid_fraction

        reps.append(rep_df)

    if len(reps) == 0:
        return pd.DataFrame(), pd.DataFrame()

    reps_long = pd.concat(reps, ignore_index=True)

    summary_rows = []

    for rep_id, rep_df in reps_long.groupby("Rep", sort=True):
        peak_idx = rep_df["Torque(Newton-Meters)"].abs().idxmax()
        peak_row = rep_df.loc[peak_idx]

        summary_rows.append(
            {
                "Rep": int(rep_id),
                "Rep Type": peak_row["Rep Type"],
                "Action Group": peak_row["Action Group"],
                "Velocity Valid Rep": bool(rep_df["Velocity Valid Rep"].iloc[0]),
                "Velocity Validity Comment": str(
                    rep_df["Velocity Validity Comment"].iloc[0]
                ),
                "Velocity Valid Fraction Angle Range": float(
                    rep_df["Velocity Valid Fraction Angle Range"].iloc[0]
                ),
                "Velocity Check Lower Angle (deg)": float(
                    rep_df["Velocity Check Lower Angle (deg)"].iloc[0]
                ),
                "Velocity Check Upper Angle (deg)": float(
                    rep_df["Velocity Check Upper Angle (deg)"].iloc[0]
                ),
                "Required Valid Fraction Angle Range": float(
                    rep_df["Required Valid Fraction Angle Range"].iloc[0]
                ),
                "Lower Velocity Threshold (d/s)": float(
                    rep_df["Lower Velocity Threshold (d/s)"].iloc[0]
                ),
                "Start Time (s)": float(rep_df["Time(Seconds)"].iloc[0]),
                "End Time (s)": float(rep_df["Time(Seconds)"].iloc[-1]),
                "Duration (s)": float(
                    rep_df["Time(Seconds)"].iloc[-1]
                    - rep_df["Time(Seconds)"].iloc[0]
                ),
                "Start Position (deg)": float(rep_df["Position(Degrees)"].iloc[0]),
                "End Position (deg)": float(rep_df["Position(Degrees)"].iloc[-1]),
                "Peak Time (s)": float(peak_row["Time(Seconds)"]),
                "Peak Torque (Nm)": float(peak_row["Torque(Newton-Meters)"]),
                "Position at Peak (deg)": float(peak_row["Position(Degrees)"]),
                "Speed at Peak (d/s)": float(peak_row["Speed(d/s)"]),
                "Mean Speed Magnitude (d/s)": float(rep_df["Abs Speed(d/s)"].mean()),
                "Minimum Speed Magnitude (d/s)": float(rep_df["Abs Speed(d/s)"].min()),
                "Maximum Speed Magnitude (d/s)": float(rep_df["Abs Speed(d/s)"].max()),
                "Samples in Rep": int(len(rep_df)),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return reps_long, summary


def filter_reps_for_export(reps_long, summary, reps_to_export, reps_to_exclude=None):
    reps_to_export = int(reps_to_export)

    if reps_to_exclude is None:
        reps_to_exclude = []

    reps_to_exclude = [int(rep) for rep in reps_to_exclude]

    filtered_reps = reps_long[
        (reps_long["Rep"] <= reps_to_export)
        & (~reps_long["Rep"].isin(reps_to_exclude))
    ].copy()

    filtered_summary = summary[
        (summary["Rep"] <= reps_to_export)
        & (~summary["Rep"].isin(reps_to_exclude))
    ].copy()

    return filtered_reps, filtered_summary


def calculate_torque_position_range_stats(reps_long_df, angle_lower, angle_upper):
    if reps_long_df.empty:
        return pd.DataFrame()

    if angle_upper < angle_lower:
        angle_lower, angle_upper = angle_upper, angle_lower

    df = reps_long_df.copy()

    in_range = df["Position(Degrees)"].between(
        angle_lower,
        angle_upper,
        inclusive="both",
    )

    df_range = df[in_range].copy()

    if df_range.empty:
        return pd.DataFrame(
            columns=[
                "Action Group",
                "Rep Type",
                "Position Range Lower (deg)",
                "Position Range Upper (deg)",
                "Number of Reps",
                "Number of Data Points",
                "Mean Torque (Nm)",
                "SD Torque (Nm)",
                "Minimum Torque (Nm)",
                "Maximum Torque (Nm)",
            ]
        )

    grouped = df_range.groupby(["Action Group", "Rep Type"], sort=True)

    stats = grouped.agg(
        Number_of_Reps=("Rep", "nunique"),
        Number_of_Data_Points=("Torque(Newton-Meters)", "count"),
        Mean_Torque_Nm=("Torque(Newton-Meters)", "mean"),
        SD_Torque_Nm=("Torque(Newton-Meters)", "std"),
        Minimum_Torque_Nm=("Torque(Newton-Meters)", "min"),
        Maximum_Torque_Nm=("Torque(Newton-Meters)", "max"),
    ).reset_index()

    stats = stats.rename(
        columns={
            "Number_of_Reps": "Number of Reps",
            "Number_of_Data_Points": "Number of Data Points",
            "Mean_Torque_Nm": "Mean Torque (Nm)",
            "SD_Torque_Nm": "SD Torque (Nm)",
            "Minimum_Torque_Nm": "Minimum Torque (Nm)",
            "Maximum_Torque_Nm": "Maximum Torque (Nm)",
        }
    )

    stats.insert(2, "Position Range Lower (deg)", float(angle_lower))
    stats.insert(3, "Position Range Upper (deg)", float(angle_upper))

    return stats


def create_mean_torque_position_curves(
    reps_long_df,
    angle_lower,
    angle_upper,
    n_points=101,
):
    """
    Creates mean torque-position curves for included reps only.

    Curves are grouped by Action Group:
    - Flexors
    - Extensors
    - other action labels if Flexors/Extensors are not present

    Each rep is interpolated onto a common position axis before calculating
    mean and SD torque at each position.
    """

    if reps_long_df.empty:
        return pd.DataFrame()

    if angle_upper < angle_lower:
        angle_lower, angle_upper = angle_upper, angle_lower

    angle_grid = np.linspace(float(angle_lower), float(angle_upper), int(n_points))

    curve_rows = []

    for action_group, group_df in reps_long_df.groupby("Action Group", sort=True):
        rep_curves = []

        for rep_id, rep_df in group_df.groupby("Rep", sort=True):
            rep_range = rep_df[
                rep_df["Position(Degrees)"].between(
                    angle_lower,
                    angle_upper,
                    inclusive="both",
                )
            ].copy()

            if len(rep_range) < 2:
                continue

            rep_range = rep_range.sort_values("Position(Degrees)")

            rep_range = (
                rep_range.groupby("Position(Degrees)", as_index=False)
                .agg({"Torque(Newton-Meters)": "mean"})
                .sort_values("Position(Degrees)")
            )

            x = rep_range["Position(Degrees)"].to_numpy(dtype=float)
            y = rep_range["Torque(Newton-Meters)"].to_numpy(dtype=float)

            if len(np.unique(x)) < 2:
                continue

            interp_y = np.interp(angle_grid, x, y)
            rep_curves.append(interp_y)

        if len(rep_curves) == 0:
            continue

        curve_array = np.vstack(rep_curves)

        mean_torque = np.nanmean(curve_array, axis=0)
        sd_torque = np.nanstd(curve_array, axis=0, ddof=1)

        if curve_array.shape[0] == 1:
            sd_torque = np.zeros_like(mean_torque)

        for idx, angle_value in enumerate(angle_grid):
            curve_rows.append(
                {
                    "Action Group": action_group,
                    "Position (deg)": float(angle_value),
                    "Mean Torque (Nm)": float(mean_torque[idx]),
                    "SD Torque (Nm)": float(sd_torque[idx]),
                    "Number of Reps": int(curve_array.shape[0]),
                }
            )

    return pd.DataFrame(curve_rows)


def make_mean_torque_position_plot(mean_curve_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    if mean_curve_df.empty:
        ax.text(
            0.5,
            0.5,
            "No mean torque-position data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    for action_group, group_df in mean_curve_df.groupby("Action Group", sort=True):
        x = group_df["Position (deg)"].to_numpy(dtype=float)
        y = group_df["Mean Torque (Nm)"].to_numpy(dtype=float)
        sd = group_df["SD Torque (Nm)"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            linewidth=2.0,
            label=f"{action_group} mean",
        )

        ax.fill_between(
            x,
            y - sd,
            y + sd,
            alpha=0.18,
            label=f"{action_group} SD",
        )

    ax.set_title("Mean Torque-Position Curves for Included Reps")
    ax.set_xlabel("Position (Degrees)")
    ax.set_ylabel("Torque (Nm)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def make_raw_plot(df, summary=None, reps_to_export=None, reps_to_exclude=None):
    if reps_to_exclude is None:
        reps_to_exclude = []

    reps_to_exclude = [int(rep) for rep in reps_to_exclude]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(
        df["Time(Seconds)"],
        df["Torque(Newton-Meters)"],
        color="tab:blue",
        linewidth=1.3,
    )

    if summary is not None and not summary.empty:
        automatically_identified = int(summary["Rep"].nunique())

        if reps_to_export is None:
            reps_to_export = automatically_identified

        for _, row in summary.iterrows():
            rep_number = int(row["Rep"])
            start_time = float(row["Start Time (s)"])
            end_time = float(row["End Time (s)"])

            if rep_number > reps_to_export or rep_number in reps_to_exclude:
                shade_colour = "lightgray"
                alpha = 0.30
            else:
                if bool(row["Velocity Valid Rep"]):
                    shade_colour = "tab:green"
                    alpha = 0.18
                else:
                    shade_colour = "tab:red"
                    alpha = 0.22

            ax.axvspan(
                start_time,
                end_time,
                color=shade_colour,
                alpha=alpha,
            )

            mid_time = (start_time + end_time) / 2
            y_top = df["Torque(Newton-Meters)"].max()

            ax.text(
                mid_time,
                y_top,
                f"Rep {rep_number}",
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
            )

        exported_count = len(
            [
                rep
                for rep in summary["Rep"].astype(int).tolist()
                if rep <= int(reps_to_export) and rep not in reps_to_exclude
            ]
        )

        title = (
            f"Raw Torque vs Time | Automatically identified reps: "
            f"{automatically_identified} | Final exported reps: {exported_count}"
        )

    else:
        title = "Raw Torque vs Time"

    ax.set_title(title)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Torque (Newton-Meters)")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig


def make_rep_plot(
    rep_df,
    rep_type,
    rep_id,
    target_velocity,
    tolerance_fraction,
    angle_lower,
    angle_upper,
):
    if angle_upper < angle_lower:
        angle_lower, angle_upper = angle_upper, angle_lower

    lower_velocity_threshold = abs(target_velocity) * (1 - tolerance_fraction)

    fig, ax1 = plt.subplots(figsize=(5.7, 3.6))

    rep_valid = bool(rep_df["Velocity Valid Rep"].iloc[0])
    valid_fraction = float(rep_df["Velocity Valid Fraction Angle Range"].iloc[0])

    if rep_valid:
        torque_colour = "tab:orange"
        title_suffix = "Valid"
    else:
        torque_colour = "tab:red"
        title_suffix = "Velocity issue"

    ax1.plot(
        rep_df["Position(Degrees)"],
        rep_df["Torque(Newton-Meters)"],
        color=torque_colour,
        linewidth=1.5,
        label="Torque",
    )

    peak_idx = rep_df["Torque(Newton-Meters)"].abs().idxmax()
    peak_row = rep_df.loc[peak_idx]

    ax1.scatter(
        [peak_row["Position(Degrees)"]],
        [peak_row["Torque(Newton-Meters)"]],
        color="black",
        s=35,
        zorder=3,
        label="Peak torque",
    )

    ax1.axvspan(
        angle_lower,
        angle_upper,
        color="lightgray",
        alpha=0.18,
    )

    ax1.set_xlabel("Angle / Position (Degrees)")
    ax1.set_ylabel("Torque (Nm)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(
        rep_df["Position(Degrees)"],
        rep_df["Abs Speed(d/s)"],
        color="tab:purple",
        linewidth=1.2,
        linestyle="--",
        label="Velocity",
    )

    ax2.axhline(
        lower_velocity_threshold,
        color="tab:green",
        linewidth=1.1,
        linestyle=":",
        label="Lower velocity threshold",
    )

    ax2.set_ylabel("Velocity magnitude (d/s)")

    ax1.set_title(
        f"Rep {rep_id}: {rep_type} | {title_suffix} | "
        f"{valid_fraction * 100:.0f}% valid"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
        fontsize=7,
    )

    fig.tight_layout()

    return fig


def build_export_excel(
    raw_df,
    summary_df,
    reps_long_df,
    angle_lower,
    angle_upper,
    mean_curve_df,
):
    output = io.BytesIO()

    torque_stats_df = calculate_torque_position_range_stats(
        reps_long_df=reps_long_df,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
    )

    mean_curve_fig = make_mean_torque_position_plot(mean_curve_df)

    temp_image_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            temp_image_path = tmp.name
            mean_curve_fig.savefig(temp_image_path, dpi=200, bbox_inches="tight")

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            raw_df.to_excel(writer, sheet_name="Raw_Data", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            torque_stats_df.to_excel(
                writer,
                sheet_name="Torque_Position_Range_Stats",
                index=False,
            )

            mean_curve_df.to_excel(
                writer,
                sheet_name="Mean_Torque_Position_Data",
                index=False,
            )

            figure_sheet = writer.book.create_sheet("Mean_Torque_Position_Figure")
            figure_sheet["A1"] = "Mean torque-position figure for included reps only"
            figure_sheet["A2"] = (
                "Curves are grouped by action group and calculated over the selected "
                "position range."
            )

            if temp_image_path is not None and os.path.exists(temp_image_path):
                xl_image = XLImage(temp_image_path)
                xl_image.anchor = "A4"
                figure_sheet.add_image(xl_image)

            reps_export = reps_long_df.drop(columns=["Abs Speed(d/s)"], errors="ignore")
            reps_export.to_excel(writer, sheet_name="All_Reps_Long", index=False)

            for rep_id, rep_df in reps_export.groupby("Rep", sort=True):
                rep_type = rep_df["Rep Type"].iloc[0]
                safe_type = rep_type.replace(" ", "_")[:20]
                sheet_name = f"Rep_{int(rep_id)}_{safe_type}"[:31]

                rep_df.to_excel(writer, sheet_name=sheet_name, index=False)

    finally:
        plt.close(mean_curve_fig)

        if temp_image_path is not None and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except OSError:
                pass

    output.seek(0)

    return output.getvalue(), torque_stats_df


def main():
    st.set_page_config(page_title="Isokinetic Trial App", layout="wide")
    st.title("Isokinetic Trial Analysis")

    st.markdown(
        """
        ### Isokinetic Data Processing App

        **Developed by Dr Jason Tallis**  
        **Contact:** AB0289@coventry.ac.uk

        This app supports the processing of isokinetic dynamometry trial data. It is
        designed to identify individual repetitions, inspect torque and velocity traces,
        apply velocity-threshold quality checks, manually exclude poor-quality or unwanted
        repetitions, and export cleaned repetition-level data for further analysis.

        **Purpose and function**

        - Upload CSV, TXT, XLSX, or XLS files containing isokinetic trial data.
        - The first four columns are read as **Time**, **Position**, **Torque**, and **Speed**.
        - Repetitions are identified using **position direction changes**.
        - The raw trace is displayed as **Torque vs Time**.
        - Individual repetition figures are displayed as **Torque vs Position**, with velocity shown on a secondary axis.
        - A lower-bound velocity threshold can be applied to identify reps where speed falls below the selected criterion.
        - The angle range for velocity checking and torque summary calculations can be manually changed.
        - Reps can be manually excluded from the final output.
        - The Excel export includes raw data, summary data, all included repetitions, individual rep sheets, mean/SD torque within the selected position range, and a mean torque-position figure for included reps.
        """
    )

    uploaded = st.file_uploader(
        "Upload data file",
        type=["csv", "txt", "xlsx", "xls"],
    )

    if uploaded is None:
        st.info("Upload a file to begin.")
        return

    selected_sheet = None
    file_name = uploaded.name.lower()

    try:
        if file_name.endswith((".xlsx", ".xls")):
            sheets = get_excel_sheets(uploaded)
            selected_sheet = st.selectbox("Select the sheet to analyse", sheets)

        df = read_uploaded_to_df(uploaded, selected_sheet)

    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    st.subheader("Loaded data preview")
    st.dataframe(df.head(15), use_container_width=True)

    controls = st.columns(3)

    with controls[0]:
        trial_type = st.radio(
            "Trial type",
            ["Con/Con", "Ecc/Ecc", "Con/Ecc"],
        )

    with controls[1]:
        target_velocity = st.number_input(
            "Target velocity (d/s)",
            min_value=0.0,
            value=60.0,
            step=1.0,
        )

    with controls[2]:
        tolerance_percent = st.number_input(
            "Allowed drop below target velocity (%)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
            help=(
                "Default is 2%. This means speed must be at least 98% of target. "
                "Speeds higher than target are accepted."
            ),
        )

    velocity_tolerance_fraction = tolerance_percent / 100.0

    st.subheader("Manual velocity validity rule")

    validity_cols = st.columns(3)

    with validity_cols[0]:
        angle_lower = st.number_input(
            "Velocity/torque calculation lower angle (degrees)",
            value=float(DEFAULT_ANGLE_CHECK_LOWER),
            step=1.0,
        )

    with validity_cols[1]:
        angle_upper = st.number_input(
            "Velocity/torque calculation upper angle (degrees)",
            value=float(DEFAULT_ANGLE_CHECK_UPPER),
            step=1.0,
        )

    with validity_cols[2]:
        required_valid_percent = st.number_input(
            "Required valid points in angle range (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=5.0,
            help=(
                "100% means every point in the selected angle range must be above "
                "the lower velocity threshold. Lower this if brief drops below the "
                "threshold are acceptable."
            ),
        )

    required_valid_fraction = required_valid_percent / 100.0

    if angle_upper < angle_lower:
        angle_lower, angle_upper = angle_upper, angle_lower

    lower_velocity_threshold = abs(target_velocity) * (1 - velocity_tolerance_fraction)

    st.info(
        f"Current red-flag rule: a rep is highlighted red if speed is below "
        f"{lower_velocity_threshold:.2f} d/s between {angle_lower:.1f} and "
        f"{angle_upper:.1f} degrees for more than "
        f"{100 - required_valid_percent:.1f}% of the available points. "
        f"Speeds above target velocity are accepted. The same angle range is used "
        f"to calculate the torque mean and SD by action type in the Excel export."
    )

    st.subheader("Raw torque-time plot")

    raw_fig = make_raw_plot(df)
    st.pyplot(raw_fig)
    plt.close(raw_fig)

    run_analysis = st.button("Run analysis", type="primary")

    if run_analysis:
        st.session_state["analysis_has_run"] = True

    if not st.session_state.get("analysis_has_run", False):
        return

    reps_long, summary = identify_reps_with_velocity_rule(
        df=df,
        target_velocity=target_velocity,
        trial_type=trial_type,
        tolerance_fraction=velocity_tolerance_fraction,
        consecutive_points=10,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
        required_valid_fraction=required_valid_fraction,
    )

    if reps_long.empty:
        st.warning(
            "No reps were identified with the current position-direction-change rule."
        )
        return

    automatically_identified_reps = int(summary["Rep"].nunique())
    valid_reps = int(summary["Velocity Valid Rep"].sum())
    invalid_reps = automatically_identified_reps - valid_reps

    st.success(
        f"Automatic detection identified {automatically_identified_reps} reps. "
        f"{valid_reps} passed the current velocity check and "
        f"{invalid_reps} were flagged red."
    )

    st.subheader("Manual rep selection for final export")

    reps_to_export = st.number_input(
        "How many of the identified reps should be considered for the final export?",
        min_value=1,
        max_value=automatically_identified_reps,
        value=automatically_identified_reps,
        step=1,
        help=(
            "This keeps the first selected number of identified reps. "
            "You can then manually exclude individual reps below."
        ),
    )

    reps_to_export = int(reps_to_export)

    candidate_reps = list(range(1, reps_to_export + 1))

    reps_to_exclude = st.multiselect(
        "Select reps to exclude from the final output",
        options=candidate_reps,
        default=[],
        help=(
            "These reps will be removed from the Summary, All_Reps_Long, individual "
            "rep sheets, rep figures, torque mean/SD calculations, and mean curves."
        ),
    )

    reps_to_exclude = [int(rep) for rep in reps_to_exclude]

    final_rep_count = reps_to_export - len(reps_to_exclude)

    if final_rep_count <= 0:
        st.error("All selected reps have been excluded. Keep at least one rep for export.")
        return

    if reps_to_export < automatically_identified_reps:
        st.warning(
            f"{automatically_identified_reps} reps were automatically identified, "
            f"but only the first {reps_to_export} reps are being considered."
        )

    if len(reps_to_exclude) > 0:
        st.warning(
            f"The following reps will be excluded from the final output: "
            f"{', '.join(str(rep) for rep in reps_to_exclude)}"
        )

    reps_long_export, summary_export = filter_reps_for_export(
        reps_long=reps_long,
        summary=summary,
        reps_to_export=reps_to_export,
        reps_to_exclude=reps_to_exclude,
    )

    torque_stats_df = calculate_torque_position_range_stats(
        reps_long_df=reps_long_export,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
    )

    mean_curve_df = create_mean_torque_position_curves(
        reps_long_df=reps_long_export,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
        n_points=101,
    )

    st.subheader("Raw torque-time plot with identified reps")

    raw_fig_with_reps = make_raw_plot(
        df=df,
        summary=summary,
        reps_to_export=reps_to_export,
        reps_to_exclude=reps_to_exclude,
    )

    st.pyplot(raw_fig_with_reps)
    plt.close(raw_fig_with_reps)

    st.caption(
        "Green shaded regions are accepted reps. Red shaded regions failed the current "
        "manual velocity rule. Grey shaded regions are excluded from the final output."
    )

    st.subheader("Rep summary table selected for export")
    st.dataframe(summary_export, use_container_width=True)

    st.subheader("Torque mean and SD within selected position range")

    st.write(
        f"Calculated using exported reps only, between {angle_lower:.1f} and "
        f"{angle_upper:.1f} degrees."
    )

    st.dataframe(torque_stats_df, use_container_width=True)

    st.subheader("Mean torque-position curves for included reps")

    mean_curve_fig = make_mean_torque_position_plot(mean_curve_df)
    st.pyplot(mean_curve_fig)
    plt.close(mean_curve_fig)

    with st.expander("Show mean torque-position curve data"):
        st.dataframe(mean_curve_df, use_container_width=True)

    st.subheader("All automatically identified reps")

    with st.expander("Show full automatic detection summary"):
        st.dataframe(summary, use_container_width=True)

    st.subheader("Rep figures selected for export")

    st.write(
        "Each individual rep figure shows torque against angle/position on the left axis "
        "and velocity magnitude on the right axis. The grey band shows the selected angle "
        "range used for the velocity check and torque mean/SD calculation."
    )

    rep_ids = sorted(reps_long_export["Rep"].unique())
    cols_per_row = 2

    for row_start in range(0, len(rep_ids), cols_per_row):
        row_cols = st.columns(cols_per_row)

        for col_obj, rep_id in zip(
            row_cols,
            rep_ids[row_start:row_start + cols_per_row],
        ):
            rep_df = reps_long_export[reps_long_export["Rep"] == rep_id]
            rep_type = rep_df["Rep Type"].iloc[0]

            fig = make_rep_plot(
                rep_df=rep_df,
                rep_type=rep_type,
                rep_id=int(rep_id),
                target_velocity=target_velocity,
                tolerance_fraction=velocity_tolerance_fraction,
                angle_lower=angle_lower,
                angle_upper=angle_upper,
            )

            col_obj.pyplot(fig)
            plt.close(fig)

    excel_bytes, torque_stats_export = build_export_excel(
        raw_df=df,
        summary_df=summary_export,
        reps_long_df=reps_long_export,
        angle_lower=angle_lower,
        angle_upper=angle_upper,
        mean_curve_df=mean_curve_df,
    )

    excel_mime = (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )

    st.download_button(
        label="Download selected reps as one Excel file (.xlsx)",
        data=excel_bytes,
        file_name="isokinetic_rep_analysis_selected_reps.xlsx",
        mime=excel_mime,
    )


if __name__ == "__main__":
    main()
