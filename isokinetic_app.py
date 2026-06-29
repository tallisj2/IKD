import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


EXPECTED_COLS = [
    "Time(Seconds)",
    "Position(Degrees)",
    "Torque(Newton-Meters)",
    "Speed(d/s)",
]


ANGLE_CHECK_LOWER = 20.0
ANGLE_CHECK_UPPER = 60.0

# The velocity rule is now lower-bound only:
# speed must not be below 98% of target velocity.
VELOCITY_TOLERANCE_FRACTION = 0.02

# Set to 1.00 so velocity must be maintained across all available points
# between 20 and 60 degrees.
MIN_VALID_FRACTION_BETWEEN_20_60 = 1.00


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
    """
    Identifies rep boundaries using position direction changes.

    This prevents several torque fluctuations within one movement from being
    incorrectly counted as separate reps.
    """

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
    """
    Lower-bound velocity rule only.

    A sample is valid if absolute speed is greater than or equal to
    target velocity minus 2%.

    For example:
    target velocity = 60 d/s
    lower threshold = 58.8 d/s

    Speeds above 60 d/s are valid.
    Speeds below 58.8 d/s are invalid.
    """

    lower = abs(target_velocity) * (1 - tolerance_fraction)

    abs_speed = df["Speed(d/s)"].abs()
    valid_velocity = abs_speed >= lower

    return abs_speed, valid_velocity, lower


def get_velocity_based_rep_window(
    segment,
    start_idx,
    valid_velocity,
):
    """
    Returns the first and last sample in the segment where speed is above
    the lower velocity threshold.

    These points are used as the shaded start and shaded end points.
    """

    segment_valid = valid_velocity.iloc[start_idx:start_idx + len(segment)]
    true_positions = np.where(segment_valid.to_numpy())[0]

    if len(true_positions) == 0:
        return None, None

    shade_start_idx = start_idx + int(true_positions[0])
    shade_end_idx = start_idx + int(true_positions[-1])

    return shade_start_idx, shade_end_idx


def check_velocity_maintained_between_20_60(
    rep_df,
    target_velocity,
    tolerance_fraction=0.02,
    angle_lower=20.0,
    angle_upper=60.0,
    required_valid_fraction=1.00,
):
    """
    Checks whether velocity stays above the lower threshold between 20 and 60 degrees.

    A rep is marked as valid only if:
    - speed reaches the lower threshold at least once
    - there are data points between 20 and 60 degrees
    - all available points between 20 and 60 degrees are above the lower threshold

    Speeds higher than the target velocity are allowed.
    """

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
        return False, 0.0, "No data between 20 and 60 degrees"

    valid_between_angles = valid_velocity[angle_mask]
    valid_fraction = float(valid_between_angles.mean())

    if valid_fraction < required_valid_fraction:
        reason = (
            f"Speed dropped below lower threshold between 20 and 60 degrees "
            f"({valid_fraction * 100:.1f}% valid)"
        )
        return False, valid_fraction, reason

    return True, valid_fraction, "Valid"


def identify_reps_with_velocity_rule(
    df,
    target_velocity,
    trial_type,
    tolerance_fraction=0.02,
    consecutive_points=10,
):
    """
    Rep detection using:
    - position direction changes to identify rep transitions
    - lower-bound velocity threshold for shaded start/end points
    - red highlighting for reps that do not reach or do not maintain speed
      above the lower threshold between 20 and 60 degrees

    Important:
    The 2% rule is NOT an upper/lower band.
    It is only a lower threshold.

    Valid speed = absolute speed >= target velocity * 0.98
    """

    working = df.copy().reset_index(drop=True)

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

        rep_valid, valid_fraction_20_60, invalid_reason = (
            check_velocity_maintained_between_20_60(
                rep_df=rep_df,
                target_velocity=target_velocity,
                tolerance_fraction=tolerance_fraction,
                angle_lower=ANGLE_CHECK_LOWER,
                angle_upper=ANGLE_CHECK_UPPER,
                required_valid_fraction=MIN_VALID_FRACTION_BETWEEN_20_60,
            )
        )

        if not speed_reached_threshold:
            rep_valid = False
            valid_fraction_20_60 = 0.0
            invalid_reason = (
                f"Speed never reached lower threshold of {lower_threshold:.2f} d/s"
            )

        rep_number += 1

        rep_df["Rep"] = rep_number
        rep_df["Rep Type"] = get_trial_label(rep_number, trial_type)
        rep_df["Velocity Valid Rep"] = rep_valid
        rep_df["Velocity Valid Fraction 20-60 deg"] = valid_fraction_20_60
        rep_df["Velocity Validity Comment"] = invalid_reason
        rep_df["Segment Start Index"] = segment_start_idx
        rep_df["Segment End Index"] = segment_end_idx
        rep_df["Shade Start Index"] = export_start_idx
        rep_df["Shade End Index"] = export_end_idx
        rep_df["Lower Velocity Threshold (d/s)"] = lower_threshold

        reps.append(rep_df)

    if len(reps) == 0:
        return pd.DataFrame(), pd.DataFrame()

    reps_long = pd.concat(reps, ignore_index=True)

    summary_rows = []

    for rep_id, rep_df in reps_long.groupby("Rep", sort=True):
        peak_idx = rep_df["Torque(Newton-Meters)"].abs().idxmax()
        peak_row = rep_df.loc[peak_idx]

        velocity_valid_rep = bool(rep_df["Velocity Valid Rep"].iloc[0])
        validity_comment = str(rep_df["Velocity Validity Comment"].iloc[0])
        valid_fraction = float(rep_df["Velocity Valid Fraction 20-60 deg"].iloc[0])
        lower_threshold = float(rep_df["Lower Velocity Threshold (d/s)"].iloc[0])

        summary_rows.append(
            {
                "Rep": int(rep_id),
                "Rep Type": peak_row["Rep Type"],
                "Velocity Valid Rep": velocity_valid_rep,
                "Velocity Validity Comment": validity_comment,
                "Velocity Valid Fraction 20-60 deg": valid_fraction,
                "Lower Velocity Threshold (d/s)": lower_threshold,
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
                "Samples in Rep": int(len(rep_df)),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return reps_long, summary


def filter_reps_for_export(reps_long, summary, reps_to_export):
    """
    Keeps only the first selected number of automatically identified reps.

    This does not create new reps. It only controls how many of the identified
    reps are included in the final summary, figures, and Excel export.
    """

    reps_to_export = int(reps_to_export)

    filtered_reps = reps_long[reps_long["Rep"] <= reps_to_export].copy()
    filtered_summary = summary[summary["Rep"] <= reps_to_export].copy()

    return filtered_reps, filtered_summary


def make_raw_plot(df, summary=None, reps_to_export=None):
    """
    Raw torque-time plot.

    Shading:
    - green = rep speed reached and stayed above the lower threshold between 20 and 60 deg
    - red = rep speed did not reach or did not stay above the lower threshold
    - grey = rep excluded by manual override

    The shaded window uses only the part of the rep where speed is above
    the lower velocity threshold.
    """

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

            if rep_number > reps_to_export:
                shade_colour = "lightgray"
                alpha = 0.25
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

        title = (
            f"Raw Torque vs Time | Automatically identified reps: "
            f"{automatically_identified} | Reps selected for export: {int(reps_to_export)}"
        )

    else:
        title = "Raw Torque vs Time"

    ax.set_title(title)
    ax.set_xlabel("Time (Seconds)")
    ax.set_ylabel("Torque (Newton-Meters)")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig


def make_rep_plot(rep_df, rep_type, rep_id):
    """
    Individual rep plot.

    These subplots use angle/position on the x-axis.
    """

    fig, ax = plt.subplots(figsize=(5, 3.2))

    rep_valid = bool(rep_df["Velocity Valid Rep"].iloc[0])

    if rep_valid:
        line_colour = "tab:orange"
    else:
        line_colour = "tab:red"

    ax.plot(
        rep_df["Position(Degrees)"],
        rep_df["Torque(Newton-Meters)"],
        color=line_colour,
        linewidth=1.4,
    )

    peak_idx = rep_df["Torque(Newton-Meters)"].abs().idxmax()
    peak_row = rep_df.loc[peak_idx]

    ax.scatter(
        [peak_row["Position(Degrees)"]],
        [peak_row["Torque(Newton-Meters)"]],
        color="black",
        s=35,
        zorder=3,
    )

    ax.axvspan(
        ANGLE_CHECK_LOWER,
        ANGLE_CHECK_UPPER,
        color="lightgray",
        alpha=0.18,
    )

    if rep_valid:
        title_suffix = "Valid"
    else:
        title_suffix = "Velocity issue"

    ax.set_title(f"Rep {rep_id}: {rep_type} | {title_suffix}")
    ax.set_xlabel("Angle / Position (Degrees)")
    ax.set_ylabel("Torque (Nm)")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig


def build_export_excel(raw_df, summary_df, reps_long_df):
    """
    Export everything into one Excel workbook with multiple tabs:
    - Raw_Data
    - Summary
    - All_Reps_Long
    - one sheet per exported rep
    """

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw_Data", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        reps_export = reps_long_df.drop(columns=["Abs Speed(d/s)"], errors="ignore")
        reps_export.to_excel(writer, sheet_name="All_Reps_Long", index=False)

        for rep_id, rep_df in reps_export.groupby("Rep", sort=True):
            rep_type = rep_df["Rep Type"].iloc[0]
            safe_type = rep_type.replace(" ", "_")[:20]
            sheet_name = f"Rep_{int(rep_id)}_{safe_type}"[:31]

            rep_df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)

    return output.getvalue()


def main():
    st.set_page_config(page_title="Isokinetic Trial App", layout="wide")
    st.title("Isokinetic Trial Analysis")

    st.write(
        "Upload a CSV, TXT, XLSX, or XLS file. The app reads the first four columns as "
        "Time, Position, Torque, and Speed. The raw data plot is shown as Torque vs Time. "
        "Individual rep plots are shown as Torque vs Angle/Position. Reps are identified "
        "using position direction changes. The speed rule is lower-bound only: speed must "
        "not fall below 98% of the target velocity. Speeds higher than target are accepted."
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
        st.markdown("**Rep detection rule**")
        st.write("Lower threshold = target velocity x 0.98.")
        st.write("Speeds above target are accepted.")
        st.write("Red reps fall below the lower threshold between 20 and 60 degrees.")

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
        tolerance_fraction=VELOCITY_TOLERANCE_FRACTION,
        consecutive_points=10,
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
        f"{valid_reps} passed the lower-threshold velocity check and "
        f"{invalid_reps} were flagged red."
    )

    st.subheader("Manual rep override for final export")

    reps_to_export = st.number_input(
        "How many of the identified reps should be included in the final export?",
        min_value=1,
        max_value=automatically_identified_reps,
        value=automatically_identified_reps,
        step=1,
        help=(
            "Use this if the automatic detection has identified too many reps. "
            "The export will include the first selected number of identified reps."
        ),
    )

    reps_to_export = int(reps_to_export)

    if reps_to_export < automatically_identified_reps:
        st.warning(
            f"{automatically_identified_reps} reps were automatically identified, "
            f"but only the first {reps_to_export} reps will be included in the final export."
        )

    reps_long_export, summary_export = filter_reps_for_export(
        reps_long=reps_long,
        summary=summary,
        reps_to_export=reps_to_export,
    )

    st.subheader("Raw torque-time plot with identified reps")

    raw_fig_with_reps = make_raw_plot(
        df=df,
        summary=summary,
        reps_to_export=reps_to_export,
    )

    st.pyplot(raw_fig_with_reps)
    plt.close(raw_fig_with_reps)

    st.caption(
        "Green shaded regions are accepted reps. Red shaded regions are reps where speed "
        "was below the lower velocity threshold between 20 and 60 degrees, or where the "
        "threshold was never reached. Grey shaded regions are excluded by the manual override."
    )

    st.subheader("Rep summary table selected for export")
    st.dataframe(summary_export, use_container_width=True)

    st.subheader("All automatically identified reps")

    with st.expander("Show full automatic detection summary"):
        st.dataframe(summary, use_container_width=True)

    st.subheader("Rep figures selected for export")

    st.write(
        "These individual rep figures use angle/position on the x-axis. "
        "The grey background band marks 20-60 degrees."
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

            fig = make_rep_plot(rep_df, rep_type, int(rep_id))
            col_obj.pyplot(fig)
            plt.close(fig)

    excel_bytes = build_export_excel(
        raw_df=df,
        summary_df=summary_export,
        reps_long_df=reps_long_export,
    )

    st.download_button(
        label="Download selected reps as one Excel file (.xlsx)",
        data=excel_bytes,
        file_name="isokinetic_rep_analysis_selected_reps.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
