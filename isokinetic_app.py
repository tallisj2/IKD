import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


EXPECTED_COLS = [
    "Time(Seconds)",
    "Position(Degrees)",
    "Torque(Newton-Meters)",
    "Speed(d/s)",
]


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

    # Keep only first 4 columns
    df = df.iloc[:, :4].copy()

    # If first row looks like a repeated header row, remove it
    if len(df) > 0 and looks_like_header_row(df.iloc[0].tolist()):
        df = df.iloc[1:].reset_index(drop=True)

    # Standardise column names
    df.columns = EXPECTED_COLS

    # Convert to numeric
    for col in EXPECTED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential values
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
            engine=engine
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

    # Con/Ecc
    return "Concentric" if odd else "Eccentric"


def identify_reps_with_velocity_rule(
    df,
    target_velocity,
    trial_type,
    tolerance_fraction=0.01,
    consecutive_points=10,
):
    """
    Rep start:
        first point of 10 consecutive samples within ±1% of target velocity

    Rep end:
        last point before 10 consecutive samples outside ±1% of target velocity
    """

    working = df.copy().reset_index(drop=True)

    lower = abs(target_velocity) * (1 - tolerance_fraction)
    upper = abs(target_velocity) * (1 + tolerance_fraction)

    # Use absolute speed magnitude for the target velocity match
    working["Abs Speed(d/s)"] = working["Speed(d/s)"].abs()
    valid = working["Abs Speed(d/s)"].between(lower, upper)

    reps = []
    in_rep = False
    start_idx = None
    rep_number = 0
    i = 0
    n = len(working)

    while i <= n - consecutive_points:
        if not in_rep:
            # Rep starts at the first point of 10 consecutive valid samples
            if bool(valid.iloc[i:i + consecutive_points].all()):
                in_rep = True
                start_idx = i
                rep_number += 1
                i += consecutive_points
                continue

        else:
            # Rep ends at the last point before 10 consecutive invalid samples
            if bool((~valid.iloc[i:i + consecutive_points]).all()):
                end_idx = i - 1

                if end_idx >= start_idx:
                    rep_df = working.iloc[start_idx:end_idx + 1].copy()
                    rep_df["Rep"] = rep_number
                    rep_df["Rep Type"] = get_trial_label(rep_number, trial_type)
                    reps.append(rep_df)

                in_rep = False
                start_idx = None
                i += consecutive_points
                continue

        i += 1

    # If file ends during a rep, keep the final rep
    if in_rep and start_idx is not None:
        rep_df = working.iloc[start_idx:].copy()
        rep_df["Rep"] = rep_number
        rep_df["Rep Type"] = get_trial_label(rep_number, trial_type)
        reps.append(rep_df)

    if len(reps) == 0:
        return pd.DataFrame(), pd.DataFrame()

    reps_long = pd.concat(reps, ignore_index=True)

    # Build summary table
    summary_rows = []

    for rep_id, rep_df in reps_long.groupby("Rep", sort=True):
        peak_idx = rep_df["Torque(Newton-Meters)"].abs().idxmax()
        peak_row = rep_df.loc[peak_idx]

        summary_rows.append(
            {
                "Rep": int(rep_id),
                "Rep Type": peak_row["Rep Type"],
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

    This plot keeps time on the x-axis.
    If reps are supplied, identified reps are shaded:
    - green = included in export
    - red = identified but excluded by manual override
    """

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(
        df["Time(Seconds)"],
        df["Torque(Newton-Meters)"],
        color="tab:blue",
        linewidth=1.3
    )

    if summary is not None and not summary.empty:
        automatically_identified = int(summary["Rep"].nunique())

        if reps_to_export is None:
            reps_to_export = automatically_identified

        for _, row in summary.iterrows():
            rep_number = int(row["Rep"])
            start_time = float(row["Start Time (s)"])
            end_time = float(row["End Time (s)"])

            if rep_number <= reps_to_export:
                shade_colour = "tab:green"
                alpha = 0.18
            else:
                shade_colour = "tab:red"
                alpha = 0.12

            ax.axvspan(
                start_time,
                end_time,
                color=shade_colour,
                alpha=alpha
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
                rotation=90
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
    Individual rep torque-angle plot.

    These subplots use angle/position on the x-axis.
    """

    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.plot(
        rep_df["Position(Degrees)"],
        rep_df["Torque(Newton-Meters)"],
        color="tab:orange",
        linewidth=1.4
    )

    peak_idx = rep_df["Torque(Newton-Meters)"].abs().idxmax()
    peak_row = rep_df.loc[peak_idx]

    ax.scatter(
        [peak_row["Position(Degrees)"]],
        [peak_row["Torque(Newton-Meters)"]],
        color="red",
        s=35,
        zorder=3
    )

    ax.set_title(f"Rep {rep_id}: {rep_type}")
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
        "Individual rep plots are shown as Torque vs Angle/Position. Reps are segmented "
        "using the ±1% target-velocity rule with a 10-consecutive-sample start/end rule."
    )

    uploaded = st.file_uploader(
        "Upload data file",
        type=["csv", "txt", "xlsx", "xls"]
    )

    if uploaded is None:
        st.info("Upload a file to begin.")
        return

    selected_sheet = None
    file_name = uploaded.name.lower()

    try:
        # Sheet selector shown before analysis for Excel files
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
            ["Con/Con", "Ecc/Ecc", "Con/Ecc"]
        )

    with controls[1]:
        target_velocity = st.number_input(
            "Target velocity (d/s)",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    with controls[2]:
        st.markdown("**Velocity rule**")
        st.write("Rep starts after 10 consecutive samples within ±1% of target velocity.")
        st.write("Rep ends when 10 consecutive samples fall outside that window.")

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
        tolerance_fraction=0.01,
        consecutive_points=10,
    )

    if reps_long.empty:
        st.warning(
            "No reps were identified with the current target velocity and ±1% rule."
        )
        return

    automatically_identified_reps = int(summary["Rep"].nunique())

    st.success(
        f"Automatic detection identified {automatically_identified_reps} reps."
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
        )
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
        reps_to_export=reps_to_export
    )

    st.subheader("Raw torque-time plot with identified reps")

    raw_fig_with_reps = make_raw_plot(
        df=df,
        summary=summary,
        reps_to_export=reps_to_export
    )

    st.pyplot(raw_fig_with_reps)
    plt.close(raw_fig_with_reps)

    st.caption(
        "Green shaded regions are reps selected for export. "
        "Red shaded regions are reps identified automatically but excluded by the manual override."
    )

    st.subheader("Rep summary table selected for export")
    st.dataframe(summary_export, use_container_width=True)

    st.subheader("All automatically identified reps")

    with st.expander("Show full automatic detection summary"):
        st.dataframe(summary, use_container_width=True)

    st.subheader("Rep figures selected for export")

    st.write(
        "These individual rep figures use angle/position on the x-axis."
    )

    rep_ids = sorted(reps_long_export["Rep"].unique())
    cols_per_row = 2

    for row_start in range(0, len(rep_ids), cols_per_row):
        row_cols = st.columns(cols_per_row)

        for col_obj, rep_id in zip(
            row_cols,
            rep_ids[row_start:row_start + cols_per_row]
        ):
            rep_df = reps_long_export[reps_long_export["Rep"] == rep_id]
            rep_type = rep_df["Rep Type"].iloc[0]

            fig = make_rep_plot(rep_df, rep_type, int(rep_id))
            col_obj.pyplot(fig)
            plt.close(fig)

    excel_bytes = build_export_excel(
        raw_df=df,
        summary_df=summary_export,
        reps_long_df=reps_long_export
    )

    st.download_button(
        label="Download selected reps as one Excel file (.xlsx)",
        data=excel_bytes,
        file_name="isokinetic_rep_analysis_selected_reps.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
