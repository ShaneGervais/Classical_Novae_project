import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------

RUNS_DIR = Path("../runs")
OUT_DIR = Path("../analysis")
FIG_DIR = Path("../figures")

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUT_DIR / "nova_mass_loss_summary.csv"

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def read_history(history_file):
    with open(history_file, "r") as f:
        lines = f.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("model_number"):
            header_line = i
            break

    if header_line is None:
        raise RuntimeError("Could not find MESA header line in history.data")

    df = pd.read_csv(
        history_file,
        sep=r"\s+",
        skiprows=header_line,
        engine="python"
    )

    return df

def find_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c
    return None

summary_rows = []

for run_dir in sorted(RUNS_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    
    run_name = run_dir.name
    history_file = run_dir / "LOGS" / "history.data"

    if not history_file.exists():
        print(f"Warning: {history_file} does not exist. Skipping.")
        continue

    df = read_history(history_file)
    # USE THIS TO SEE THE HEADER COLUMNS FOR HISTORY.DATA
    #print(f"\n[DEBUG] Columns in {run_name}:\n{df.columns.tolist()}\n")


    # Identify relevant columns
    age_col = find_column(df, ["star_age", "age"])
    time_col = find_column(df, ["time_step_sec"])
    logdt_col = find_column(df, ["log_dt"])
    mass_col = find_column(df, ["star_mass", "mass"])
    logxmstar_col = find_column(df, ["log_xmstar"])
    log_abs_mdot_col = find_column(df, ["log_abs_mdot"])
    mass_conv_core_col = find_column(df, ["mass_conv_core"])
    conv_mx1_top_col = find_column(df, ["conv_mx1_top"])
    conv_mx1_bot_col = find_column(df, ["conv_mx1_bot"])
    conv_mx2_top_col = find_column(df, ["conv_mx2_top"])
    conv_mx2_bot_col = find_column(df, ["conv_mx2_bot"])
    mx1_bot_col = find_column(df, ["mx1_bot"])
    mx2_bot_col = find_column(df, ["mx2_bot"])
    mx1_top_col = find_column(df, ["mx1_top"])
    mx2_top_col = find_column(df, ["mx2_top"])
    log_LH_col = find_column(df, ["log_LH"])
    log_LHe_col = find_column(df, ["log_LHe"])
    log_LZ_col = find_column(df, ["log_LZ"])
    log_Lnuc_col = find_column(df, ["log_Lnuc"])
    pp_col = find_column(df, ["pp"])
    cno_col = find_column(df, ["cno"])
    tri_alpha_col = find_column(df, ["tri_alpha"])
    epsnuc_M_1_col = find_column(df, ["epsnuc_M_1"])
    epsnuc_M_2_col = find_column(df, ["epsnuc_M_2"])
    epsnuc_M_3_col = find_column(df, ["epsnuc_M_3"])
    epsnuc_M_4_col = find_column(df, ["epsnuc_M_4"])
    epsnuc_M_5_col = find_column(df, ["epsnuc_M_5"])
    epsnuc_M_6_col = find_column(df, ["epsnuc_M_6"])
    epsnuc_M_7_col = find_column(df, ["epsnuc_M_7"])
    epsnuc_M_8_col = find_column(df, ["epsnuc_M_8"])
    he_core_mass_col = find_column(df, ["he_core_mass"])
    co_core_mass_col = find_column(df, ["co_core_mass"])
    one_core_mass_col = find_column(df, ["one_core_mass"])
    fe_core_mass_col = find_column(df, ["fe_core_mass"])
    neutron_rich_core_mass_col = find_column(df, ["neutron_rich_core_mass"])
    kh_timescales_col = find_column(df, ["kh_timescales"])
    effective_T_col = find_column(df, ["effective_T"])
    log_Teff_col = find_column(df, ["log_Teff"])
    log_L_col = find_column(df, ["log_L"])
    log_R_col = find_column(df, ["log_R"])
    radius_cm_col = find_column(df, ["radius_cm"])
    log_R_cm_col = find_column(df, ["log_R_cm"])
    log_g_col = find_column(df, ["log_g"])
    log_center_Rho_col = find_column(df, ["log_center_Rho"])
    log_center_P_col = find_column(df, ["log_center_P"])
    total_mass_h1_col = find_column(df, ["total_mass_h1"])
    total_mass_he4_col = find_column(df, ["total_mass_he4"])
    max_tau_conv_col = find_column(df, ["max_tau_conv"])
    num_retries_col = find_column(df, ["num_retries"])
    num_iters_col = find_column(df, ["num_iters"])
    model_number_col = find_column(df, ["model_number"])
    num_zones_col = find_column(df, ["num_zones"])
    center_c12_col = find_column(df, ["center_c12"])
    center_o16_col = find_column(df, ["center_o16"])
    surface_c12_col = find_column(df, ["surface_c12"])
    surface_o16_col = find_column(df, ["surface_o16"])

    if age_col is None or mass_col is None or log_abs_mdot_col is None:
        print(f"Warning: Required columns not found in {history_file}. Skipping.")
        continue
    """

    if age_col is None or mass_col is None:
        print(f"Warning: Required columns not found in {history_file}. Skipping.")
        continue
    """

    age = df[age_col].values # years
    mass = df[mass_col].values # solar masses
    log_abs_mdot = df[log_abs_mdot_col].values # log10(solar masses / year)
    log_Lnuc = df[log_Lnuc_col].values if log_Lnuc_col is not None else None
    time = df[time_col].values if time_col is not None else None

    # --- handle mdot consistently ---
    if log_abs_mdot_col is not None:
        mdot = 10.0 ** df[log_abs_mdot_col].values
    else:
        mdot = None

    # --- compute mass loss ---
    M_initial = mass[0]
    M_final = mass[-1]
    delta_M = M_initial - M_final
    print(f"Mass loss (M_final - M_initial): {delta_M:.3e} for {run_name}")

    # integrate mdot if available
    if mdot is not None:
        delta_t = np.diff(age)
        mdot_mid = 0.5 * (mdot[1:] + mdot[:-1])
        integrated_mdot = np.sum(mdot_mid * delta_t)
    else:
        integrated_mdot = np.nan

    # --- store summary ---
    summary_rows.append({
        "run": run_name,
        "M_initial_Msun": M_initial,
        "M_final_Msun": M_final,
        "Delta_M_Msun": delta_M,
        "Integrated_mdot_Msun": integrated_mdot,
        "t_start_yr": age[0],
        "t_end_yr": age[-1]
    })

    # --- plots ---
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(age, mass, color="black")
    ax1.set_xlabel("Age (yr)")
    ax1.set_ylabel("Star mass (Msun)")
    ax1.set_title(run_name)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{run_name}_mass_vs_age.png", dpi=200)
    plt.close(fig)

    if mdot is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(age, mdot)
        ax.set_yscale("symlog")
        ax.set_xlabel("Age (yr)")
        ax.set_ylabel("Mass loss rate (Msun / yr)")
        ax.set_title(run_name)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{run_name}_mdot_vs_age.png", dpi=200)
        plt.close(fig)

    plt.figure()
    plt.plot(age, log_Lnuc)
    plt.xlabel("Age (yr)")
    plt.ylabel("Log10(Nuclear Luminosity)")
    plt.title(run_name)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_log_Lnuc_vs_age.png", dpi=200)
    plt.close()

    # finding when (age) log_Lnuc changes
    log_Lnuc_diff = np.gradient(log_Lnuc)
    change_points = np.where(np.abs(log_Lnuc_diff) > 1e-5)[0]
    #for cp in change_points:
    #    print(f"Log10(Nuclear Luminosity) changes at age {age[cp]}")
    
    # plot derivative plots

    plt.figure()
    plt.plot(age, log_Lnuc_diff)
    plt.xlabel("Age (yr)")
    plt.ylabel("d(Log10(Nuclear Luminosity))/dt")
    plt.title(run_name)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_log_Lnuc_diff_vs_age.png", dpi=200)
    plt.close()

    # plot change in mass vs mdot
    plt.figure()
    plt.plot(mdot, mass)
    plt.xlabel("Mass loss rate (Msun / yr)")
    plt.ylabel("Star mass (Msun)")
    plt.title(run_name)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_mass_vs_mdot.png", dpi=200)
    plt.close()

    # plot pp, cno, tri_alpha if available in one plot for each
    plt.figure()
    if pp_col is not None:
        plt.plot(age, df[pp_col].values, label="pp Fusion Rate")
    if cno_col is not None:
        plt.plot(age, df[cno_col].values, label="CNO Fusion Rate")
    if tri_alpha_col is not None:
        plt.plot(age, df[tri_alpha_col].values, label="Triple-alpha Fusion Rate")
    plt.xlabel("Age (yr)")
    plt.ylabel("Fusion Rate")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_pp+cno+tri_alpha_vs_age.png", dpi=200)
    plt.close()

    # plotting their derivatives
    plt.figure()
    if pp_col is not None:
        plt.plot(age, np.gradient(df[pp_col].values), label="pp Fusion Rate")
    if cno_col is not None:
        plt.plot(age, np.gradient(df[cno_col].values), label="CNO Fusion Rate")
    if tri_alpha_col is not None:
        plt.plot(age, np.gradient(df[tri_alpha_col].values), label="Triple-alpha Fusion Rate")
    plt.xlabel("Age (yr)")
    plt.ylabel("Fusion Rate Derivative")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_pp+cno+tri_alpha_vs_age_derivative.png", dpi=200)
    plt.close()

    # create another plot of pp+cno+triple_alpha with lognuc overlayed (should probably put this in arbitrary units of 1)
    plt.figure()
    if pp_col is not None:
        plt.plot(age, df[pp_col].values, label="pp Fusion Rate")
    if cno_col is not None:
        plt.plot(age, df[cno_col].values, label="CNO Fusion Rate")
    if tri_alpha_col is not None:
        plt.plot(age, df[tri_alpha_col].values, label="Triple-alpha Fusion Rate")
    plt.plot(age, log_Lnuc, label="Log10(Nuclear Luminosity)", linestyle="--")
    plt.xlabel("Age (yr)")
    plt.ylabel("Fusion Rate / Log10(Nuclear Luminosity)")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_pp+cno+tri_alpha+log_Lnuc_vs_age.png", dpi=200)
    plt.close()

    t_eruption = age[np.argmax(log_Lnuc)]

    # plot center_c12
    plt.figure()
    plt.plot(age, df[center_c12_col].values, label="Center C12 Abundance")
    plt.xlabel("Age (yr)")
    plt.ylabel("Center C12 Abundance")
    plt.title(run_name)
    #plt.axvline(x=t_eruption, color='r', linestyle='--', label="Eruption Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_center_c12_vs_age.png", dpi=200)
    plt.close()

    # plot center_o16
    plt.figure()
    plt.plot(age, df[center_o16_col].values, label="Center O16 Abundance")
    plt.xlabel("Age (yr)")
    plt.ylabel("Center O16 Abundance")
    plt.title(run_name)
    #plt.axvline(x=t_eruption, color='r', linestyle='--', label="Eruption Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_center_o16_vs_age.png", dpi=200)
    plt.close()

    # plot surface_c12
    plt.figure()
    plt.plot(age, df[surface_c12_col].values, label="Surface C12 Abundance")
    plt.xlabel("Age (yr)")
    plt.ylabel("Surface C12 Abundance")
    plt.title(run_name)
    #plt.axvline(x=t_eruption, color='r', linestyle='--', label="Eruption Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_surface_c12_vs_age.png", dpi=200)
    plt.close()

    # plot surface_o16
    plt.figure()
    plt.plot(age, df[surface_o16_col].values, label="Surface O16 Abundance")
    plt.xlabel("Age (yr)")
    plt.ylabel("Surface O16 Abundance")
    plt.title(run_name)
    #plt.axvline(x=t_eruption, color='r', linestyle='--', label="Eruption Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_surface_o16_vs_age.png", dpi=200)
    plt.close()

    # plot surface_o16
    plt.figure()
    plt.plot(age, df[surface_o16_col].values, label="Surface O16 Abundance")
    plt.xlabel("Age (yr)")
    plt.ylabel("Surface O16 Abundance")
    plt.title(run_name)
    #plt.axvline(x=t_eruption, color='r', linestyle='--', label="Eruption Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_surface_o16_vs_age.png", dpi=200)
    plt.close()

    # plot total_mass_h1
    plt.figure()
    plt.plot(age, df[total_mass_h1_col].values, label="Total Mass H1")
    plt.xlabel("Age (yr)")
    plt.ylabel("Total Mass H1")
    plt.title(run_name)
    #plt.axvline(x=t_eruption, color='r', linestyle='--', label="Eruption Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{run_name}_total_mass_h1_vs_age.png", dpi=200)
    plt.close()

# -------------------------------------------------
# Write summary CSV
# -------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

print(f"[DONE] Summary written to {SUMMARY_CSV}")
