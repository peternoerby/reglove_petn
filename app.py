import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io

# --- App setup ---
st.set_page_config(layout="wide")
st.title("Glove Simulation")

# --- Sidebar Inputs in Expanders ---
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("Initial"):
    use = st.number_input("Daily use (use)", min_value=1, value=1000, step=1)
    initial_days = st.number_input(
        "Initial supply period (days)",
        min_value=1, value=14, step=1,
        help="Hvor mange dages forbrug skal puljen dække?"
    )
    pool_init = use * initial_days
    sim_days = st.number_input("Simulation days", min_value=1, value=365, step=1)

with st.sidebar.expander("Washing"):
    discard_pct = st.slider("Discard % on use", 0, 100, 20)
    wash_time = st.number_input("Wash time (days_out threshold)", 1, 7, 3)

with st.sidebar.expander("Periodic removal"):
    remove_interval = st.number_input(
        "Periodic removal interval (days)", 0, sim_days, 0,
        help="0 = aldrig periodisk fjernelse"
    )
    remove_count = st.number_input("Periodic removal count (f)", 0, pool_init, 0)

with st.sidebar.expander("Adding gloves"):
    in_interval = st.number_input("New-gloves interval (days)", 1, sim_days, 1)
    eq_mode = st.checkbox("Equilibrium mode")
    if eq_mode:
        discarded_per_interval = use * discard_pct / 100 * in_interval
        periodic_removed = remove_count * (in_interval / remove_interval) if remove_interval > 0 else 0
        in_amm_eq = int(round(discarded_per_interval + periodic_removed))
        in_amm = st.number_input(
            "New gloves per interval (y)", value=in_amm_eq,
            disabled=True, help="Låst i equilibrium mode"
        )
    else:
        in_amm = st.number_input("New gloves per interval (y)", 0, 10_000, 200)

# --- Simulation function ---
def simulate_population(use, pool_init, wash_time, sim_days,
                        discard_pct, in_interval, in_amm,
                        remove_interval, remove_count, seed=42):
    max_new = in_amm * ((sim_days // in_interval) + 1)
    max_total = pool_init + max_new
    active = np.zeros((max_total, 3), int)
    dead = np.zeros((max_total, 3), int)
    active[:pool_init, 2] = 0
    active_count = pool_init
    dead_count = 0
    rng = np.random.default_rng(seed)
    active_counts, dead_counts = [], []
    age_active, age_dead = [], []

    for t in range(sim_days):
        # days_out update
        mask = active[:active_count,1] > 0
        active[:active_count,1][mask] += 1
        active[:active_count,1][active[:active_count,1] > wash_time] = 0
        # daily use
        ready = np.where(active[:active_count,1] == 0)[0]
        if ready.size < use:
            break
        sel = rng.choice(ready, size=use, replace=False)
        # discard
        n_discard = int(use * discard_pct / 100)
        discard_idx = rng.choice(sel, size=n_discard, replace=False) if n_discard>0 else np.array([],int)
        keep_idx = np.array([i for i in sel if i not in discard_idx], int)
        for idx in discard_idx:
            dead[dead_count] = [active[idx,0], active[idx,2], t]
            dead_count += 1
            active[idx] = active[active_count-1]
            active_count -= 1
        # wash
        active[keep_idx,0] += 1
        active[keep_idx,1] = 1
        # periodic removal
        if remove_interval>0 and t>0 and t%remove_interval==0 and remove_count>0:
            rem = min(remove_count, active_count)
            idxs = rng.choice(np.arange(active_count), size=rem, replace=False)
            for idx in idxs:
                dead[dead_count] = [active[idx,0], active[idx,2], t]
                dead_count += 1
                active[idx] = active[active_count-1]
                active_count -= 1
        # inflow
        if t>0 and t%in_interval==0 and in_amm>0:
            for _ in range(in_amm):
                active[active_count] = [0,0,t]
                active_count += 1
        # record metrics
        active_counts.append(active_count)
        dead_counts.append(dead_count)
        age_active.append((t - active[:active_count,2]).mean() if active_count>0 else 0)
        ages_d = dead[:dead_count,2] - dead[:dead_count,1]
        age_dead.append(ages_d.mean() if dead_count>0 else 0)

    return (np.array(active_counts), np.array(dead_counts),
            np.array(age_active), np.array(age_dead),
            active[:active_count], dead[:dead_count])

# --- Run simulation ---
active_ct, dead_ct, age_act, age_dead, active_arr, dead_arr = simulate_population(
    use, pool_init, wash_time, sim_days,
    discard_pct, in_interval, in_amm,
    remove_interval, remove_count
)

# --- Prepare DataFrames and lifetimes ---
active_df = pd.DataFrame(active_arr, columns=["washed_count","days_out","generated_day"])
dead_df   = pd.DataFrame(dead_arr,   columns=["washed_count","generated_day","t_out"])
active_life = len(active_ct) - active_df["generated_day"]
dead_life   = dead_df["t_out"] - dead_df["generated_day"]



# ── Extra metrics ──────────────────────────────────────────────────────────
total_gloves_used   = int(active_ct[-1] + dead_ct[-1])        # ever existed
baseline_single_use = use * sim_days                          # disposable baseline
gloves_saved        = baseline_single_use - total_gloves_used

never_used_active = (active_df["washed_count"] == 0).sum()
never_used_dead   = (dead_df["washed_count"]   == 0).sum()
never_used_total  = never_used_active + never_used_dead

one_time_active = (active_df["washed_count"] == 1).sum()
one_time_dead   = (dead_df["washed_count"]   == 1).sum()
one_time_total  = one_time_active + one_time_dead

# wash-count percentiles
wash_p90_active = np.percentile(active_df["washed_count"], 90)
wash_p99_active = np.percentile(active_df["washed_count"], 99)
wash_p90_dead   = np.percentile(dead_df["washed_count"],   90)
wash_p99_dead   = np.percentile(dead_df["washed_count"],   99)


def fmt(x, ndigits=0):
    return f"{x:,.{ndigits}f}" if isinstance(x, float) else f"{x:,}"

inputs = [
    f"- Daily use: **{fmt(use)}** gloves",
    f"- Starting pool: **{fmt(pool_init)}**  ({initial_days} days × daily use)",
    f"- Simulation length: **{sim_days}** days",
    f"- Discard after use: **{discard_pct}%**",
    f"- Wash time: **{wash_time}** days",
]

# periodic removal
inputs.append(
    "- Periodic removal: **never**"
    if remove_interval == 0 or remove_count == 0
    else f"- Periodic removal: every **{remove_interval}** dsay remove "
         f"**{fmt(remove_count)}**"
)

# inflow
inputs.append(
    f"- Inflow: every **{in_interval}** day add **{fmt(in_amm)}** new gloves"
)

outputs = [
    f"- Gloves used over **{sim_days}** days: **{fmt(total_gloves_used)}**",
    f"- Final active gloves: **{fmt(int(active_ct[-1]))}**",
    f"- Final discarded gloves: **{fmt(int(dead_ct[-1]))}**",
    f"- Gloves saved vs single-use baseline: **{fmt(gloves_saved)}** "
      f"({gloves_saved / baseline_single_use:.1%})",
    f"- Never-washed gloves: **{fmt(never_used_total)}**",
    f"    - Never used: **{fmt(never_used_active)}**",
    f"    - Used once: **{fmt(one_time_total)}**",
    "",
    "Wash statistics:",
    f"- Average washes per ACTIVE glove: **{active_df['washed_count'].mean():.2f}**",
    f"- Average washes per DISCARDED glove: **{dead_df['washed_count'].mean():.2f}**",
    f"- 10 % of ACTIVE gloves are washed ≥ **{wash_p90_active:.0f}** times",
    f"- 1 % of ACTIVE gloves are washed ≥ **{wash_p99_active:.0f}** times",
    f"- 10 % of DISCARDED gloves were washed ≥ **{wash_p90_dead:.0f}** times",
    f"- 1 % of DISCARDED gloves were washed ≥ **{wash_p99_dead:.0f}** times",
    "",
    "Lifetime statistics:",
    f"- 10 % of ACTIVE gloves stay in the pool ≥ **{np.percentile(active_life, 90):.0f}** days",
    f"- 1 % of ACTIVE gloves stay in the pool ≥ **{np.percentile(active_life, 99):.0f}** days",
    f"- 10 % of DISCARDED gloves stayed in the pool ≥ **{np.percentile(dead_life, 90):.0f}** days",
    f"- 1 % of DISCARDED gloves stayed in the pool ≥ **{np.percentile(dead_life, 99):.0f}** days",
]


# ─────────────────────  Plain-text version for PDF  ───────────────────────
def md2txt(s: str) -> str: return s.replace("**", "")
report_text = (
    "INPUTS\n"
    "------\n" + "\n".join(md2txt(x) for x in inputs) + "\n\n"
    "OUTPUTS\n"
    "-------\n" + "\n".join(md2txt(x) for x in outputs)
)


# --- Time-series plots side by side ---
col1, col2 = st.columns(2)
with col1:
    fig_pop, ax_pop = plt.subplots()
    ax_pop.plot(np.arange(len(active_ct)), active_ct, label="Active")
    ax_pop.plot(np.arange(len(dead_ct)),   dead_ct,   label="Dead")
    ax_pop.set(xlabel="Day", ylabel="Count", title="Population over Time")
    ax_pop.legend()
    st.pyplot(fig_pop)
with col2:
    fig_age, ax_age = plt.subplots()
    ax_age.plot(np.arange(len(age_act)),  age_act,  label="Active age")
    ax_age.plot(np.arange(len(age_dead)), age_dead, label="Dead age")
    ax_age.set(xlabel="Day", ylabel="Age (days)", title="Average Age over Time")
    ax_age.legend()
    st.pyplot(fig_age)

# --- Histograms with Avg/90%/99% lines ---
st.subheader("Post-simulation Histograms")
fig_hist, axes = plt.subplots(2,2, figsize=(12,8))
datasets = [
    (active_df["washed_count"], "Usage (active)"),
    (dead_df["washed_count"],   "Usage (dead)"),
    (active_life,               "Lifetime (active)"),
    (dead_life,                 "Lifetime (dead)")
]
for ax, (data, title) in zip(axes.ravel(), datasets):
    weights = np.ones_like(data) / len(data) * 100
    avg = data.mean()
    p90 = np.percentile(data, 90)
    p99 = np.percentile(data, 99)
    bins = np.arange(data.min(), data.max()+2) - 0.5
    ax.hist(data, bins=bins, weights=weights)


    
    ax.axvline(avg, color='gray', linestyle='dotted', label='Avg')
    ax.axvline(p90, color='black', linestyle='dotted', label='90%')
    ax.axvline(p99, color='red',   linestyle='dotted', label='99%')
    # ax.set(title=title, xlabel="Value", ylabel="Percentage (%)")
            # dynamisk xlabel efter type
    if "Usage" in title:
            xlabel = "Wash count"
    else:
            xlabel = "Days"
    ax.legend()
    # ax.text(0.5, -0.25, f"Avg: {avg:.2f}\n90%: {p90:.0f}\n99%: {p99:.0f}",
    #         transform=ax.transAxes, ha='center', va='top')
    ax.set(title=title, xlabel=xlabel, ylabel="Percentage (%)")
    ax.text(0.5, -0.20,
                    f"Avg: {avg:.2f} - 90%: {p90:.0f} - 99%: {p99:.0f}",
                    transform=ax.transAxes, ha='center', va='top')
plt.tight_layout()
st.pyplot(fig_hist)


if st.button("Download report as PDF"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # page 1 – inputs & outputs ----------------------------------------
        fig0, ax0 = plt.subplots(figsize=(8.27, 11.69))
        ax0.axis("off")
        ax0.text(0.03, 0.97, report_text,
                 family="monospace", fontsize=10, ha="left", va="top", wrap=True)
        pdf.savefig(fig0); plt.close(fig0)

        # --- Side 2: A4 med tidsserier øverst, histogram midt, whitespace nederst ---
        fig = plt.figure(figsize=(8.27,11.69))

        # Beregninger af højder (i tommer):
        hist_h = 8.27 / 1.5               # 5.513" → 1.5:1 på de fire histogrammer
        ts_h   = (8.27 / 2) * (3/4)       # 3.101" → 4:3 på hvert af tidsserierne
        ws_h   = 11.69 - hist_h - ts_h    # resten → whitespace nederst

        # 3 vandrette bånd: tidsserier, histogram, whitespace
        gs = fig.add_gridspec(
            nrows=3, ncols=1,
            height_ratios=[ts_h, hist_h, ws_h],
            hspace=0.3
        )

        # --- Tidsserier (øverste bånd) ---
        gs_ts = gs[0].subgridspec(1, 2, wspace=0.3)
        ax_pop = fig.add_subplot(gs_ts[0, 0])
        ax_age = fig.add_subplot(gs_ts[0, 1])

        ax_pop.plot(np.arange(len(active_ct)), active_ct, label="Active")
        ax_pop.plot(np.arange(len(dead_ct)),   dead_ct,   label="Dead")
        ax_pop.set(title="Population over Time", xlabel="Day", ylabel="Count")
        ax_pop.legend()

        ax_age.plot(np.arange(len(age_act)),  age_act,  label="Active age")
        ax_age.plot(np.arange(len(age_dead)), age_dead, label="Dead age")
        ax_age.set(title="Average Age over Time", xlabel="Day", ylabel="Age (days)")
        ax_age.legend()

        # --- Histograms (midterste bånd) med ekstra lodret mellemrum ---
        gs_hist = gs[1].subgridspec(2, 2, hspace=0.8, wspace=0.3)
        axes = [fig.add_subplot(gs_hist[i, j]) for i in range(2) for j in range(2)]

        datasets = [
            (active_df["washed_count"],                    "Usage (active)"),
            (dead_df["washed_count"],                      "Usage (dead)"),
            (len(active_ct) - active_df["generated_day"],  "Lifetime (active)"),
            (dead_df["t_out"] - dead_df["generated_day"],  "Lifetime (dead)")
        ]
        for ax, (data, title) in zip(axes, datasets):
            weights = np.ones_like(data) / len(data) * 100
            avg = data.mean()
            p90 = np.percentile(data, 90)
            p99 = np.percentile(data, 99)
            bins = np.arange(data.min(), data.max()+2) - 0.5

            ax.hist(data, bins=bins, weights=weights)
            ax.axvline(avg, linestyle='dotted')
            ax.axvline(p90, linestyle='dotted')
            ax.axvline(p99, linestyle='dotted', color='red')

            # dynamisk xlabel efter type
            if "Usage" in title:
                xlabel = "Wash count"
            else:
                xlabel = "Days"

            ax.set(title=title, xlabel=xlabel, ylabel="Percentage (%)")
            ax.text(0.5, -0.4,
                    f"Avg: {avg:.2f} - 90%: {p90:.0f} - 99%: {p99:.0f}",
                    transform=ax.transAxes, ha='center', va='top')

        # --- Nederste bånd er nu blank whitespace ---

        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    st.download_button(
        "Click to download your report",
        buffer, "glove_report.pdf", "application/pdf"
    )



# --- Display Inputs & Outputs ---
col_in, col_out = st.columns(2)
# ── Nicely formatted Inputs ────────────────────────────────────────────────

if eq_mode:
    inputs.append("- Equilibrium mode: **on**")

with col_in:
    st.subheader("Inputs")
    st.markdown("\n".join(inputs))
with col_out:
    st.subheader("Outputs")
    st.markdown("\n".join(outputs))
