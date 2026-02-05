import os
import glob
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ============================================================
# LFWC Post-Disaster AI Triage Dashboard (Local + Cloud Safe)
# Includes:
#  - Dashboard tab (map + queues)
#  - Image Review tab (shows photos the model used, if available)
#  - Artifacts tab (CSVs + training vs holdout, if available)
#
# This version avoids any hard-coded C:\ paths.
# It resolves files relative to this app.py file, so it runs:
#   - locally
#   - on Streamlit Community Cloud
# ============================================================

st.set_page_config(layout="wide")
st.title("LFWC Post-Disaster AI Triage Dashboard (Demo)")

# ----------------------------
# Path helpers (portable)
# ----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def p(*parts):
    """Join paths relative to app folder."""
    return os.path.join(APP_DIR, *parts)

def find_first_existing(paths):
    """Return first existing path from a list."""
    for path in paths:
        if path and os.path.exists(path):
            return path
    return ""

def safe_exists(path):
    return bool(path) and os.path.exists(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def list_images(folder, limit=12):
    if not safe_exists(folder):
        return []
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs.extend(glob.glob(os.path.join(folder, ext)))
    imgs = sorted(imgs, key=os.path.getmtime, reverse=True)
    return imgs[:limit]

def marker_color(needs_review: int, priority: float) -> str:
    # Warm palette inspired by your StoryMap
    if int(needs_review) == 1:
        return "#9B5DE5"   # attention color
    if priority >= 70:
        return "#8C2F1B"   # deep burnt red
    if priority >= 60:
        return "#C84B31"   # ember red-orange
    if priority >= 50:
        return "#D4A017"   # goldenrod
    return "#E6D5B8"       # warm sand

def find_best_image_path(value: str, folder: str):
    """
    BestMeterImg / BestLocationImg may be:
    - a bare filename
    - a relative path
    - an absolute path (local only)
    We try:
      1) absolute
      2) APP_DIR + relative
      3) folder + filename
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = str(value).strip()

    # 1) absolute
    if os.path.isabs(value) and os.path.exists(value):
        return value

    # 2) relative to app folder
    rel1 = p(value)
    if os.path.exists(rel1):
        return rel1

    # 3) inside a known folder (meter/location)
    rel2 = os.path.join(folder, value)
    if os.path.exists(rel2):
        return rel2

    return ""

# ----------------------------
# Expected repo structure (portable)
# ----------------------------
# Prefer data/ folder, but allow root fallback too:
DASHBOARD_CSV = find_first_existing([
    p("data", "dashboard_data.csv"),
    p("dashboard_data.csv"),
])

PRED_CSV = find_first_existing([
    p("data", "predictions_with_priority.csv"),
    p("predictions_with_priority.csv"),
])

HOLDOUT_CSV = find_first_existing([
    p("data", "holdout_predictions.csv"),
    p("holdout_predictions.csv"),
])

NEEDSREVIEW_HOLDOUT_CSV = find_first_existing([
    p("data", "holdout_needs_review_queue.csv"),
    p("holdout_needs_review_queue.csv"),
])

# Optional image folders (will exist locally, usually not in cloud)
METER_IMG_DIR = find_first_existing([p("data", "meter"), p("meter")])
LOC_IMG_DIR = find_first_existing([p("data", "location"), p("location")])

# Optional training folders (likely local only)
TRAIN_METER_DIR = find_first_existing([p("datasets", "meter", "train")])
TRAIN_LOC_DIR = find_first_existing([p("datasets", "location", "train")])

# Holdout gallery (optional)
HOLDOUT_GALLERY_DIR = find_first_existing([p("gallery_holdout")])
HOLDOUT_GALLERY_HTML = find_first_existing([p("gallery_holdout", "index.html")])
HOLDOUT_GALLERY_IMG_DIR = find_first_existing([p("gallery_holdout", "images")])

# ----------------------------
# Load dashboard data (required)
# ----------------------------
if not DASHBOARD_CSV:
    st.error("Missing dashboard_data.csv. Put it in either:\n- data/dashboard_data.csv (recommended)\n- dashboard_data.csv (repo root)")
    st.write("Files in app directory:", os.listdir(APP_DIR))
    st.stop()

df = load_csv(DASHBOARD_CSV)

required_cols = ["X", "Y", "PredMeter", "MeterConf", "PredLocation", "LocationConf", "NeedsReview", "PriorityScore"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"dashboard_data.csv missing columns: {missing}")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Coerce numeric
for c in ["X", "Y", "MeterConf", "LocationConf", "PriorityScore"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["NeedsReview"] = pd.to_numeric(df["NeedsReview"], errors="coerce").fillna(0).astype(int)
df["MinConf"] = df[["MeterConf", "LocationConf"]].min(axis=1)

HAS_BEST_METER = "BestMeterImg" in df.columns
HAS_BEST_LOC = "BestLocationImg" in df.columns

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Image Review", "Artifacts"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:
    st.sidebar.header("Filters")

    min_priority = st.sidebar.slider("Min PriorityScore", 0, 100, 40)
    review_only = st.sidebar.checkbox("Only NeedsReview = 1", value=False)

    conf_cut = st.sidebar.slider("Low-confidence threshold", 0.0, 1.0, 0.65)
    low_conf_only = st.sidebar.checkbox("Only low-confidence cases", value=False)

    meter_choices = ["(all)"] + sorted([x for x in df["PredMeter"].dropna().unique()])
    loc_choices = ["(all)"] + sorted([x for x in df["PredLocation"].dropna().unique()])
    meter_filter = st.sidebar.selectbox("Filter PredMeter", meter_choices)
    loc_filter = st.sidebar.selectbox("Filter PredLocation", loc_choices)

    f = df[df["PriorityScore"] >= min_priority].copy()
    if review_only:
        f = f[f["NeedsReview"] == 1]
    if low_conf_only:
        f = f[f["MinConf"] < conf_cut]
    if meter_filter != "(all)":
        f = f[f["PredMeter"] == meter_filter]
    if loc_filter != "(all)":
        f = f[f["PredLocation"] == loc_filter]

    st.write(f"Records shown: {len(f)} of {len(df)}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total evaluated (demo)", len(df))
    k2.metric("High priority (>=60)", int((df["PriorityScore"] >= 60).sum()))
    k3.metric("Needs review", int(df["NeedsReview"].sum()))
    k4.metric("Low confidence (< cutoff)", int((df["MinConf"] < conf_cut).sum()))

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Map: Priority & Review")

        if len(f) > 0 and f["Y"].notna().any() and f["X"].notna().any():
            center_lat = float(f["Y"].mean())
            center_lon = float(f["X"].mean())
        else:
            center_lat, center_lon = 34.20, -118.14

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            control_scale=True,
            tiles="CartoDB positron"
        )

        f_map = f.dropna(subset=["X", "Y"]).copy()
        for _, r in f_map.iterrows():
            pr = float(r.get("PriorityScore", 0))
            nr = int(r.get("NeedsReview", 0))
            c = marker_color(nr, pr)

            street = str(r.get("StreetAddress", "")) if "StreetAddress" in r else ""
            caps = str(r.get("AddressCAPS", "")) if "AddressCAPS" in r else ""
            apn = str(r.get("APN", "")) if "APN" in r else ""

            popup = f"""
            <b>Address:</b> {street or caps}<br>
            <b>APN:</b> {apn}<br>
            <b>PriorityScore:</b> {pr:.1f}<br>
            <b>NeedsReview:</b> {nr}<br>
            <b>PredMeter:</b> {r.get('PredMeter','')} (conf {float(r.get('MeterConf',0)):.2f})<br>
            <b>PredLocation:</b> {r.get('PredLocation','')} (conf {float(r.get('LocationConf',0)):.2f})<br>
            <b>MinConf:</b> {float(r.get('MinConf',0)):.2f}<br>
            """

            folium.CircleMarker(
                location=[float(r["Y"]), float(r["X"])],
                radius=6,
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.75,
                popup=folium.Popup(popup, max_width=380),
            ).add_to(m)

        st_folium(m, width=950, height=600)

    with right:
        st.subheader("Queues")

        st.write("Top 10 PriorityScore")
        top10 = f.sort_values("PriorityScore", ascending=False).head(10)
        show_cols = ["OBJECTID", "PredMeter", "MeterConf", "PredLocation", "LocationConf", "NeedsReview", "PriorityScore"]
        st.dataframe(top10[[c for c in show_cols if c in top10.columns]], use_container_width=True)

        st.write("NeedsReview queue (lowest confidence first)")
        queue = f.sort_values(["NeedsReview", "MinConf"], ascending=[False, True]).head(15)
        queue_cols = ["OBJECTID", "PredMeter", "MeterConf", "PredLocation", "LocationConf",
                      "NeedsReview", "PriorityScore", "MinConf"]
        st.dataframe(queue[[c for c in queue_cols if c in queue.columns]], use_container_width=True)

        st.subheader("Distributions")
        st.write("Meter counts")
        st.write(f["PredMeter"].value_counts(dropna=False))

        st.write("Location counts")
        st.write(f["PredLocation"].value_counts(dropna=False))

# ============================================================
# TAB 2: IMAGE REVIEW
# ============================================================
with tab2:
    st.subheader("Image Review (Evidence Viewer)")
    st.write("This shows the photos the model used (if images are available in this environment).")

    if "OBJECTID" not in df.columns:
        st.warning("No OBJECTID column found.")
    else:
        chosen = st.selectbox("Choose OBJECTID", df["OBJECTID"].tolist())
        row = df[df["OBJECTID"] == chosen].iloc[0]

        meta_cols = ["OBJECTID", "PriorityScore", "NeedsReview", "PredMeter", "MeterConf", "PredLocation", "LocationConf", "MinConf"]
        if "StreetAddress" in df.columns:
            meta_cols.insert(1, "StreetAddress")
        elif "AddressCAPS" in df.columns:
            meta_cols.insert(1, "AddressCAPS")
        if "APN" in df.columns:
            meta_cols.insert(2, "APN")

        st.write(row[[c for c in meta_cols if c in df.columns]])

        meter_img = ""
        loc_img = ""

        if HAS_BEST_METER and METER_IMG_DIR:
            meter_img = find_best_image_path(row.get("BestMeterImg", ""), METER_IMG_DIR)
        if HAS_BEST_LOC and LOC_IMG_DIR:
            loc_img = find_best_image_path(row.get("BestLocationImg", ""), LOC_IMG_DIR)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Meter image**")
            if meter_img and os.path.exists(meter_img):
                st.image(meter_img, use_column_width=True)
            else:
                st.info("No meter image available here (expected on Streamlit Cloud unless you upload images).")

        with c2:
            st.markdown("**Location image**")
            if loc_img and os.path.exists(loc_img):
                st.image(loc_img, use_column_width=True)
            else:
                st.info("No location image available here (expected on Streamlit Cloud unless you upload images).")

# ============================================================
# TAB 3: ARTIFACTS
# ============================================================
with tab3:
    st.subheader("Artifacts & Reproducibility")
    st.write("This section lists the outputs used to build the demo. Some artifacts exist locally only.")

    st.markdown("### Key output files")
    files = [
        ("Dashboard data", DASHBOARD_CSV),
        ("Predictions + priority", PRED_CSV),
        ("Holdout predictions (unlabeled)", HOLDOUT_CSV),
        ("Holdout NeedsReview queue", NEEDSREVIEW_HOLDOUT_CSV),
        ("Holdout gallery HTML", HOLDOUT_GALLERY_HTML),
    ]

    for name, path in files:
        if safe_exists(path):
            st.write(f"✅ {name}: {path}")
        else:
            st.write(f"⚠️ {name}: not found in this environment")

    st.markdown("### Download CSVs")
    for name, path in [("Predictions + priority", PRED_CSV),
                       ("Holdout predictions", HOLDOUT_CSV),
                       ("Holdout NeedsReview queue", NEEDSREVIEW_HOLDOUT_CSV)]:
        if safe_exists(path):
            with open(path, "rb") as f:
                st.download_button(f"Download: {name}", data=f, file_name=os.path.basename(path))

    st.markdown("### Training data preview (local only)")
    colA, colB = st.columns(2)
    with colA:
        st.write("Meter training folder:", TRAIN_METER_DIR if TRAIN_METER_DIR else "(not available here)")
        imgs = list_images(TRAIN_METER_DIR, limit=8) if TRAIN_METER_DIR else []
        if imgs:
            st.image(imgs, caption=[os.path.basename(x) for x in imgs], use_column_width=True)
        else:
            st.info("Training images not available here (expected on Streamlit Cloud unless uploaded).")

    with colB:
        st.write("Location training folder:", TRAIN_LOC_DIR if TRAIN_LOC_DIR else "(not available here)")
        imgs = list_images(TRAIN_LOC_DIR, limit=8) if TRAIN_LOC_DIR else []
        if imgs:
            st.image(imgs, caption=[os.path.basename(x) for x in imgs], use_column_width=True)
        else:
            st.info("Training images not available here (expected on Streamlit Cloud unless uploaded).")

    st.markdown("### Holdout gallery preview (local only unless uploaded)")
    if safe_exists(HOLDOUT_GALLERY_HTML):
        st.write("Holdout gallery HTML:", HOLDOUT_GALLERY_HTML)

    imgs = list_images(HOLDOUT_GALLERY_IMG_DIR, limit=10) if HOLDOUT_GALLERY_IMG_DIR else []
    if imgs:
        st.image(imgs, caption=[os.path.basename(x) for x in imgs], use_column_width=True)
    else:
        st.info("Holdout gallery images not available here unless uploaded.")
