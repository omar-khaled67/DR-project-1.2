# """
# RetinoGrade demo page that talks to the API. It loads NO model: all analysis happens in api_server.py.

#     1) uvicorn api_server:app --host 127.0.0.1 --port 8000
#     2) streamlit run app/streamlit_client.py
#     RG_API_URL   address of the API (default http://127.0.0.1:8000)
# """
# import io
# import os

# import altair as alt
# import numpy as np
# import pandas as pd
# import requests
# import streamlit as st
# from PIL import Image

# API_URL = os.environ.get("RG_API_URL", "http://127.0.0.1:8000").rstrip("/")
# st.set_page_config(page_title="RetinoGrade", page_icon="👁", layout="wide")

# MAX_MB = 25
# SHOWN_SIDE = 1000
# GRADE_COLORS = ["#388e3c", "#8d8b1e", "#ef6c00", "#e64a19", "#d32f2f"]
# CACHE = {"show_spinner": False, "ttl": 600}


# class ApiError(Exception):
#     pass


# def call(path: str, data: bytes, params=None) -> dict:
#     try:
#         r = requests.post(f"{API_URL}{path}", files={"file": ("photo", data)}, params=params, timeout=180)
#     except requests.exceptions.ConnectionError:
#         raise ApiError(f"Cannot reach the API at {API_URL}. Start it first: uvicorn api_server:app --port 8000") from None
#     except requests.exceptions.Timeout:
#         raise ApiError("The API took too long to answer.") from None
#     if r.status_code != 200:
#         try:
#             detail = r.json().get("detail", r.text)
#         except ValueError:
#             detail = r.text
#         raise ApiError(str(detail))
#     return r.json()


# @st.cache_data(max_entries=8, **CACHE)
# def analyse(data: bytes) -> dict:
#     return call("/analyse", data)


# @st.cache_data(max_entries=8, **CACHE)
# def heatmap(data: bytes, target: str) -> dict:
#     return call("/heatmap", data, {"target": target})


# def percent(p: float) -> str:
#     return ">99%" if p >= 0.995 else f"{p:.0%}"


# def probability_chart(probs: dict) -> alt.Chart:
#     d = pd.DataFrame({"grade": [f"{k} {n}" for k, n in enumerate(probs)], "probability": list(probs.values())})
#     return alt.Chart(d).mark_bar().encode(
#         x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
#         y=alt.Y("grade:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)),
#         tooltip=["grade", alt.Tooltip("probability:Q", format=".1%")]).properties(height=200)


# def shown(data: bytes) -> Image.Image:
#     im = Image.open(io.BytesIO(data)).convert("RGB")
#     f = SHOWN_SIDE / max(im.size)
#     return im if f >= 1 else im.resize((int(im.width * f), int(im.height * f)))


# def png(b64: str) -> Image.Image:
#     import base64
#     return Image.open(io.BytesIO(base64.b64decode(b64)))


# # ------------------------------------------------------------------ page
# st.title("RetinoGrade - diabetic retinopathy grading")
# st.caption("Upload one retinal (fundus) photo. The analysis runs on the API server.")

# with st.sidebar:
#     st.header("About")
#     st.write("One model grades the photo on the 0-4 scale. Two flags mark eyes that should be referred. "
#              "A heatmap shows where the model looked.")
#     st.markdown("**Grades:** 0 No DR · 1 Mild · 2 Moderate · 3 Severe · 4 Proliferative")
#     st.caption(f"API: {API_URL}")
#     st.warning("This is a research prototype for decision support. It must not be used to diagnose or treat patients. "
#                "Every result needs a qualified eye-care professional.")

# up = st.file_uploader("Fundus photo (PNG or JPG)", type=["png", "jpg", "jpeg"])
# if up is None:
#     st.info("Waiting for a photo.")
#     st.stop()
# data = up.getvalue()
# if len(data) > MAX_MB * 1024 * 1024:
#     st.error(f"The file is larger than {MAX_MB} MB.")
#     st.stop()

# try:
#     with st.spinner("Analysing the photo..."):
#         result = analyse(data)
# except ApiError as e:
#     st.error(str(e))
#     st.stop()

# quality = result["quality"]
# a = result["review"]
# th = result["thresholds"]
# w, h_ = result["image_size"]
# left, right = st.columns([1, 1])
# with left:
#     st.image(shown(data), caption=f"Uploaded photo ({w}×{h_})", use_container_width=True)

# with right:
#     if quality is None:
#         st.warning("The quality model is not available. Image quality was NOT checked.")
#     elif quality["action"] == "retake":
#         st.error(f"❌ {quality['message']}")
#         st.markdown("No grade is shown for a photo that cannot be judged reliably.")
#         if quality["probabilities"]:
#             st.caption("Quality check: " + " · ".join(f"{n} {v:.0%}" for n, v in quality["probabilities"].items()))
#         st.stop()
#     elif quality["action"] == "warn":
#         st.warning(f"⚠️ {quality['message']}")
#     else:
#         st.success(f"✅ {quality['message']}")

#     g = result["grade"]
#     st.markdown(f"<h2 style='color:{GRADE_COLORS[g]};margin-bottom:0'>Grade {g} - {result['grade_name']}</h2>", unsafe_allow_html=True)
#     st.metric("Calibrated confidence", percent(a["confidence_calibrated"]))
#     if a["needs_review"]:
#         st.error("👤 **Needs human review**\n\n" + "\n".join(f"- {r}" for r in a["reasons"]))
#     c1, c2 = st.columns(2)
#     c1.metric("Referable DR (grade ≥ 2)", "YES" if result["referable"] else "no",
#               help=f"P(grade ≥ 2) = {result['p_referable']:.2f}, threshold {th['referable']:.2f}")
#     c2.metric("Sight-threatening DR (grade ≥ 3)", "YES" if result["sight_threatening"] else "no",
#               help=f"P(grade ≥ 3) = {result['p_sight_threatening']:.3f}, threshold {th['sight_threatening']:.3f}")
#     st.caption("The flags use their own thresholds, so they can be on although the predicted grade is lower.")

# st.subheader("Class probabilities")
# st.altair_chart(probability_chart(result["probabilities"]), use_container_width=True)

# st.subheader("Where did the model look? (Grad-CAM)")
# choice = st.radio("Explain", ["Referral decision (grade ≥ 2)", "Predicted grade"], horizontal=True)
# try:
#     with st.spinner("Computing the heatmap..."):
#         hm = heatmap(data, "pred" if choice == "Predicted grade" else "referable")
# except ApiError as e:
#     st.error(str(e))
#     st.stop()
# h1, h2 = st.columns(2)
# h1.image(png(hm["prepared_png"]), caption="Prepared image the model sees", use_container_width=True)
# h2.image(png(hm["overlay_png"]), caption=f"Heatmap: {choice.lower()}", use_container_width=True)
# if hm["empty"]:
#     st.info("No region increased this score, so the heatmap is empty.")
# st.caption("It is coarse (12x12 cells) and does not outline lesions. Red is the strongest region of this map, not how serious it is.")

# st.divider()
# st.markdown(f"⚕️ **{result['disclaimer']}**")
