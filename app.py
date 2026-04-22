import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image

from model import UNet

NUM_CLASSES  = 23
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "unet_cityscape.pth")
RESULTS_PATH = os.path.join(BASE_DIR, "test_results.txt")
PLOTS_PATH   = os.path.join(BASE_DIR, "training_plots.png")

COLORS = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
    (0,60,100),(0,80,100),(0,0,230),(119,11,32),(0,255,0),
    (255,165,0),(255,20,147),(0,255,255)
]

@st.cache_resource
def load_model():
    model = UNet(3, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def mask_to_color(mask_np):
    h, w   = mask_np.shape
    color  = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(COLORS):
        color[mask_np == i] = c
    return color

def predict(model, img_path):
    img     = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rsz = cv2.resize(img_rgb, (128, 96), interpolation=cv2.INTER_NEAREST)
    tensor  = torch.from_numpy(img_rsz.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    return img_rsz, mask_to_color(pred)

def load_results():
    r = {}
    if os.path.exists(RESULTS_PATH):
        for line in open(RESULTS_PATH):
            k, v = line.strip().split(":")
            r[k] = float(v)
    return r

# ── APP ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CityScape Segmentation", layout="wide")
page = st.sidebar.radio("Navigate", ["📊 Page 1: Training Results", "🖼️ Page 2: Inference"])

if page == "📊 Page 1: Training Results":
    st.title("📊 CityScape UNet — Training Dashboard")
    r = load_results()
    if r:
        c1, c2 = st.columns(2)
        c1.metric("Test mIOU",  f"{r.get('mIOU', 0):.4f}")
        c2.metric("Test mDice", f"{r.get('mDice',0):.4f}")
    else:
        st.warning("Run train.py first!")

    st.subheader("Training Curves")
    if os.path.exists(PLOTS_PATH):
        st.image(PLOTS_PATH, use_column_width=True)
    else:
        st.warning("training_plots.png not found. Run train.py first.")

else:
    st.title("🖼️ CityScape UNet — Segmentation Inference")
    st.write("Upload **exactly 4** images from the test set.")
    model    = load_model()
    uploaded = st.file_uploader("Upload 4 test images", type=["png","jpg","jpeg"], accept_multiple_files=True)

    if uploaded:
        if len(uploaded) != 4:
            st.warning(f"Please upload exactly 4 images (got {len(uploaded)}).")
        else:
            cols = st.columns(4)
            for i, (f, col) in enumerate(zip(uploaded, cols)):
                tmp = f"/tmp/up_{i}.png"
                with open(tmp, "wb") as fp:
                    fp.write(f.read())
                orig, pred_color = predict(model, tmp)
                with col:
                    st.markdown(f"**Image {i+1}**")
                    st.image(orig,       caption="Input",          use_column_width=True)
                    st.image(pred_color, caption="Predicted Mask", use_column_width=True)