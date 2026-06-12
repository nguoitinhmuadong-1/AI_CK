<<<<<<< HEAD
import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="AD Food Bill",
=======
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

# =========================
# CẤU HÌNH APP
# =========================
st.set_page_config(
    page_title="AD Food Tray",
>>>>>>> da9f624 (first commit)
    page_icon="🍱",
    layout="wide"
)

<<<<<<< HEAD

# =========================
# CSS GIAO DIỆN
=======
MODEL_PATH = "food_model_final.h5"

# ĐÚNG THEO train_generator.class_indices CỦA ANH
CLASS_NAMES = [
    "ca_hu_kho",        # 0
    "canh_chua",        # 1
    "canh_rau",         # 2
    "com_trang",        # 3
    "dau_hu_sot_ca",    # 4
    "rau_xao",          # 5
    "suon_nuong",       # 6
    "thit_kho",         # 7
    "thit_kho_trung",   # 8
    "trung_chien"       # 9
]

DISPLAY_NAMES = {
    "ca_hu_kho": "Cá hú kho",
    "canh_chua": "Canh chua",
    "canh_rau": "Canh rau",
    "com_trang": "Cơm trắng",
    "dau_hu_sot_ca": "Đậu hũ sốt cà",
    "rau_xao": "Rau xào",
    "suon_nuong": "Sườn nướng",
    "thit_kho": "Thịt kho",
    "thit_kho_trung": "Thịt kho trứng",
    "trung_chien": "Trứng chiên"
}

PRICE_TABLE = {
    "ca_hu_kho": 30000,
    "canh_chua": 10000,
    "canh_rau": 7000,
    "com_trang": 10000,
    "dau_hu_sot_ca": 25000,
    "rau_xao": 10000,
    "suon_nuong": 30000,
    "thit_kho": 25000,
    "thit_kho_trung": 30000,
    "trung_chien": 25000
}

# Vùng cắt 5 ô theo tỉ lệ ảnh khay mẫu
# Format: x1, y1, x2, y2
ROI_RATIOS = {
    "Ô trên trái":  (0.03, 0.08, 0.31, 0.52),
    "Ô trên giữa": (0.31, 0.06, 0.57, 0.52),
    "Ô trên phải": (0.57, 0.04, 0.98, 0.54),
    "Ô dưới trái": (0.03, 0.47, 0.44, 0.98),
    "Ô dưới phải": (0.44, 0.49, 0.98, 0.98),
}

BOX_COLORS = {
    "Ô trên trái": "red",
    "Ô trên giữa": "blue",
    "Ô trên phải": "green",
    "Ô dưới trái": "orange",
    "Ô dưới phải": "purple",
}


# =========================
# CSS
>>>>>>> da9f624 (first commit)
# =========================
st.markdown(
    """
    <style>
    .stApp {
<<<<<<< HEAD
        background: linear-gradient(135deg, #fff8e7, #ffe6c7);
        color: #1f1f1f;
    }

    html, body, [class*="css"] {
        color: #1f1f1f !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #1f1f1f !important;
    }

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 900;
        color: #7a3e00 !important;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #333333 !important;
        margin-bottom: 30px;
    }

    .card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 18px;
        border: 2px solid #f2c078;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
        margin-bottom: 15px;
        color: #1f1f1f !important;
    }

    .card b {
        color: #7a3e00 !important;
    }

    .total-box {
        background-color: #fff3cd;
        border: 2px solid #e0a800;
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .total-text {
        font-size: 36px;
        font-weight: 900;
        color: #d35400 !important;
    }

    .stButton > button {
        background-color: #d35400 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.6rem 1rem !important;
    }

    .stButton > button:hover {
        background-color: #a84300 !important;
        color: white !important;
    }

    .stFileUploader label,
    .stCameraInput label {
        color: #1f1f1f !important;
        font-weight: 600 !important;
    }

    button[data-baseweb="tab"] {
        color: #1f1f1f !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #d35400 !important;
    }

    .stAlert div {
        color: #1f1f1f !important;
    }

    /* Làm vùng bảng sáng hơn */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
        padding: 8px !important;
        border: 2px solid #f2c078 !important;
        color: #1f1f1f !important;
    }

    div[data-testid="stDataFrame"] div,
    div[data-testid="stDataEditor"] div {
        color: #1f1f1f !important;
    }

    /* Dropdown trong data_editor */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #f2c078 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #1f1f1f !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] svg {
        color: #1f1f1f !important;
        fill: #1f1f1f !important;
    }

    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border-radius: 10px !important;
    }

    ul[role="listbox"],
    div[role="listbox"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #f2c078 !important;
    }

    li[role="option"],
    div[role="option"] {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }

    li[role="option"] span,
    div[role="option"] span {
        color: #1f1f1f !important;
    }

    li[role="option"]:hover,
    div[role="option"]:hover {
        background-color: #ffe6c7 !important;
        color: #1f1f1f !important;
    }

    li[aria-selected="true"],
    div[aria-selected="true"] {
        background-color: #ffd9a8 !important;
        color: #1f1f1f !important;
    }

    li[aria-selected="true"] span,
    div[aria-selected="true"] span {
        color: #1f1f1f !important;
=======
        background: linear-gradient(135deg, #fff7e6, #ffe0b2, #fff3e0);
    }

    .title {
        text-align: center;
        font-size: 46px;
        font-weight: 900;
        color: #5d2e0c;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #7a4a1d;
        margin-bottom: 25px;
    }

    .note-box {
        background-color: #fff8e1;
        color: #4e2600;
        border-left: 8px solid #ff9800;
        padding: 18px;
        border-radius: 14px;
        font-size: 17px;
        margin-bottom: 20px;
    }

    .total-box {
        background-color: white;
        border: 3px solid #ff9800;
        border-radius: 22px;
        padding: 25px;
        text-align: center;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.15);
    }

    .total-title {
        font-size: 25px;
        font-weight: 800;
        color: #5d2e0c;
    }

    .total-money {
        font-size: 48px;
        font-weight: 900;
        color: #d35400;
    }

    div.stButton > button {
        background-color: #ff9800;
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        padding: 10px 22px;
    }

    div.stButton > button:hover {
        background-color: #e67e00;
        color: white;
>>>>>>> da9f624 (first commit)
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
<<<<<<< HEAD
# LOAD MODEL
# =========================
@st.cache_resource
def load_food_model():
    return load_model("best_food_model.h5")


model = load_food_model()


# =========================
# CLASS ĐÚNG THEO MODEL TRAIN
# =========================
class_names = [
    "Cá hú kho",
    "Canh chua",
    "Canh rau",
    "Cơm trắng",
    "Đậu hũ sốt cà",
    "Rau xào",
    "Sườn nướng",
    "Thịt kho",
    "Thịt kho trứng",
    "Trứng chiên"
]


# =========================
# BẢNG GIÁ
# =========================
price_dict = {
    "Cơm trắng": 10000,
    "Đậu hũ sốt cà": 25000,
    "Cá hú kho": 30000,
    "Thịt kho trứng": 30000,
    "Thịt kho": 25000,
    "Canh chua": 10000,
    "Sườn nướng": 30000,
    "Canh rau": 7000,
    "Rau xào": 10000,
    "Trứng chiên": 25000
}


editable_food_options = [
    "Cá hú kho",
    "Canh chua",
    "Canh rau",
    "Cơm trắng",
    "Đậu hũ sốt cà",
    "Rau xào",
    "Sườn nướng",
    "Thịt kho",
    "Thịt kho trứng",
    "Trứng chiên"
]


# =========================
# FORMAT TIỀN
# =========================
def format_money(value):
    return f"{value:,.0f}".replace(",", ".") + " VNĐ"


# =========================
# SẮP XẾP 4 ĐIỂM KHAY
# =========================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# =========================
# TỰ CĂN KHAY BẰNG OPENCV
# =========================
def auto_warp_tray(pil_img, output_width=2048, output_height=1536):
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    min_area = img_area * 0.18

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h

            if 1.1 <= aspect_ratio <= 2.4:
                candidates.append((area, approx))

    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        tray_contour = candidates[0][1]

        pts = tray_contour.reshape(4, 2).astype("float32")
        rect = order_points(pts)

        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_rgb, M, (output_width, output_height))

        warped_pil = Image.fromarray(warped)

        return warped_pil, True

    h, w = img_rgb.shape[:2]

    x1 = int(w * 0.02)
    y1 = int(h * 0.18)
    x2 = int(w * 0.98)
    y2 = int(h * 0.94)

    cropped = img_rgb[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped).resize((output_width, output_height))

    return cropped_pil, False


# =========================
# CẮT 5 VÙNG MÓN ĂN
# =========================
def crop_tray_foods(warped_img):
    crop_boxes = {
        "Ô 1": (90, 90, 620, 560),
        "Ô 2": (720, 90, 1230, 560),
        "Ô 3": (1320, 90, 1960, 560),
        "Ô 4": (80, 680, 790, 1450),
        "Ô 5": (980, 680, 1950, 1450)
    }

    cropped_images = {}

    for area_name, box in crop_boxes.items():
        cropped_images[area_name] = warped_img.crop(box)

    return cropped_images, crop_boxes


# =========================
# VẼ KHUNG CROP
# =========================
def draw_boxes(img, crop_boxes):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    for name, box in crop_boxes.items():
        draw.rectangle(box, outline="red", width=6)
        draw.text((box[0] + 10, box[1] + 10), name, fill="red")

    return img_draw


# =========================
# CENTER CROP PHỤ
# =========================
def center_crop_pil(img, crop_ratio):
    w, h = img.size

    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


# =========================
# SMART CROP - CẮT SÁT MÓN ĂN HƠN
# =========================
def smart_food_crop(crop_img):
    """
    Cố gắng cắt sát vùng món ăn bên trong từng ô.
    Nếu không tìm được vùng món rõ ràng thì fallback về center crop.
    """

    img = np.array(crop_img.convert("RGB"))
    h, w = img.shape[:2]

    border_x = int(w * 0.06)
    border_y = int(h * 0.06)

    inner = img[border_y:h-border_y, border_x:w-border_x]

    hsv = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)

    h_ch, s_ch, v_ch = cv2.split(hsv)

    color_mask = (s_ch > 35).astype(np.uint8) * 255

    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    mask = cv2.bitwise_or(color_mask, edge_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return center_crop_pil(crop_img, 0.88)

    valid_boxes = []
    inner_area = inner.shape[0] * inner.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < inner_area * 0.015:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        valid_boxes.append((x, y, x + bw, y + bh))

    if len(valid_boxes) == 0:
        return center_crop_pil(crop_img, 0.88)

    x1 = min(box[0] for box in valid_boxes)
    y1 = min(box[1] for box in valid_boxes)
    x2 = max(box[2] for box in valid_boxes)
    y2 = max(box[3] for box in valid_boxes)

    margin_x = int((x2 - x1) * 0.18)
    margin_y = int((y2 - y1) * 0.18)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(inner.shape[1], x2 + margin_x)
    y2 = min(inner.shape[0], y2 + margin_y)

    x1 += border_x
    x2 += border_x
    y1 += border_y
    y2 += border_y

    crop_area = (x2 - x1) * (y2 - y1)
    full_area = w * h

    if crop_area < full_area * 0.12:
        return center_crop_pil(crop_img, 0.88)

    return crop_img.crop((x1, y1, x2, y2))


# =========================
# DỰ ĐOÁN 1 ẢNH CROP
# =========================
def predict_food(crop_img):
    """
    Dự đoán nhiều phiên bản ảnh:
    - ảnh gốc
    - ảnh cắt sát món bằng OpenCV
    - center crop nhiều mức
    - tăng contrast/sharpness nhẹ

    Sau đó lấy trung bình có trọng số để kết quả ổn định hơn.
    """

    img_width, img_height = 224, 224
    crop_img = crop_img.convert("RGB")

    smart_crop = smart_food_crop(crop_img)

    contrast_img = ImageEnhance.Contrast(crop_img).enhance(1.15)
    sharp_img = ImageEnhance.Sharpness(crop_img).enhance(1.2)

    image_versions = [
        (crop_img, 2.0),
        (smart_crop, 1.5),
        (center_crop_pil(crop_img, 0.94), 1.0),
        (center_crop_pil(crop_img, 0.88), 1.0),
        (center_crop_pil(crop_img, 0.82), 0.8),
        (contrast_img, 0.8),
        (sharp_img, 0.8)
    ]

    weighted_predictions = []
    total_weight = 0

    for img_version, weight in image_versions:
        img_version = img_version.resize((img_width, img_height))

        img_array = img_to_array(img_version)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array, verbose=0)[0]

        weighted_predictions.append(pred * weight)
        total_weight += weight

    avg_prediction = np.sum(weighted_predictions, axis=0) / total_weight
    avg_prediction = np.expand_dims(avg_prediction, axis=0)

    pred_index = int(np.argmax(avg_prediction))
    confidence = float(np.max(avg_prediction) * 100)

    food_name = class_names[pred_index]

    return food_name, confidence, avg_prediction


# =========================
# TOP 3 DỰ ĐOÁN
# =========================
def get_top3(prediction):
    top_3 = np.argsort(prediction[0])[-3:][::-1]

    results = []

    for i in top_3:
        results.append({
            "Món ăn": class_names[i],
            "Tỉ lệ": float(prediction[0][i] * 100)
        })

    return results


# =========================
# TÍNH LẠI GIÁ SAU KHI NHÂN VIÊN SỬA
# =========================
def rebuild_bill_df(df):
    df = df.copy()
    df["Giá tiền"] = df["Món ăn"].map(price_dict)
    df["Giá tiền"] = df["Giá tiền"].fillna(0).astype(int)
    return df


# =========================
# TIÊU ĐỀ APP
# =========================
st.markdown('<div class="main-title">🍱 AD Food Bill</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI nhận diện món ăn, nhân viên kiểm tra lại và xác nhận hóa đơn</div>',
    unsafe_allow_html=True
)


# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["📸 Nhận diện & xác nhận bill", "💰 Bảng giá"])


# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Tải ảnh hoặc chụp ảnh khay cơm")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Tải ảnh khay cơm",
            type=["jpg", "jpeg", "png"]
        )

    with col2:
        camera_file = st.camera_input("Chụp ảnh trực tiếp")

    image_file = uploaded_file if uploaded_file is not None else camera_file

    if image_file is not None:
        image_bytes = image_file.getvalue()
        image_key = str(hash(image_bytes))

        if st.session_state.get("current_image_key") != image_key:
            st.session_state["current_image_key"] = image_key
            st.session_state["ai_results"] = None
            st.session_state["bill_confirmed"] = False
            st.session_state["final_bill"] = None

        original_img = Image.open(image_file).convert("RGB")

        st.markdown("### Ảnh gốc")
        st.image(original_img, use_container_width=True)

        warped_img, found_tray = auto_warp_tray(original_img)

        if found_tray:
            st.success("Đã tự căn khay cơm thành công.")
        else:
            st.warning("Không tìm rõ viền khay. App sẽ crop bớt nền bàn rồi cắt tạm.")

        st.markdown("### Ảnh sau khi căn khay")
        st.image(warped_img, use_container_width=True)

        cropped_images, crop_boxes = crop_tray_foods(warped_img)
        preview_img = draw_boxes(warped_img, crop_boxes)

        st.markdown("### Vùng món ăn được cắt")
        st.image(preview_img, use_container_width=True)

        st.markdown("### Ảnh từng món sau khi crop")
        crop_cols = st.columns(len(cropped_images))

        for idx, (area_name, crop_img) in enumerate(cropped_images.items()):
            with crop_cols[idx]:
                st.image(crop_img, caption=area_name, use_container_width=True)

        if st.button("🔍 AI nhận diện món ăn", use_container_width=True):
            with st.spinner("Đang xử lý nhận diện món ăn..."):
                ai_results = []

                for area_name, crop_img in cropped_images.items():
                    food_name, confidence, prediction = predict_food(crop_img)
                    top3 = get_top3(prediction)

                    ai_results.append({
                        "Vị trí": area_name,
                        "Món AI dự đoán": food_name,
                        "Độ tin cậy": round(confidence, 2),
                        "Top 3": top3
                    })

                st.session_state["ai_results"] = ai_results
                st.session_state["bill_confirmed"] = False
                time.sleep(0.5)

        if st.session_state.get("ai_results") is not None:
            st.markdown("## Kết quả AI dự đoán")

            pred_cols = st.columns(len(cropped_images))

            for idx, item in enumerate(st.session_state["ai_results"]):
                area_name = item["Vị trí"]
                predicted_food = item["Món AI dự đoán"]
                confidence = item["Độ tin cậy"]
                top3 = item["Top 3"]
                crop_img = cropped_images[area_name]

                top3_text = ""
                for top_item in top3:
                    top3_text += f"{top_item['Món ăn']}: {top_item['Tỉ lệ']:.2f}%<br>"

                with pred_cols[idx]:
                    st.image(crop_img, caption=area_name, use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="card">
                            <b>AI dự đoán:</b> {predicted_food}<br>
                            <b>Độ tin cậy:</b> {confidence:.2f}%<br><br>
                            <b>Top 3:</b><br>
                            {top3_text}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("## Nhân viên chỉnh bill trực tiếp trên bảng")

            st.info(
                "Giữ nguyên kết quả AI ở phía trên. Nếu món nào sai, nhân viên sửa trực tiếp "
                "cột 'Món ăn' trong bảng bên dưới rồi bấm xác nhận."
            )

            review_df = pd.DataFrame([
                {
                    "Vị trí": item["Vị trí"],
                    "Món ăn": item["Món AI dự đoán"],
                    "Giá tiền": price_dict.get(item["Món AI dự đoán"], 0)
                }
                for item in st.session_state["ai_results"]
            ])

            edited_df = st.data_editor(
                review_df,
                hide_index=True,
                use_container_width=True,
                key=f"editor_{image_key}",
                column_config={
                    "Vị trí": st.column_config.TextColumn(
                        "Vị trí",
                        disabled=True
                    ),
                    "Món ăn": st.column_config.SelectboxColumn(
                        "Món ăn",
                        options=editable_food_options,
                        required=True
                    ),
                    "Giá tiền": st.column_config.NumberColumn(
                        "Giá tiền",
                        format="%d",
                        disabled=True
                    )
                },
                disabled=["Vị trí", "Giá tiền"]
            )

            edited_df = rebuild_bill_df(edited_df)

            display_df = edited_df.copy()
            display_df["Giá tiền"] = display_df["Giá tiền"].apply(format_money)

            st.markdown("### Bill tạm sau khi nhân viên chỉnh")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            temp_total = edited_df["Giá tiền"].sum()

            st.markdown(
                f"""
                <div class="total-box">
                    <div class="total-text">
                        Tạm tính: {format_money(temp_total)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("✅ Nhân viên xác nhận bill", use_container_width=True):
                with st.spinner("Đang xử lý xác nhận bill..."):
                    final_df = edited_df.copy()
                    st.session_state["final_bill"] = final_df
                    st.session_state["bill_confirmed"] = True
                    time.sleep(0.5)

        if st.session_state.get("bill_confirmed") is True and st.session_state.get("final_bill") is not None:
            st.markdown("## Hóa đơn đã xác nhận")

            final_df = st.session_state["final_bill"].copy()
            final_display_df = final_df.copy()
            final_display_df["Giá tiền"] = final_display_df["Giá tiền"].apply(format_money)

            st.dataframe(final_display_df, use_container_width=True, hide_index=True)

            final_total = final_df["Giá tiền"].sum()

            st.markdown(
                f"""
                <div class="total-box">
                    <div class="total-text">
                        Tổng tiền: {format_money(final_total)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success("Bill đã được nhân viên xác nhận.")

    else:
        st.info("Anh hãy tải ảnh hoặc chụp ảnh khay cơm để bắt đầu.")


# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Bảng giá món ăn")

    price_table = pd.DataFrame([
        ["Cơm trắng", format_money(10000)],
        ["Đậu hũ sốt cà", format_money(25000)],
        ["Cá hú kho", format_money(30000)],
        ["Thịt kho trứng", format_money(30000)],
        ["Thịt kho", format_money(25000)],
        ["Canh chua", format_money(10000)],
        ["Sườn nướng", format_money(30000)],
        ["Canh rau", format_money(7000)],
        ["Rau xào", format_money(10000)],
        ["Trứng chiên", format_money(25000)]
    ], columns=["Món ăn", "Giá tiền"])

    st.dataframe(price_table, use_container_width=True, hide_index=True)
=======
# HÀM XỬ LÝ
# =========================
def format_money(value):
    return f"{value:,.0f}".replace(",", ".") + " đ"


@st.cache_resource
def load_food_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Không tìm thấy file model: {MODEL_PATH}")
        st.stop()

    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
    except Exception:
        import keras
        model = keras.models.load_model(MODEL_PATH, compile=False)

    return model


def get_model_input_size(model):
    try:
        shape = model.input_shape

        if isinstance(shape, list):
            shape = shape[0]

        h = shape[1]
        w = shape[2]

        if h is None or w is None:
            return (224, 224)

        return (int(w), int(h))
    except Exception:
        return (224, 224)


def crop_by_ratio(image, ratio_box, margin=0.025):
    w, h = image.size
    x1, y1, x2, y2 = ratio_box

    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    box_w = x2 - x1
    box_h = y2 - y1

    x1 = x1 + int(box_w * margin)
    y1 = y1 + int(box_h * margin)
    x2 = x2 - int(box_w * margin)
    y2 = y2 - int(box_h * margin)

    return image.crop((x1, y1, x2, y2))


def draw_boxes(image):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    w, h = preview.size

    for name, ratio_box in ROI_RATIOS.items():
        x1, y1, x2, y2 = ratio_box

        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        color = BOX_COLORS.get(name, "red")

        draw.rectangle((x1, y1, x2, y2), outline=color, width=6)
        draw.text((x1 + 12, y1 + 12), name, fill=color)

    return preview


def predict_food(model, crop_img, input_size):
    img = crop_img.convert("RGB")
    img = img.resize(input_size)

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])

    class_key = CLASS_NAMES[class_index]
    food_name = DISPLAY_NAMES[class_key]
    price = PRICE_TABLE[class_key]

    return class_key, food_name, price, confidence


# =========================
# GIAO DIỆN APP
# =========================
st.markdown('<div class="title">🍱 AD Food Tray Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Nhận diện khay đồ ăn 5 ô và tự động tính tổng tiền</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note-box">
    <b>Lưu ý:</b> App đang cắt theo khay mẫu cố định anh gửi. 
    Khi chụp hoặc tải ảnh lên, anh nên để ảnh ngang, thấy rõ toàn bộ khay, hạn chế nghiêng quá nhiều.
    <br><br>
    <b>Canh chua có cá và không cá đã được gộp thành:</b> Canh chua = 10.000 đ.
    Hiện tại chưa có chức năng đếm số trứng.
    </div>
    """,
    unsafe_allow_html=True
)

model = load_food_model()
input_size = get_model_input_size(model)

st.success(f"Đã tải model thành công. Kích thước ảnh model dùng: {input_size[0]}x{input_size[1]}")

left, right = st.columns(2)

with left:
    st.subheader("📤 Tải ảnh khay đồ ăn")
    uploaded_file = st.file_uploader(
        "Chọn ảnh khay",
        type=["jpg", "jpeg", "png"]
    )

with right:
    st.subheader("📷 Chụp trực tiếp")
    camera_file = st.camera_input("Chụp ảnh khay đồ ăn")

image_file = uploaded_file if uploaded_file is not None else camera_file

if image_file is None:
    st.info("Anh hãy tải ảnh khay đồ ăn hoặc chụp trực tiếp để bắt đầu nhận diện.")
    st.stop()

image = Image.open(image_file).convert("RGB")

st.write("---")
st.subheader("1. Ảnh gốc và vùng cắt 5 ô")

col_img1, col_img2 = st.columns(2)

with col_img1:
    st.image(image, caption="Ảnh gốc", use_container_width=True)

with col_img2:
    boxed_image = draw_boxes(image)
    st.image(boxed_image, caption="Vùng app sẽ cắt để nhận diện", use_container_width=True)

st.write("---")
st.subheader("2. Ảnh sau khi cắt từng ô")

results = []
crop_data = []

for position, ratio_box in ROI_RATIOS.items():
    crop_img = crop_by_ratio(image, ratio_box)
    class_key, food_name, price, confidence = predict_food(model, crop_img, input_size)

    results.append({
        "Vị trí": position,
        "Class": class_key,
        "Món nhận diện": food_name,
        "Độ tin cậy": confidence,
        "Giá tiền": price
    })

    crop_data.append({
        "position": position,
        "crop_img": crop_img,
        "class_key": class_key,
        "food_name": food_name,
        "price": price,
        "confidence": confidence
    })

crop_cols = st.columns(5)

for i, item in enumerate(crop_data):
    with crop_cols[i]:
        st.image(
            item["crop_img"],
            caption=f'{item["position"]}\n{item["food_name"]} - {item["confidence"] * 100:.1f}%',
            use_container_width=True
        )

st.write("---")
st.subheader("3. Kết quả nhận diện và tính tiền")

df = pd.DataFrame(results)

df_show = df.copy()
df_show["Độ tin cậy"] = df_show["Độ tin cậy"].apply(lambda x: f"{x * 100:.2f}%")
df_show["Giá tiền"] = df_show["Giá tiền"].apply(format_money)

st.dataframe(df_show, use_container_width=True, hide_index=True)

total_price = int(df["Giá tiền"].sum())

st.markdown(
    f"""
    <div class="total-box">
        <div class="total-title">Tổng tiền khay đồ ăn</div>
        <div class="total-money">{format_money(total_price)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")
st.subheader("4. Chỉnh lại món nếu model nhận sai")

st.caption("Phần này giúp anh sửa nhanh món bị nhận diện sai, tổng tiền sẽ tính lại theo món anh chọn.")

manual_total = 0
manual_rows = []

food_options = list(DISPLAY_NAMES.keys())

for idx, item in enumerate(crop_data):
    col_a, col_b, col_c = st.columns([1.2, 2, 1])

    with col_a:
        st.image(item["crop_img"], caption=item["position"], use_container_width=True)

    with col_b:
        default_index = food_options.index(item["class_key"])

        selected_key = st.selectbox(
            f"Chọn món cho {item['position']}",
            options=food_options,
            index=default_index,
            format_func=lambda key: DISPLAY_NAMES[key],
            key=f"select_food_{idx}"
        )

    with col_c:
        selected_price = PRICE_TABLE[selected_key]
        st.metric("Giá", format_money(selected_price))

    manual_total += selected_price

    manual_rows.append({
        "Vị trí": item["position"],
        "Món sau chỉnh": DISPLAY_NAMES[selected_key],
        "Giá tiền": selected_price
    })

manual_df = pd.DataFrame(manual_rows)
manual_df_show = manual_df.copy()
manual_df_show["Giá tiền"] = manual_df_show["Giá tiền"].apply(format_money)

st.write("")
st.dataframe(manual_df_show, use_container_width=True, hide_index=True)

st.markdown(
    f"""
    <div class="total-box">
        <div class="total-title">Tổng tiền sau khi chỉnh</div>
        <div class="total-money">{format_money(manual_total)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

low_conf_df = df[df["Độ tin cậy"] < 0.5]

if len(low_conf_df) > 0:
    st.warning("Có món có độ tin cậy dưới 50%. Anh nên kiểm tra lại vùng cắt hoặc chỉnh món thủ công ở phần bên dưới.")
>>>>>>> da9f624 (first commit)
