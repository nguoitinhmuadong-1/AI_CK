import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AD Food Tray",
    page_icon="🍱",
    layout="wide"
)

MODEL_PATH = "food_model_final.h5"

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

# Cắt trực tiếp từ ảnh khay
ROI_RATIOS = {
    "Ô trên trái":  (0.04, 0.08, 0.30, 0.47),
    "Ô trên giữa": (0.31, 0.07, 0.57, 0.47),
    "Ô trên phải": (0.62, 0.04, 0.98, 0.48),
    "Ô dưới trái": (0.04, 0.50, 0.42, 0.98),
    "Ô dưới phải": (0.45, 0.50, 0.98, 0.97),
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
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff7e6, #ffe0b2, #fff3e0);
    }

    .title {
        text-align: center;
        font-size: 44px;
        font-weight: 900;
        color: #4e2600;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 19px;
        color: #5c3300;
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
        color: #4e2600;
    }

    .total-money {
        font-size: 46px;
        font-weight: 900;
        color: #d35400;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #4e2600 !important;
    }

    .stImage figcaption {
        color: #4e2600 !important;
        font-weight: 700 !important;
    }

    div.stButton > button {
        background-color: #ff9800;
        color: white !important;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        padding: 10px 22px;
    }

    div.stButton > button:hover {
        background-color: #e67e00;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# FUNCTIONS
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

        height = shape[1]
        width = shape[2]

        if height is None or width is None:
            return (224, 224)

        return (int(width), int(height))
    except Exception:
        return (224, 224)


def crop_by_ratio(image, ratio_box, margin=0.01):
    w, h = image.size
    x1, y1, x2, y2 = ratio_box

    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    box_w = x2 - x1
    box_h = y2 - y1

    x1 += int(box_w * margin)
    y1 += int(box_h * margin)
    x2 -= int(box_w * margin)
    y2 -= int(box_h * margin)

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


def get_color_features(crop_img):
    """
    Phân tích màu chính trong ảnh crop.
    """
    img = crop_img.convert("RGB").resize((224, 224))
    rgb = np.array(img).astype("uint8")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    total = h.shape[0] * h.shape[1]

    # cơm trắng / vùng sáng
    white_mask = (s < 45) & (v > 160)

    # xanh lá
    green_mask = (h >= 35) & (h <= 95) & (s > 45) & (v > 50)

    # vàng
    yellow_mask = (h >= 18) & (h <= 38) & (s > 60) & (v > 80)

    # nâu / cam
    brown_orange_mask = (
        ((h >= 5) & (h <= 25) & (s > 55) & (v > 45)) |
        ((h >= 0) & (h <= 10) & (s > 60) & (v > 40))
    )

    # đỏ / cam rõ
    red_orange_mask = (
        ((h >= 0) & (h <= 15) & (s > 60) & (v > 80)) |
        ((h >= 170) & (h <= 179) & (s > 60) & (v > 80))
    )

    return {
        "white_ratio": float(np.sum(white_mask) / total),
        "green_ratio": float(np.sum(green_mask) / total),
        "yellow_ratio": float(np.sum(yellow_mask) / total),
        "brown_orange_ratio": float(np.sum(brown_orange_mask) / total),
        "red_orange_ratio": float(np.sum(red_orange_mask) / total),
    }


def smart_correct_by_color(raw_class_key, confidence, top3, crop_img):
    """
    Tự sửa theo màu khi model không chắc.
    Chỉ sửa trong phạm vi Top 3.
    """
    features = get_color_features(crop_img)
    top3_keys = [item["class_key"] for item in top3]

    corrected_key = raw_class_key
    note = ""

    # nếu model rất chắc thì giữ nguyên
    if confidence >= 0.80:
        return corrected_key, note, features

    white_ratio = features["white_ratio"]
    green_ratio = features["green_ratio"]
    yellow_ratio = features["yellow_ratio"]
    brown_ratio = features["brown_orange_ratio"]
    red_orange_ratio = features["red_orange_ratio"]

    # Cơm trắng
    if white_ratio > 0.45 and "com_trang" in top3_keys:
        corrected_key = "com_trang"
        note = "Tự sửa theo màu: ảnh có nhiều vùng trắng nên chọn Cơm trắng"

    # Canh rau
    elif green_ratio > 0.18 and "canh_rau" in top3_keys and raw_class_key in ["canh_chua", "rau_xao", "dau_hu_sot_ca"]:
        corrected_key = "canh_rau"
        note = "Tự sửa theo màu: ảnh nhiều xanh nên chọn Canh rau"

    # Rau xào
    elif green_ratio > 0.22 and "rau_xao" in top3_keys and raw_class_key in ["canh_rau", "canh_chua"]:
        corrected_key = "rau_xao"
        note = "Tự sửa theo màu: ảnh nhiều xanh đậm nên chọn Rau xào"

    # Trứng chiên
    elif yellow_ratio > 0.18 and brown_ratio < 0.10 and "trung_chien" in top3_keys:
        corrected_key = "trung_chien"
        note = "Tự sửa theo màu: ảnh nhiều vàng nên chọn Trứng chiên"

    # Thịt kho
    elif brown_ratio > 0.13 and "thit_kho" in top3_keys:
        corrected_key = "thit_kho"
        note = "Tự sửa theo màu: ảnh có nhiều nâu/cam nên chọn Thịt kho"

    # Thịt kho trứng
    elif brown_ratio > 0.10 and yellow_ratio > 0.08 and "thit_kho_trung" in top3_keys:
        corrected_key = "thit_kho_trung"
        note = "Tự sửa theo màu: ảnh có nâu/cam và vàng nên chọn Thịt kho trứng"

    # Canh chua
    elif red_orange_ratio > 0.10 and green_ratio < 0.15 and "canh_chua" in top3_keys:
        corrected_key = "canh_chua"
        note = "Tự sửa theo màu: ảnh có đỏ/cam nên chọn Canh chua"

    return corrected_key, note, features


def predict_one_image(model, image, input_size, use_color_correction=True):
    img = image.convert("RGB")
    img = img.resize(input_size)

    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    # giống Colab
    arr = preprocess_input(arr)

    prediction = model.predict(arr, verbose=0)[0]

    pred_index = int(np.argmax(prediction))
    confidence = float(prediction[pred_index])
    raw_class_key = CLASS_NAMES[pred_index]

    top_indices = np.argsort(prediction)[::-1][:3]

    top3 = []
    for idx in top_indices:
        idx = int(idx)
        key = CLASS_NAMES[idx]
        top3.append({
            "class_key": key,
            "food_name": DISPLAY_NAMES[key],
            "confidence": float(prediction[idx])
        })

    correction_note = ""
    features = {}

    if use_color_correction:
        class_key, correction_note, features = smart_correct_by_color(
            raw_class_key,
            confidence,
            top3,
            image
        )
    else:
        class_key = raw_class_key

    food_name = DISPLAY_NAMES[class_key]
    price = PRICE_TABLE[class_key]

    return raw_class_key, class_key, food_name, price, confidence, top3, correction_note, features


def top3_to_text(top3):
    return " | ".join([
        f"{item['food_name']} {item['confidence'] * 100:.1f}%"
        for item in top3
    ])


# =========================
# SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = {}

if "manual_results" not in st.session_state:
    st.session_state.manual_results = {}


# =========================
# APP
# =========================
st.markdown('<div class="title">🍱 AD Food Tray Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Cắt 5 ô trong khay, nhận diện từng ô và tự sửa theo màu</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note-box">
    <b>Cách dùng:</b> Anh upload hoặc chụp ảnh khay. App sẽ cắt ra 5 ô trước.
    Sau đó anh chọn từng ô và bấm <b>Nhận diện ô này</b>.
    <br><br>
    <b>Tự sửa theo màu:</b> chỉ hỗ trợ khi model chưa chắc chắn.
    <br>
    <b>Canh chua:</b> tính chung 10.000 đ. Hiện tại chưa đếm số trứng.
    </div>
    """,
    unsafe_allow_html=True
)

model = load_food_model()
input_size = get_model_input_size(model)

if len(CLASS_NAMES) != model.output_shape[-1]:
    st.error(
        f"Số class trong app là {len(CLASS_NAMES)}, nhưng model output là {model.output_shape[-1]}."
    )
    st.stop()

st.success(f"Đã tải model thành công. Input model: {input_size[0]}x{input_size[1]}")

use_color_correction = st.checkbox(
    "Bật tự sửa theo màu khi model nhận diện chưa chắc",
    value=True
)

left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader(
        "📤 Tải ảnh khay đồ ăn",
        type=["jpg", "jpeg", "png"]
    )

with right:
    camera_file = st.camera_input("📷 Chụp ảnh khay đồ ăn")

image_file = uploaded_file if uploaded_file is not None else camera_file

if image_file is None:
    st.info("Anh hãy tải ảnh khay đồ ăn hoặc chụp trực tiếp để bắt đầu.")
    st.stop()

image = Image.open(image_file).convert("RGB")

# reset nếu đổi ảnh
image_id = image_file.name if hasattr(image_file, "name") else "camera_image"

if "current_image_id" not in st.session_state:
    st.session_state.current_image_id = image_id

if st.session_state.current_image_id != image_id:
    st.session_state.current_image_id = image_id
    st.session_state.results = {}
    st.session_state.manual_results = {}

# =========================
# CẮT 5 Ô
# =========================
crops = {}

for position, box in ROI_RATIOS.items():
    crops[position] = crop_by_ratio(image, box)

st.write("---")
st.subheader("1. Ảnh khay và vùng cắt")

col1, col2 = st.columns(2)

with col1:
    st.image(image, caption="Ảnh gốc", use_container_width=True)

with col2:
    boxed_image = draw_boxes(image)
    st.image(boxed_image, caption="5 ô được cắt trong khay", use_container_width=True)

st.write("---")
st.subheader("2. 5 ảnh đã cắt từ khay")

crop_cols = st.columns(5)

for i, position in enumerate(ROI_RATIOS.keys()):
    with crop_cols[i]:
        st.image(crops[position], caption=position, use_container_width=True)

# =========================
# NHẬN DIỆN TỪNG Ô
# =========================
st.write("---")
st.subheader("3. Chọn từng ô để nhận diện")

selected_position = st.selectbox(
    "Chọn ô cần nhận diện",
    options=list(ROI_RATIOS.keys())
)

col_a, col_b = st.columns([1, 1.4])

with col_a:
    st.image(crops[selected_position], caption=f"Ảnh đang chọn: {selected_position}", use_container_width=True)

with col_b:
    if st.button("🔍 Nhận diện ô này"):
        raw_class_key, class_key, food_name, price, confidence, top3, correction_note, features = predict_one_image(
            model,
            crops[selected_position],
            input_size,
            use_color_correction=use_color_correction
        )

        st.session_state.results[selected_position] = {
            "position": selected_position,
            "raw_class_key": raw_class_key,
            "class_key": class_key,
            "food_name": food_name,
            "price": price,
            "confidence": confidence,
            "top3": top3,
            "correction_note": correction_note,
            "features": features
        }

        st.session_state.manual_results[selected_position] = class_key

    if selected_position in st.session_state.results:
        item = st.session_state.results[selected_position]

        st.success(f"Món dự đoán: {item['food_name']}")
        st.write(f"Độ tin cậy model gốc: **{item['confidence'] * 100:.2f}%**")
        st.write(f"Top 3: {top3_to_text(item['top3'])}")
        st.write(f"Giá: **{format_money(item['price'])}**")

        if item.get("correction_note"):
            st.warning(item["correction_note"])
            st.caption(f"Model gốc đoán: {DISPLAY_NAMES[item['raw_class_key']]}")

        if item.get("features"):
            f = item["features"]
            st.caption(
                f"Màu ảnh: trắng={f.get('white_ratio',0):.2f} | "
                f"xanh={f.get('green_ratio',0):.2f} | "
                f"vàng={f.get('yellow_ratio',0):.2f} | "
                f"nâu/cam={f.get('brown_orange_ratio',0):.2f} | "
                f"đỏ/cam={f.get('red_orange_ratio',0):.2f}"
            )
    else:
        st.info("Anh bấm nút nhận diện để dự đoán ô này.")

# =========================
# KẾT QUẢ ĐÃ NHẬN DIỆN
# =========================
st.write("---")
st.subheader("4. Các ô đã nhận diện")

if len(st.session_state.results) == 0:
    st.warning("Chưa có ô nào được nhận diện.")
else:
    rows = []

    for position, item in st.session_state.results.items():
        rows.append({
            "Vị trí": position,
            "Model gốc": DISPLAY_NAMES[item["raw_class_key"]],
            "Món sau sửa": item["food_name"],
            "Độ tin cậy": item["confidence"],
            "Top 3": top3_to_text(item["top3"]),
            "Giá tiền": item["price"]
        })

    df = pd.DataFrame(rows)

    df_show = df.copy()
    df_show["Độ tin cậy"] = df_show["Độ tin cậy"].apply(lambda x: f"{x * 100:.2f}%")
    df_show["Giá tiền"] = df_show["Giá tiền"].apply(format_money)

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    total_model = int(df["Giá tiền"].sum())

    st.markdown(
        f"""
        <div class="total-box">
            <div class="total-title">Tổng tiền các ô đã nhận diện</div>
            <div class="total-money">{format_money(total_model)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# CHỈNH MÓN THỦ CÔNG
# =========================
if len(st.session_state.results) > 0:
    st.write("---")
    st.subheader("5. Chỉnh món nếu model vẫn sai")

    food_options = list(DISPLAY_NAMES.keys())
    manual_rows = []
    manual_total = 0

    for position, item in st.session_state.results.items():
        col_img, col_select, col_price = st.columns([1.2, 2, 1])

        with col_img:
            st.image(crops[position], caption=position, use_container_width=True)

        with col_select:
            default_key = st.session_state.manual_results.get(position, item["class_key"])
            default_index = food_options.index(default_key)

            selected_key = st.selectbox(
                f"Chọn món đúng cho {position}",
                options=food_options,
                index=default_index,
                format_func=lambda key: DISPLAY_NAMES[key],
                key=f"manual_{position}"
            )

            st.session_state.manual_results[position] = selected_key
            st.caption("Top 3: " + top3_to_text(item["top3"]))

        with col_price:
            selected_price = PRICE_TABLE[selected_key]
            st.metric("Giá", format_money(selected_price))

        manual_total += selected_price

        manual_rows.append({
            "Vị trí": position,
            "Món sau chỉnh": DISPLAY_NAMES[selected_key],
            "Giá tiền": selected_price
        })

    manual_df = pd.DataFrame(manual_rows)
    manual_df_show = manual_df.copy()
    manual_df_show["Giá tiền"] = manual_df_show["Giá tiền"].apply(format_money)

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

# =========================
# RESET
# =========================
st.write("---")

if st.button("🗑️ Xóa kết quả nhận diện"):
    st.session_state.results = {}
    st.session_state.manual_results = {}
    st.rerun()