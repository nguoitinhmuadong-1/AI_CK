import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import cv2
import hashlib
from io import BytesIO

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AD Food Cashier",
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

# Tọa độ cắt 5 ô sau khi ảnh đã được crop sát khay
ROI_RATIOS = {
    "Ô trên trái":  (0.03, 0.07, 0.31, 0.49),
    "Ô trên giữa": (0.31, 0.07, 0.59, 0.49),
    "Ô trên phải": (0.59, 0.07, 0.98, 0.49),
    "Ô dưới trái": (0.03, 0.50, 0.43, 0.98),
    "Ô dưới phải": (0.43, 0.50, 0.98, 0.98),
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

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 900;
        color: #4e2600;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #6b3b00;
        margin-bottom: 25px;
    }

    .note-box {
        background-color: #fff8e1;
        color: #4e2600;
        border-left: 8px solid #ff9800;
        padding: 18px;
        border-radius: 16px;
        font-size: 17px;
        margin-bottom: 20px;
    }

    .cashier-panel {
        background-color: #ffffff;
        border-radius: 24px;
        padding: 22px;
        border: 3px solid #ff9800;
        box-shadow: 0px 6px 22px rgba(0,0,0,0.15);
    }

    .bill-title {
        text-align: center;
        font-size: 30px;
        font-weight: 900;
        color: #4e2600;
        margin-bottom: 12px;
    }

    .total-box {
        background: linear-gradient(135deg, #fff3e0, #ffffff);
        border: 3px solid #ff9800;
        border-radius: 22px;
        padding: 25px;
        text-align: center;
        box-shadow: 0px 5px 18px rgba(0,0,0,0.12);
        margin-top: 15px;
    }

    .total-title {
        font-size: 25px;
        font-weight: 800;
        color: #4e2600;
    }

    .total-money {
        font-size: 52px;
        font-weight: 900;
        color: #d35400;
    }

    .food-card {
        background-color: white;
        border-radius: 18px;
        padding: 14px;
        border: 2px solid #ffd18a;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.10);
        margin-bottom: 12px;
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
        border-radius: 14px;
        border: none;
        font-weight: 800;
        padding: 12px 24px;
        font-size: 17px;
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


def get_output_units(model):
    try:
        shape = model.output_shape

        if isinstance(shape, list):
            shape = shape[0]

        return int(shape[-1])

    except Exception:
        return len(CLASS_NAMES)


def auto_crop_tray(image_pil):
    """
    Tự tìm vùng khay và cắt sát khay.
    Không warp, không làm méo ảnh.
    Nếu không tìm được khay thì dùng ảnh gốc.
    """
    image_rgb = np.array(image_pil.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    scale = 1.0
    max_side = 1000

    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale))
        )

    small_h, small_w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(gray, 40, 130)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image_pil, "Không tìm được khay, dùng ảnh gốc"

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = small_w * small_h
    best_box = None

    for cnt in contours[:20]:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        ratio = bw / max(bh, 1)

        if area < image_area * 0.20:
            continue

        if area > image_area * 0.98:
            continue

        if ratio < 1.05 or ratio > 2.8:
            continue

        if bw < small_w * 0.40 or bh < small_h * 0.30:
            continue

        best_box = (x, y, bw, bh)
        break

    if best_box is None:
        return image_pil, "Không cắt được khay, dùng ảnh gốc"

    x, y, bw, bh = best_box

    x = int(x / scale)
    y = int(y / scale)
    bw = int(bw / scale)
    bh = int(bh / scale)

    pad_x = int(bw * 0.025)
    pad_y = int(bh * 0.025)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = original[y1:y2, x1:x2]

    if cropped.size == 0:
        return image_pil, "Crop lỗi, dùng ảnh gốc"

    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    return Image.fromarray(cropped_rgb), "Đã tự cắt sát khay"


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


def predict_one_image(model, image, input_size):
    img = image.convert("RGB")
    img = img.resize(input_size)

    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    arr = preprocess_input(arr)

    prediction = model.predict(arr, verbose=0)[0]

    pred_index = int(np.argmax(prediction))
    confidence = float(prediction[pred_index])

    class_key = CLASS_NAMES[pred_index]
    food_name = DISPLAY_NAMES[class_key]
    price = PRICE_TABLE[class_key]

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

    return class_key, food_name, price, confidence, top3


def top3_to_text(top3):
    return " | ".join([
        f"{item['food_name']} {item['confidence'] * 100:.1f}%"
        for item in top3
    ])


def run_prediction_for_position(position, crop_img, model, input_size):
    class_key, food_name, price, confidence, top3 = predict_one_image(
        model,
        crop_img,
        input_size
    )

    st.session_state.results[position] = {
        "position": position,
        "class_key": class_key,
        "food_name": food_name,
        "price": price,
        "confidence": confidence,
        "top3": top3
    }

    st.session_state.manual_results[position] = class_key


# =========================
# SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = {}

if "manual_results" not in st.session_state:
    st.session_state.manual_results = {}

if "current_image_hash" not in st.session_state:
    st.session_state.current_image_hash = None


# =========================
# APP
# =========================
st.markdown('<div class="main-title">🍱 AD Food Cashier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hệ thống nhận diện món ăn trong khay và tự động tính tiền</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note-box">
    <b>Quy trình:</b> Upload hoặc chụp ảnh khay → app tự cắt sát khay → cắt 5 ô món ăn → nhận diện → lập hóa đơn.
    <br>
    <b>Lưu ý:</b> Nếu model nhận sai, anh chỉnh lại món trong hóa đơn, tổng tiền sẽ tự cập nhật.
    <br>
    <b>Canh chua:</b> tính chung 10.000 đ. Hiện tại chưa đếm số trứng.
    </div>
    """,
    unsafe_allow_html=True
)

model = load_food_model()
input_size = get_model_input_size(model)
output_units = get_output_units(model)

if len(CLASS_NAMES) != output_units:
    st.error(
        f"Số class trong app là {len(CLASS_NAMES)}, nhưng model output là {output_units}."
    )
    st.stop()

left_upload, right_upload = st.columns(2)

with left_upload:
    uploaded_file = st.file_uploader(
        "📤 Tải ảnh khay đồ ăn",
        type=["jpg", "jpeg", "png"]
    )

with right_upload:
    camera_file = st.camera_input("📷 Chụp ảnh khay đồ ăn")

image_file = uploaded_file if uploaded_file is not None else camera_file

if image_file is None:
    st.info("Anh hãy tải ảnh khay đồ ăn hoặc chụp trực tiếp để bắt đầu.")
    st.stop()

image_bytes = image_file.getvalue()
image_hash = hashlib.md5(image_bytes).hexdigest()

if st.session_state.current_image_hash != image_hash:
    st.session_state.current_image_hash = image_hash
    st.session_state.results = {}
    st.session_state.manual_results = {}

original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

tray_image, tray_status = auto_crop_tray(original_image)

crops = {}

for position, box in ROI_RATIOS.items():
    crops[position] = crop_by_ratio(tray_image, box)

# =========================
# MAIN LAYOUT
# =========================
st.write("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("📷 Ảnh khay xử lý")

    tab1, tab2, tab3 = st.tabs(["Ảnh gốc", "Khay đã cắt", "Vùng cắt 5 ô"])

    with tab1:
        st.image(original_image, caption="Ảnh gốc", use_container_width=True)

    with tab2:
        st.image(tray_image, caption=tray_status, use_container_width=True)

    with tab3:
        boxed_image = draw_boxes(tray_image)
        st.image(boxed_image, caption="5 vùng món ăn được cắt", use_container_width=True)

    st.write("")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🔍 Nhận diện & Tính tiền"):
            with st.spinner("Đang nhận diện món ăn trong khay..."):
                for position in ROI_RATIOS.keys():
                    run_prediction_for_position(
                        position,
                        crops[position],
                        model,
                        input_size
                    )

            st.success("Đã nhận diện xong và tạo hóa đơn.")

    with col_btn2:
        if st.button("🗑️ Làm mới kết quả"):
            st.session_state.results = {}
            st.session_state.manual_results = {}
            st.rerun()

with right:
    st.markdown('<div class="cashier-panel">', unsafe_allow_html=True)
    st.markdown('<div class="bill-title">🧾 HÓA ĐƠN</div>', unsafe_allow_html=True)

    if len(st.session_state.results) == 0:
        st.warning("Chưa có hóa đơn. Anh bấm nút **Nhận diện & Tính tiền**.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        food_options = list(DISPLAY_NAMES.keys())
        bill_rows = []
        final_total = 0

        for idx, position in enumerate(ROI_RATIOS.keys(), start=1):
            item = st.session_state.results[position]

            default_key = st.session_state.manual_results.get(
                position,
                item["class_key"]
            )

            default_index = food_options.index(default_key)

            st.markdown('<div class="food-card">', unsafe_allow_html=True)

            col_img, col_info = st.columns([0.9, 1.4])

            with col_img:
                st.image(crops[position], caption=f"Món {idx}", use_container_width=True)

            with col_info:
                selected_key = st.selectbox(
                    f"Món {idx} - {position}",
                    options=food_options,
                    index=default_index,
                    format_func=lambda key: DISPLAY_NAMES[key],
                    key=f"bill_select_{position}"
                )

                st.session_state.manual_results[position] = selected_key

                selected_price = PRICE_TABLE[selected_key]
                final_total += selected_price

                st.write(f"**Model đoán:** {item['food_name']}")
                st.write(f"**Độ tin cậy:** {item['confidence'] * 100:.2f}%")
                st.caption("Top 3: " + top3_to_text(item["top3"]))
                st.write(f"**Giá:** {format_money(selected_price)}")

                bill_rows.append({
                    "STT": idx,
                    "Vị trí": position,
                    "Món": DISPLAY_NAMES[selected_key],
                    "Giá tiền": selected_price
                })

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="total-box">
                <div class="total-title">Tổng tiền cần thanh toán</div>
                <div class="total-money">{format_money(final_total)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CROP PREVIEW + BILL TABLE
# =========================
st.write("---")
st.subheader("🍱 5 món đã cắt từ khay")

crop_cols = st.columns(5)

for i, position in enumerate(ROI_RATIOS.keys()):
    with crop_cols[i]:
        st.image(crops[position], caption=position, use_container_width=True)

if len(st.session_state.results) > 0:
    st.write("---")
    st.subheader("📋 Bảng chi tiết hóa đơn")

    table_rows = []
    total_table = 0

    for idx, position in enumerate(ROI_RATIOS.keys(), start=1):
        item = st.session_state.results[position]
        selected_key = st.session_state.manual_results.get(
            position,
            item["class_key"]
        )

        selected_price = PRICE_TABLE[selected_key]
        total_table += selected_price

        table_rows.append({
            "STT": idx,
            "Vị trí": position,
            "Món model đoán": item["food_name"],
            "Món sau chỉnh": DISPLAY_NAMES[selected_key],
            "Độ tin cậy": f"{item['confidence'] * 100:.2f}%",
            "Giá tiền": format_money(selected_price)
        })

    df_bill = pd.DataFrame(table_rows)
    st.dataframe(df_bill, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="total-box">
            <div class="total-title">Tổng tiền cuối cùng</div>
            <div class="total-money">{format_money(total_table)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )