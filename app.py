import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import cv2

# =========================
# CẤU HÌNH APP
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

# Sau khi đã tự tìm và cắt riêng cái khay,
# 5 ô này sẽ được cắt theo tỉ lệ trên ảnh khay đã chuẩn hóa
ROI_RATIOS = {
    "Ô trên trái":  (0.02, 0.05, 0.30, 0.49),
    "Ô trên giữa": (0.30, 0.05, 0.58, 0.49),
    "Ô trên phải": (0.58, 0.04, 0.98, 0.50),
    "Ô dưới trái": (0.02, 0.50, 0.43, 0.98),
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

    .title {
        text-align: center;
        font-size: 46px;
        font-weight: 900;
        color: #4e2600;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
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
        font-size: 48px;
        font-weight: 900;
        color: #d35400;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #4e2600 !important;
    }

    .stImage figcaption {
        color: #4e2600 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    div.stButton > button {
        background-color: #ff9800;
        color: white !important;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        padding: 10px 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HÀM TIỆN ÍCH
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


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 10 or max_height < 10:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def smart_find_tray(image_pil):
    """
    Tự tìm khay trong ảnh.
    Nếu tìm được khay: cắt riêng khay ra.
    Nếu không tìm được: dùng ảnh gốc làm fallback.
    """

    image_rgb = np.array(image_pil.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    # Resize nhỏ lại để xử lý nhanh
    max_side = 1200
    scale = 1.0

    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale))
        )

    small_h, small_w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Bắt cạnh khay
    edges = cv2.Canny(gray, 40, 120)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image_pil, "Không tìm được contour khay, dùng ảnh gốc."

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_box = None
    image_area = small_w * small_h

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)

        # Bỏ contour quá nhỏ
        if area < image_area * 0.15:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="float32")

        bw = rect[1][0]
        bh = rect[1][1]

        if bw < small_w * 0.3 or bh < small_h * 0.3:
            continue

        best_box = box
        break

    if best_box is None:
        # Fallback: lấy bounding box contour lớn nhất
        cnt = contours[0]
        x, y, ww, hh = cv2.boundingRect(cnt)

        x = int(x / scale)
        y = int(y / scale)
        ww = int(ww / scale)
        hh = int(hh / scale)

        cropped = original[y:y+hh, x:x+ww]

        if cropped.size == 0:
            return image_pil, "Không cắt được khay, dùng ảnh gốc."

        if cropped.shape[0] > cropped.shape[1]:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb), "Fallback: cắt theo bounding box."

    # Scale box về ảnh gốc
    best_box = best_box / scale

    warped = four_point_transform(original, best_box)

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    tray_pil = Image.fromarray(warped_rgb)

    return tray_pil, "Đã tự tìm và chuẩn hóa khay."


def crop_by_ratio(image, ratio_box, margin=0.02):
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


def predict_food(model, crop_img, input_size):
    img = crop_img.convert("RGB")
    img = img.resize(input_size)

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:3]

    top3 = []
    for idx in top_indices:
        idx = int(idx)
        class_key = CLASS_NAMES[idx]

        top3.append({
            "class_key": class_key,
            "food_name": DISPLAY_NAMES[class_key],
            "confidence": float(preds[idx])
        })

    best_idx = int(top_indices[0])
    class_key = CLASS_NAMES[best_idx]
    food_name = DISPLAY_NAMES[class_key]
    price = PRICE_TABLE[class_key]
    confidence = float(preds[best_idx])

    return class_key, food_name, price, confidence, top3


# =========================
# GIAO DIỆN APP
# =========================
st.markdown('<div class="title">🍱 AD Food Tray Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Tự cắt khay thông minh, nhận diện món ăn và tính tiền</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note-box">
    <b>Cách cắt mới:</b> App sẽ tự tìm cái khay trong ảnh trước, sau đó mới cắt 5 ô đồ ăn.
    Vì vậy ảnh có bị xa, lệch nhẹ hoặc dư nền bàn thì vẫn ổn hơn kiểu cắt cố định trên ảnh gốc.
    <br><br>
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
    st.info("Anh hãy tải ảnh khay đồ ăn hoặc chụp trực tiếp để bắt đầu.")
    st.stop()

original_image = Image.open(image_file).convert("RGB")

# =========================
# CẮT KHAY THÔNG MINH
# =========================
tray_image, tray_status = smart_find_tray(original_image)
boxed_tray = draw_boxes(tray_image)

st.write("---")
st.subheader("1. Ảnh gốc và khay sau khi tự cắt")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(original_image, caption="Ảnh gốc", use_container_width=True)

with col2:
    st.image(tray_image, caption=f"Khay đã tự cắt - {tray_status}", use_container_width=True)

with col3:
    st.image(boxed_tray, caption="5 ô sẽ được cắt từ khay", use_container_width=True)

# =========================
# CẮT 5 Ô + DỰ ĐOÁN
# =========================
st.write("---")
st.subheader("2. Ảnh sau khi cắt từng ô")

results = []
crop_data = []

for position, ratio_box in ROI_RATIOS.items():
    crop_img = crop_by_ratio(tray_image, ratio_box)
    class_key, food_name, price, confidence, top3 = predict_food(model, crop_img, input_size)

    top3_text = " | ".join(
        [f"{item['food_name']} {item['confidence'] * 100:.1f}%" for item in top3]
    )

    results.append({
        "Vị trí": position,
        "Class": class_key,
        "Món nhận diện": food_name,
        "Top 3 dự đoán": top3_text,
        "Độ tin cậy": confidence,
        "Giá tiền": price
    })

    crop_data.append({
        "position": position,
        "crop_img": crop_img,
        "class_key": class_key,
        "food_name": food_name,
        "price": price,
        "confidence": confidence,
        "top3": top3
    })

crop_cols = st.columns(5)

for i, item in enumerate(crop_data):
    with crop_cols[i]:
        st.image(
            item["crop_img"],
            caption=f'{item["position"]}\n{item["food_name"]} - {item["confidence"] * 100:.1f}%',
            use_container_width=True
        )

# =========================
# BẢNG KẾT QUẢ
# =========================
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
        <div class="total-title">Tổng tiền theo model</div>
        <div class="total-money">{format_money(total_price)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# CHỈNH THỦ CÔNG NẾU MODEL NHẬN SAI
# =========================
st.write("---")
st.subheader("4. Chỉnh món nếu model nhận sai")

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

        top3_view = " | ".join(
            [f"{x['food_name']} {x['confidence'] * 100:.1f}%" for x in item["top3"]]
        )
        st.caption(f"Top 3: {top3_view}")

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