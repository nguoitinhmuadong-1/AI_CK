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
    page_icon="🍱",
    layout="wide"
)

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
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
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