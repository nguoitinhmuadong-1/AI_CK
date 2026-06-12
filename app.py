import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# CẤU HÌNH MODEL
# =========================
MODEL_PATH = "food_model_final.h5"
IMG_SIZE = 224

CLASS_NAMES = [
    "ca_hu_kho",
    "canh_chua",
    "canh_rau",
    "com_trang",
    "dau_hu_sot_ca",
    "rau_xao",
    "suon_nuong",
    "thit_kho",
    "thit_kho_trung",
    "trung_chien"
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
    "ca_hu_kho": 25000,
    "canh_chua": 10000,
    "canh_rau": 8000,
    "com_trang": 5000,
    "dau_hu_sot_ca": 12000,
    "rau_xao": 10000,
    "suon_nuong": 25000,
    "thit_kho": 20000,
    "thit_kho_trung": 25000,
    "trung_chien": 12000
}

# Model của anh train bằng preprocessing_function=preprocess_input
PREPROCESS_MODE = "mobilenetv2"


# =========================
# SETUP STREAMLIT
# =========================
st.set_page_config(
    page_title="AD VietFood Vision",
    page_icon="🍱",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 45%, #fee2e2 100%);
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        color: #9a3412;
        margin-bottom: 8px;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #7c2d12;
        margin-bottom: 30px;
    }

    .card {
        background: white;
        padding: 24px;
        border-radius: 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.10);
        margin-bottom: 18px;
        color: #1f2937;
    }

    .feature-card {
        background: white;
        border: 2px solid #fed7aa;
        padding: 24px;
        border-radius: 22px;
        text-align: center;
        height: 230px;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .feature-card h3 {
        color: #c2410c;
    }

    .result-box {
        background: white;
        border: 2px solid #fed7aa;
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 16px;
        color: #1f2937;
    }

    .total-box {
        background: linear-gradient(135deg, #f97316, #dc2626);
        color: white;
        padding: 25px;
        border-radius: 22px;
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        margin-top: 20px;
    }

    .small-note {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        padding: 16px;
        border-radius: 14px;
        color: #7c2d12;
        margin-top: 15px;
    }

    div.stButton > button {
        background-color: #f97316;
        color: white;
        border-radius: 14px;
        border: none;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 16px;
    }

    div.stButton > button:hover {
        background-color: #ea580c;
        color: white;
    }

    .stDataFrame {
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_food_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


try:
    model = load_food_model()
except Exception as e:
    st.error("Không load được model. Anh kiểm tra lại tên file model phải là: food_model_final.h5")
    st.exception(e)
    st.stop()


# =========================
# HÀM PHỤ
# =========================
def format_money(value):
    return f"{value:,.0f}đ".replace(",", ".")


def preprocess_crop(crop_rgb):
    img = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)

    if PREPROCESS_MODE == "mobilenetv2":
        img = preprocess_input(img)
    else:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img


def predict_food(crop_rgb):
    x = preprocess_crop(crop_rgb)
    pred = model.predict(x, verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])

    class_name = CLASS_NAMES[class_id]
    display_name = DISPLAY_NAMES[class_name]
    price = PRICE_TABLE[class_name]

    return class_name, display_name, confidence, price


def nms_boxes(boxes, overlap_thresh=0.35):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)

    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2]
    y2 = boxes_np[:, 1] + boxes_np[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)

    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]

        if len(idxs) == 0:
            break

        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs]
        idxs = idxs[overlap <= overlap_thresh]

    return boxes_np[pick].astype(int).tolist()


def detect_food_regions(image_rgb):
    """
    Tự tìm vùng món ăn bằng OpenCV.
    Bản này phù hợp để demo với model classification .h5.
    """
    h, w = image_rgb.shape[:2]

    max_width = 1000
    scale = 1.0

    if w > max_width:
        scale = max_width / w
        image_small = cv2.resize(image_rgb, (max_width, int(h * scale)))
    else:
        image_small = image_rgb.copy()

    hsv = cv2.cvtColor(image_small, cv2.COLOR_RGB2HSV)

    # Giữ vùng có màu món ăn, bỏ vùng quá trắng/sáng của khay
    lower = np.array([0, 20, 20])
    upper = np.array([179, 255, 245])
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = np.ones((25, 25), np.uint8)
    kernel_open = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = image_small.shape[0] * image_small.shape[1]

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < img_area * 0.008:
            continue

        if bw < 45 or bh < 45:
            continue

        ratio = bw / float(bh)
        if ratio < 0.35 or ratio > 3.5:
            continue

        boxes.append([x, y, bw, bh])

    boxes = nms_boxes(boxes)

    final_boxes = []

    for x, y, bw, bh in boxes:
        x = int(x / scale)
        y = int(y / scale)
        bw = int(bw / scale)
        bh = int(bh / scale)

        pad = 18
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        final_boxes.append([x1, y1, x2 - x1, y2 - y1])

    final_boxes = sorted(final_boxes, key=lambda b: (b[1], b[0]))
    return final_boxes


def draw_boxes(image_rgb, detections):
    image_draw = image_rgb.copy()

    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        label = det["display_name"]
        conf = det["confidence"]

        cv2.rectangle(image_draw, (x, y), (x + w, y + h), (255, 102, 0), 4)

        text = f"{i + 1}. {label} {conf * 100:.1f}%"
        cv2.putText(
            image_draw,
            text,
            (x, max(35, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 102, 0),
            2,
            cv2.LINE_AA
        )

    return image_draw


def recognize_image(image_rgb, mode):
    detections = []

    if mode == "Nhận diện 1 món":
        h, w = image_rgb.shape[:2]
        boxes = [[0, 0, w, h]]
    else:
        boxes = detect_food_regions(image_rgb)

        if len(boxes) == 0:
            h, w = image_rgb.shape[:2]
            boxes = [[0, 0, w, h]]

    for box in boxes:
        x, y, w, h = box
        crop = image_rgb[y:y + h, x:x + w]

        class_name, display_name, confidence, price = predict_food(crop)

        detections.append({
            "box": box,
            "crop": crop,
            "class_name": class_name,
            "display_name": display_name,
            "confidence": confidence,
            "price": price
        })

    return detections


# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "image_rgb" not in st.session_state:
    st.session_state.image_rgb = None

if "detections" not in st.session_state:
    st.session_state.detections = []


def go_page(page_name):
    st.session_state.page = page_name
    st.rerun()


# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":
    st.markdown('<div class="main-title">🍱 AD VietFood Vision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Hệ thống nhận diện và tính tiền khay cơm căn tin tự động</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📷 Thanh toán tại quầy</h3>
                <p>Chụp hoặc tải ảnh khay cơm, hệ thống tự nhận diện món ăn và tính tổng tiền.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Vào nhận diện", use_container_width=True):
            go_page("payment")

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📋 Thực đơn</h3>
                <p>Tra cứu danh sách món ăn, tên món và giá bán trong căn tin.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Xem thực đơn", use_container_width=True):
            go_page("menu")

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🛒 Đặt món từ xa</h3>
                <p>Chọn món, số lượng, áp mã giảm giá và tạo đơn hàng nhanh.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Đặt món", use_container_width=True):
            go_page("order")

    st.markdown("---")

    st.markdown(
        """
        <div class="card">
            <h3 style="color:#c2410c;">Giới thiệu hệ thống</h3>
            <p>
            Ứng dụng sử dụng Python, Streamlit, OpenCV và TensorFlow/Keras để xây dựng
            hệ thống nhận diện món ăn trên khay cơm. Sau khi nhận diện, hệ thống tự động
            tra cứu bảng giá và tính tổng hóa đơn cho người dùng.
            </p>
            <p>
            Phiên bản hiện tại hỗ trợ nhận diện 10 món ăn phổ biến trong căn tin.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# PHÂN HỆ 1: THANH TOÁN
# =========================
elif st.session_state.page == "payment":
    st.markdown('<div class="main-title">📷 Thanh toán tại quầy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Nhận diện món ăn trên khay và tính tổng tiền tự động</div>',
        unsafe_allow_html=True
    )

    col_back, col_clear = st.columns(2)

    with col_back:
        if st.button("⬅ Quay lại trang chủ"):
            st.session_state.image_rgb = None
            st.session_state.detections = []
            go_page("home")

    with col_clear:
        if st.button("🧹 Xóa kết quả"):
            st.session_state.image_rgb = None
            st.session_state.detections = []
            st.rerun()

    st.markdown("### Chế độ nhận diện")

    mode = st.radio(
        "Chọn chế độ:",
        ["Nhận diện khay nhiều món", "Nhận diện 1 món"],
        horizontal=True
    )

    st.markdown("### Chọn ảnh khay cơm")

    col_upload, col_camera = st.columns(2)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Tải ảnh lên",
            type=["jpg", "jpeg", "png"]
        )

    with col_camera:
        camera_file = st.camera_input("Chụp ảnh trực tiếp")

    input_file = uploaded_file if uploaded_file is not None else camera_file

    if input_file is not None:
        image = Image.open(input_file).convert("RGB")
        image_rgb = np.array(image)

        st.session_state.image_rgb = image_rgb

        st.markdown("### Ảnh đầu vào")
        st.image(image_rgb, use_container_width=True)

        if st.button("🔍 Nhận diện và tính tiền"):
            st.session_state.detections = recognize_image(image_rgb, mode)
            st.rerun()

    if st.session_state.image_rgb is not None and len(st.session_state.detections) > 0:
        st.markdown("### Ảnh sau khi nhận diện")

        image_draw = draw_boxes(st.session_state.image_rgb, st.session_state.detections)
        st.image(image_draw, use_container_width=True)

        st.markdown("### Hóa đơn món ăn")

        total = 0
        bill_rows = []

        for i, det in enumerate(st.session_state.detections):
            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            col_img, col_info = st.columns([1, 2])

            with col_img:
                st.image(det["crop"], caption=f"Món {i + 1}", use_container_width=True)

            with col_info:
                st.write(f"**Dự đoán:** {det['display_name']}")
                st.write(f"**Độ tin cậy:** {det['confidence'] * 100:.2f}%")

                current_index = CLASS_NAMES.index(det["class_name"])

                corrected_class = st.selectbox(
                    f"Sửa món {i + 1} nếu nhận diện sai:",
                    CLASS_NAMES,
                    index=current_index,
                    format_func=lambda x: DISPLAY_NAMES[x],
                    key=f"correct_{i}"
                )

                corrected_name = DISPLAY_NAMES[corrected_class]
                corrected_price = PRICE_TABLE[corrected_class]

                st.write(f"**Giá:** {format_money(corrected_price)}")

                total += corrected_price

                bill_rows.append({
                    "STT": i + 1,
                    "Tên món": corrected_name,
                    "Độ tin cậy": f"{det['confidence'] * 100:.2f}%",
                    "Giá": format_money(corrected_price)
                })

            st.markdown("</div>", unsafe_allow_html=True)

        df_bill = pd.DataFrame(bill_rows)
        st.dataframe(df_bill, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="total-box">
                Tổng tiền: {format_money(total)}
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# PHÂN HỆ 2: THỰC ĐƠN
# =========================
elif st.session_state.page == "menu":
    st.markdown('<div class="main-title">📋 Thực đơn căn tin</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Danh sách món ăn và giá bán</div>',
        unsafe_allow_html=True
    )

    if st.button("⬅ Quay lại trang chủ"):
        go_page("home")

    menu_data = []

    for class_name in CLASS_NAMES:
        menu_data.append({
            "Mã món": class_name,
            "Tên món": DISPLAY_NAMES[class_name],
            "Giá": format_money(PRICE_TABLE[class_name])
        })

    df_menu = pd.DataFrame(menu_data)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df_menu, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="small-note">
        Giá món ăn có thể chỉnh trực tiếp trong biến PRICE_TABLE ở file app.py.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# PHÂN HỆ 3: ĐẶT MÓN TỪ XA
# =========================
elif st.session_state.page == "order":
    st.markdown('<div class="main-title">🛒 Đặt món từ xa</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Chọn món, số lượng và tính tổng đơn hàng</div>',
        unsafe_allow_html=True
    )

    if st.button("⬅ Quay lại trang chủ"):
        go_page("home")

    st.markdown("### Tạo đơn hàng")

    order_rows = []
    subtotal = 0

    for class_name in CLASS_NAMES:
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"**{DISPLAY_NAMES[class_name]}**")

        with col2:
            st.write(format_money(PRICE_TABLE[class_name]))

        with col3:
            qty = st.number_input(
                "Số lượng",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                key=f"qty_{class_name}",
                label_visibility="collapsed"
            )

        if qty > 0:
            item_total = PRICE_TABLE[class_name] * qty
            subtotal += item_total

            order_rows.append({
                "Tên món": DISPLAY_NAMES[class_name],
                "Số lượng": qty,
                "Đơn giá": format_money(PRICE_TABLE[class_name]),
                "Thành tiền": format_money(item_total)
            })

    st.markdown("---")

    voucher = st.text_input("Nhập mã giảm giá nếu có:")

    discount = 0

    if voucher.strip().upper() == "AD10":
        discount = int(subtotal * 0.10)
    elif voucher.strip().upper() == "AD20":
        discount = int(subtotal * 0.20)

    final_total = subtotal - discount

    if len(order_rows) > 0:
        st.markdown("### Chi tiết đơn hàng")

        df_order = pd.DataFrame(order_rows)
        st.dataframe(df_order, use_container_width=True, hide_index=True)

        st.write(f"**Tạm tính:** {format_money(subtotal)}")
        st.write(f"**Giảm giá:** {format_money(discount)}")

        st.markdown(
            f"""
            <div class="total-box">
                Tổng thanh toán: {format_money(final_total)}
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("✅ Xác nhận đặt món"):
            st.success("Đơn hàng đã được tạo thành công!")

    else:
        st.info("Anh hãy chọn số lượng món để tạo đơn hàng.")

    st.markdown("### Mã giảm giá mẫu")
    st.write("`AD10`: giảm 10%")
    st.write("`AD20`: giảm 20%")