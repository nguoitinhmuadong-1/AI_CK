import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


# =========================
# CẤU HÌNH MODEL
# =========================
MODEL_PATH = "best_food_mobilenetv2.h5"
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

    .info-card {
        background: white;
        padding: 24px;
        border-radius: 22px;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.10);
        margin-top: 20px;
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
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_food_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Không tìm thấy file model: {MODEL_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


# =========================
# HÀM PHỤ
# =========================
def format_money(value):
    return f"{value:,.0f} đ".replace(",", ".")


def preprocess_crop(crop_rgb):
    img = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict_food(crop_rgb, model):
    x = preprocess_crop(crop_rgb)
    pred = model.predict(x, verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])

    class_name = CLASS_NAMES[class_id]
    display_name = DISPLAY_NAMES[class_name]
    price = PRICE_TABLE[class_name]

    top_indices = np.argsort(pred)[::-1][:3]
    top3 = []

    for idx in top_indices:
        idx = int(idx)
        key = CLASS_NAMES[idx]
        top3.append({
            "class_name": key,
            "display_name": DISPLAY_NAMES[key],
            "confidence": float(pred[idx])
        })

    return class_name, display_name, confidence, price, top3


def top3_to_text(top3):
    return " | ".join([
        f"{item['display_name']} {item['confidence'] * 100:.1f}%"
        for item in top3
    ])


def detect_tray_box(image_rgb):
    """
    OpenCV tìm vùng khay lớn trong ảnh.
    Nếu tìm không được thì dùng gần toàn bộ ảnh.
    """
    h, w = image_rgb.shape[:2]

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((9, 9), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    image_area = h * w

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < image_area * 0.25:
            continue

        if area > image_area * 0.98:
            continue

        ratio = bw / float(bh)

        if 1.0 <= ratio <= 2.8:
            candidates.append((x, y, bw, bh, area))

    if len(candidates) > 0:
        x, y, bw, bh, _ = max(candidates, key=lambda item: item[4])

        pad_x = int(bw * 0.02)
        pad_y = int(bh * 0.02)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        return [x1, y1, x2 - x1, y2 - y1], True

    return [
        int(0.03 * w),
        int(0.08 * h),
        int(0.94 * w),
        int(0.86 * h)
    ], False


def get_5_tray_boxes(image_rgb):
    """
    Cắt khay thành 5 vùng:
    3 ô nhỏ phía trên + 2 ô phía dưới.
    """
    tray_box, found_tray = detect_tray_box(image_rgb)
    tx, ty, tw, th = tray_box

    boxes = [
        # Ô 1: trên trái
        [
            tx + int(0.03 * tw),
            ty + int(0.07 * th),
            int(0.28 * tw),
            int(0.39 * th)
        ],

        # Ô 2: trên giữa
        [
            tx + int(0.31 * tw),
            ty + int(0.07 * th),
            int(0.28 * tw),
            int(0.39 * th)
        ],

        # Ô 3: trên phải
        [
            tx + int(0.59 * tw),
            ty + int(0.07 * th),
            int(0.39 * tw),
            int(0.39 * th)
        ],

        # Ô 4: dưới trái
        [
            tx + int(0.03 * tw),
            ty + int(0.48 * th),
            int(0.40 * tw),
            int(0.49 * th)
        ],

        # Ô 5: dưới phải
        [
            tx + int(0.43 * tw),
            ty + int(0.48 * th),
            int(0.55 * tw),
            int(0.49 * th)
        ],
    ]

    h, w = image_rgb.shape[:2]
    final_boxes = []

    for x, y, bw, bh in boxes:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        final_boxes.append([x1, y1, x2 - x1, y2 - y1])

    return final_boxes, tray_box, found_tray


def recognize_image(image_rgb, model):
    detections = []

    # Chỉ nhận diện khay 5 món
    boxes, tray_box, found_tray = get_5_tray_boxes(image_rgb)

    for box in boxes:
        x, y, bw, bh = box
        crop = image_rgb[y:y + bh, x:x + bw]

        if crop.size == 0:
            continue

        class_name, display_name, confidence, price, top3 = predict_food(crop, model)

        detections.append({
            "box": box,
            "crop": crop,
            "class_name": class_name,
            "display_name": display_name,
            "confidence": confidence,
            "price": price,
            "top3": top3
        })

    return detections, tray_box, found_tray


def draw_boxes(image_rgb, detections, tray_box=None):
    image_draw = image_rgb.copy()

    if tray_box is not None:
        x, y, w, h = tray_box
        cv2.rectangle(image_draw, (x, y), (x + w, y + h), (0, 180, 255), 4)
        cv2.putText(
            image_draw,
            "Khay com",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 255),
            2,
            cv2.LINE_AA
        )

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
            0.75,
            (255, 102, 0),
            2,
            cv2.LINE_AA
        )

    return image_draw


# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page not in ["home", "payment", "menu"]:
    st.session_state.page = "home"

if "image_rgb" not in st.session_state:
    st.session_state.image_rgb = None

if "detections" not in st.session_state:
    st.session_state.detections = []

if "tray_box" not in st.session_state:
    st.session_state.tray_box = None

if "found_tray" not in st.session_state:
    st.session_state.found_tray = True

if "result_id" not in st.session_state:
    st.session_state.result_id = 0


def go_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def clear_result():
    st.session_state.image_rgb = None
    st.session_state.detections = []
    st.session_state.tray_box = None
    st.session_state.found_tray = True


# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":
    st.markdown('<div class="main-title">🍱 AD VietFood Vision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Hệ thống nhận diện và tính tiền khay cơm căn tin tự động</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📷 Thanh toán tại quầy</h3>
                <p>Chụp hoặc tải ảnh khay cơm, hệ thống tự cắt 5 ô, nhận diện món ăn và tính tiền.</p>
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


    st.markdown(
        """
        <div class="info-card">
            <h3 style="color:#c2410c;">Giới thiệu hệ thống</h3>
            <p>
            Ứng dụng sử dụng Python, Streamlit, OpenCV và TensorFlow/Keras để xây dựng
            hệ thống nhận diện món ăn trên khay cơm. OpenCV hỗ trợ tìm khay và cắt 5 vùng món ăn,
            sau đó mô hình CNN MobileNetV2 nhận diện từng món và hệ thống tự động tính tổng hóa đơn.
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
        '<div class="sub-title">OpenCV cắt 5 ô khay cơm, AI nhận diện từng món và tính tiền</div>',
        unsafe_allow_html=True
    )

    col_back, col_clear = st.columns(2)

    with col_back:
        if st.button("⬅ Quay lại trang chủ"):
            clear_result()
            go_page("home")

    with col_clear:
        if st.button("🧹 Xóa kết quả"):
            clear_result()
            st.rerun()

    food_model = load_food_model()

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
            detections, tray_box, found_tray = recognize_image(
                image_rgb,
                food_model
            )

            st.session_state.detections = detections
            st.session_state.tray_box = tray_box
            st.session_state.found_tray = found_tray
            st.session_state.result_id += 1

            st.rerun()

    if st.session_state.image_rgb is not None and len(st.session_state.detections) > 0:
        st.markdown("### Ảnh sau khi OpenCV cắt vùng")

        image_draw = draw_boxes(
            st.session_state.image_rgb,
            st.session_state.detections,
            st.session_state.tray_box
        )
        st.image(image_draw, use_container_width=True)

        if st.session_state.found_tray is False and st.session_state.tray_box is not None:
            st.warning(
                "OpenCV chưa tìm được khay rõ ràng nên app dùng vùng khay mặc định. "
                "Nếu box bị lệch, anh nên chụp ảnh thẳng từ trên xuống và để khay nằm giữa ảnh."
            )

        st.markdown("### Hóa đơn món ăn")

        total = 0
        bill_rows = []

        for i, det in enumerate(st.session_state.detections):
            with st.container(border=True):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    st.image(det["crop"], caption=f"Món {i + 1}", use_container_width=True)

                with col_info:
                    st.write(f"**Dự đoán:** {det['display_name']}")
                    st.write(f"**Độ tin cậy:** {det['confidence'] * 100:.2f}%")
                    st.caption("Top 3: " + top3_to_text(det["top3"]))

                    current_index = CLASS_NAMES.index(det["class_name"])

                    corrected_class = st.selectbox(
                        f"Sửa món {i + 1} nếu nhận diện sai:",
                        CLASS_NAMES,
                        index=current_index,
                        format_func=lambda x: DISPLAY_NAMES[x],
                        key=f"correct_{st.session_state.result_id}_{i}"
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
    st.dataframe(df_menu, use_container_width=True, hide_index=True)

    st.info("Anh có thể đổi giá món trong biến PRICE_TABLE ở đầu file app.py.")

