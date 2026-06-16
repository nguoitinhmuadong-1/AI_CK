import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from urllib.parse import quote
import os

try:
    import qrcode
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False


# =========================
# CẤU HÌNH MODEL
# =========================
MODEL_PATH = "best_food_mobilenetv2.h5"
IMG_SIZE = 224

CLASS_NAMES = [
    "ca_hu_kho",
    "cquý khách_chua",
    "cquý khách_rau",
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
    "cquý khách_chua": "Cquý khách chua",
    "cquý khách_rau": "Cquý khách rau",
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
    "cquý khách_chua": 10000,
    "cquý khách_rau": 7000,
    "com_trang": 10000,
    "dau_hu_sot_ca": 25000,
    "rau_xao": 10000,
    "suon_nuong": 30000,
    "thit_kho": 25000,
    "thit_kho_trung": 30000,
    "trung_chien": 25000
}

# =========================
# CẤU HÌNH TÍCH ĐIỂM THÀNH VIÊN
# =========================
MEMBER_POINTS_FILE = "member_points.csv"
POINT_MONEY_RATE = 10000  # 10.000đ = 1 điểm
POINT_REDEEM_VALUE = 1000  # 1 điểm đổi được 1.000đ
POINT_MAX_DISCOUNT_RATE = 0.5  # Điểm chỉ được giảm tối đa 50% hóa đơn



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

    .input-card {
        background: rgba(255,255,255,0.92);
        border: 2px solid #fed7aa;
        border-radius: 22px;
        padding: 20px;
        margin: 12px 0 18px 0;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .input-card h3 {
        color: #c2410c;
        margin-bottom: 8px;
    }

    .camera-guide {
        background: #fff7ed;
        border-left: 6px solid #f97316;
        border-radius: 16px;
        padding: 14px 18px;
        color: #7c2d12;
        font-weight: 600;
        margin-bottom: 14px;
    }

    div[data-testid="stCameraInput"] {
        background: white;
        border: 2px dashed #fb923c;
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    div[data-testid="stCameraInput"] video {
        border-radius: 18px;
        border: 4px solid #fdba74;
    }

    div[data-testid="stCameraInput"] button {
        min-height: 62px !important;
        width: 100% !important;
        border-radius: 18px !important;
        background: linear-gradient(135deg, #f97316, #dc2626) !important;
        color: white !important;
        font-size: 19px !important;
        font-weight: 900 !important;
        margin-top: 10px !important;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #fb923c;
        border-radius: 22px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    div[data-testid="stFileUploader"] button {
        min-height: 52px !important;
        border-radius: 16px !important;
        background: #f97316 !important;
        color: white !important;
        font-weight: 800 !important;
    }


    .payment-card {
        background: white;
        border: 2px solid #fed7aa;
        border-radius: 22px;
        padding: 22px;
        margin-top: 18px;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .payment-card h3 {
        color: #c2410c;
        margin-top: 0;
    }

    .member-card {
        background: linear-gradient(135deg, #ffffff, #fff7ed);
        border: 2px solid #fdba74;
        border-radius: 22px;
        padding: 22px;
        margin-top: 18px;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .member-card h3 {
        color: #c2410c;
        margin-top: 0;
    }

    .point-box {
        background: linear-gradient(135deg, #fef3c7, #fed7aa);
        border: 2px solid #fb923c;
        border-radius: 18px;
        padding: 16px;
        text-align: center;
        color: #7c2d12;
        font-weight: 900;
        font-size: 20px;
    }

    .qr-card {
        background: #fff7ed;
        border: 2px dashed #fb923c;
        border-radius: 22px;
        padding: 20px;
        text-align: center;
        color: #7c2d12;
        margin-top: 12px;
    }

    .review-card {
        background: white;
        border: 2px solid #fed7aa;
        border-radius: 22px;
        padding: 22px;
        margin: 14px 0;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .summary-card {
        background: white;
        border: 2px solid #fed7aa;
        border-radius: 22px;
        padding: 18px 22px;
        margin: 14px 0;
        color: #1f2937;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    .summary-card h3 {
        color: #c2410c;
        margin-top: 0;
    }



    .mode-card {
        background: linear-gradient(135deg, #ffffff, #fff7ed);
        border: 3px solid #fb923c;
        border-radius: 24px;
        padding: 22px 26px;
        margin: 16px 0 12px 0;
        color: #7c2d12;
        box-shadow: 0 8px 26px rgba(249, 115, 22, 0.22);
    }

    .mode-title {
        font-size: 26px;
        font-weight: 900;
        color: #9a3412;
        margin-bottom: 6px;
    }

    .mode-desc {
        font-size: 17px;
        font-weight: 650;
        color: #7c2d12;
    }

    div[data-testid="stRadio"] {
        background: rgba(255, 255, 255, 0.96);
        border: 2px solid #fdba74;
        border-radius: 20px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.08);
        width: 100% !important;
        display: block !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stRadio"] > div {
        width: 100% !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] {
        width: 100% !important;
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 14px !important;
    }

    div[data-testid="stRadio"] label {
        background: #fff7ed;
        border: 2px solid #fed7aa;
        border-radius: 16px;
        padding: 12px 18px;
        margin-right: 16px;
        color: #7c2d12 !important;
        font-size: 18px !important;
        font-weight: 850 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }

    div[data-testid="stRadio"] label:hover {
        background: #ffedd5;
        border-color: #fb923c;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, #fed7aa, #fdba74);
        border-color: #ea580c;
        box-shadow: 0 5px 16px rgba(249, 115, 22, 0.35);
    }


    /* Làm nổi phần tổng tiền / giảm điểm / còn phải trả */
    [data-testid="stMetric"] {
        background: #fff7ed !important;
        border: 2px solid #fb923c !important;
        border-radius: 18px !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 16px rgba(249, 115, 22, 0.18) !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {
        color: #7c2d12 !important;
        font-weight: 900 !important;
        font-size: 16px !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #9a3412 !important;
        font-weight: 950 !important;
        font-size: 34px !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] div,
    [data-testid="stMetricDelta"] svg {
        color: #166534 !important;
        fill: #166534 !important;
    }


    /* ===== FIX RADIO FULL WIDTH - CHỌN KIỂU THANH TOÁN / PHƯƠNG THỨC THANH TOÁN ===== */
    div.stRadio,
    div[data-testid="stRadio"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        display: block !important;
        box-sizing: border-box !important;
        align-self: stretch !important;
    }

    div.stRadio > div,
    div[data-testid="stRadio"] > div {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        box-sizing: border-box !important;
    }

    div.stRadio div[role="radiogroup"],
    div[data-testid="stRadio"] div[role="radiogroup"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        display: grid !important;
        grid-template-columns: repeat(2, minmax(240px, 1fr)) !important;
        gap: 18px !important;
        box-sizing: border-box !important;
    }

    div.stRadio div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        margin: 0 !important;
        padding: 16px 20px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-sizing: border-box !important;
    }

    .mode-card,
    .payment-card,
    .member-card,
    .summary-card,
    .total-box,
    .point-box {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }


    /* ===== RADIO GIỮ GIAO DIỆN CŨ NHƯNG ÉP FULL WIDTH ===== */
    div[data-testid="stElementContainer"]:has(div.stRadio),
    div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]),
    div[data-testid="stVerticalBlock"] div:has(> div.stRadio),
    div[data-testid="stVerticalBlock"] div:has(> div[data-testid="stRadio"]) {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        display: block !important;
        box-sizing: border-box !important;
    }

    div.stRadio,
    div[data-testid="stRadio"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        display: block !important;
        box-sizing: border-box !important;
        background: rgba(255, 255, 255, 0.92) !important;
        border: 2px solid #fdba74 !important;
        border-radius: 20px !important;
        padding: 22px 28px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 5px 18px rgba(0,0,0,0.08) !important;
    }

    div.stRadio > div,
    div[data-testid="stRadio"] > div {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        box-sizing: border-box !important;
    }

    div.stRadio div[role="radiogroup"],
    div[data-testid="stRadio"] div[role="radiogroup"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 24px !important;
        align-items: stretch !important;
        box-sizing: border-box !important;
    }

    div.stRadio div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        min-height: 70px !important;
        margin: 0 !important;
        padding: 16px 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-sizing: border-box !important;
        background: #fff7ed !important;
        border: 2px solid #fed7aa !important;
        border-radius: 18px !important;
        color: #7c2d12 !important;
        font-size: 18px !important;
        font-weight: 850 !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06) !important;
    }

    div.stRadio div[role="radiogroup"] label:hover,
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #ffedd5 !important;
        border-color: #fb923c !important;
    }

    div.stRadio div[role="radiogroup"] label:has(input:checked),
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #fed7aa, #fdba74) !important;
        border-color: #ea580c !important;
        box-shadow: 0 5px 16px rgba(249, 115, 22, 0.35) !important;
    }

    .mode-card,
    .payment-card,
    .member-card,
    .summary-card,
    .total-box,
    .point-box {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }


    /* ===== LÀM NỔI BẬT PHẦN THÔNG TIN THÀNH VIÊN ===== */
    .member-input-card {
        background: linear-gradient(135deg, #fff7ed, #fed7aa);
        border: 3px solid #f97316;
        border-radius: 24px;
        padding: 22px 26px;
        margin: 18px 0 16px 0;
        color: #7c2d12;
        box-shadow: 0 8px 28px rgba(249, 115, 22, 0.24);
        box-sizing: border-box;
    }

    .member-input-card h3 {
        color: #9a3412;
        font-size: 27px;
        font-weight: 950;
        margin: 0 0 8px 0;
    }

    .member-input-card p {
        color: #7c2d12;
        font-size: 17px;
        font-weight: 750;
        margin: 0;
    }

    div[data-testid="stTextInput"] {
        background: rgba(255, 255, 255, 0.90);
        border: 2px solid #fdba74;
        border-radius: 18px;
        padding: 14px 16px 16px 16px;
        box-shadow: 0 5px 16px rgba(249, 115, 22, 0.14);
        margin-bottom: 12px;
    }

    div[data-testid="stTextInput"] label p {
        color: #7c2d12 !important;
        font-size: 17px !important;
        font-weight: 900 !important;
    }

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        border: 2px solid #fb923c !important;
        border-radius: 14px !important;
        color: #111827 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        min-height: 54px !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9ca3af !important;
        font-weight: 650 !important;
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


def normalize_member_id(value):
    return "".join(ch for ch in str(value).strip() if ch.isalnum())


def load_member_points():
    if not os.path.exists(MEMBER_POINTS_FILE):
        return {}

    try:
        df = pd.read_csv(MEMBER_POINTS_FILE, dtype={"member_id": str, "member_name": str})
    except Exception:
        return {}

    points_data = {}
    for _, row in df.iterrows():
        member_id = normalize_member_id(row.get("member_id", ""))
        if not member_id:
            continue

        try:
            points = int(row.get("points", 0))
        except Exception:
            points = 0

        member_name = str(row.get("member_name", "")).strip()
        points_data[member_id] = {
            "member_name": member_name,
            "points": points
        }

    return points_data


def save_member_points(points_data):
    rows = []
    for member_id, info in points_data.items():
        rows.append({
            "member_id": member_id,
            "member_name": info.get("member_name", ""),
            "points": int(info.get("points", 0))
        })

    df = pd.DataFrame(rows, columns=["member_id", "member_name", "points"])
    df.to_csv(MEMBER_POINTS_FILE, index=False, encoding="utf-8-sig")


def get_member_info(member_id):
    member_id = normalize_member_id(member_id)
    points_data = load_member_points()

    if member_id in points_data:
        return points_data[member_id]

    return {"member_name": "", "points": 0}


def update_member_points_after_payment(member_id, member_name, payable_money, used_points=0):
    member_id = normalize_member_id(member_id)
    member_name = str(member_name).strip()

    if not member_id:
        return None

    points_data = load_member_points()
    old_info = points_data.get(member_id, {"member_name": member_name, "points": 0})
    old_points = int(old_info.get("points", 0))

    if not member_name:
        member_name = old_info.get("member_name", "")

    used_points = int(max(0, min(used_points, old_points)))
    discount_money = used_points * POINT_REDEEM_VALUE
    earned_points = int(payable_money // POINT_MONEY_RATE)
    new_points = old_points - used_points + earned_points

    points_data[member_id] = {
        "member_name": member_name,
        "points": new_points
    }
    save_member_points(points_data)

    return {
        "member_id": member_id,
        "member_name": member_name,
        "old_points": old_points,
        "used_points": used_points,
        "discount_money": discount_money,
        "earned_points": earned_points,
        "new_points": new_points
    }


def build_order_text(order_rows, total):
    lines = ["AD VietFood Vision", f"Tong tien: {format_money(total)}", "Chi tiet mon:"]

    for row in order_rows:
        lines.append(f"- {row['Tên món']}: {row['Giá']}")

    return "\n".join(lines)


def create_payment_qr(order_rows, total):
    qr_content = build_order_text(order_rows, total)

    if QR_AVAILABLE:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=4
        )
        qr.add_data(qr_content)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        return qr_img, qr_content

    return None, qr_content


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

if "payment_step" not in st.session_state:
    st.session_state.payment_step = "scan"

if "checkout_rows" not in st.session_state:
    st.session_state.checkout_rows = []

if "checkout_total" not in st.session_state:
    st.session_state.checkout_total = 0

if "multi_order_rows" not in st.session_state:
    st.session_state.multi_order_rows = []

if "multi_total" not in st.session_state:
    st.session_state.multi_total = 0

if "tray_count" not in st.session_state:
    st.session_state.tray_count = 0


if "member_id" not in st.session_state:
    st.session_state.member_id = ""

if "member_name" not in st.session_state:
    st.session_state.member_name = ""

if "last_points_info" not in st.session_state:
    st.session_state.last_points_info = None

if "payment_success_message" not in st.session_state:
    st.session_state.payment_success_message = ""


def go_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def clear_current_scan():
    st.session_state.image_rgb = None
    st.session_state.detections = []
    st.session_state.tray_box = None
    st.session_state.found_tray = True


def clear_result():
    clear_current_scan()
    st.session_state.payment_step = "scan"
    st.session_state.checkout_rows = []
    st.session_state.checkout_total = 0
    st.session_state.multi_order_rows = []
    st.session_state.multi_total = 0
    st.session_state.tray_count = 0
    st.session_state.member_id = ""
    st.session_state.member_name = ""
    st.session_state.last_points_info = None


def add_tray_column(rows, tray_number):
    rows_with_tray = []

    for row in rows:
        new_row = row.copy()
        new_row = {"Khay": tray_number, **new_row}
        rows_with_tray.append(new_row)

    return rows_with_tray


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
                <h3>📷 Thquý khách toán tại quầy</h3>
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
                <p>Tra cứu dquý khách sách món ăn, tên món và giá bán trong căn tin.</p>
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
    st.markdown('<div class="main-title">📷 Thquý khách toán tại quầy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">OpenCV cắt 5 ô khay cơm, AI nhận diện từng món và tính tiền</div>',
        unsafe_allow_html=True
    )

    if st.session_state.payment_success_message:
        st.success(st.session_state.payment_success_message)
        st.session_state.payment_success_message = ""

    col_back, col_clear = st.columns(2)

    with col_back:
        if st.button("⬅ Quay lại trang chủ"):
            clear_result()
            go_page("home")

    with col_clear:
        if st.button("🧹 Xóa kết quả"):
            clear_result()
            st.rerun()

    st.markdown(
        """
        <div class="mode-card">
            <div class="mode-title">💳 Chọn kiểu thquý khách toán</div>
            <div class="mode-desc">Quý khách chọn thquý khách toán cho 1 khay hiện tại hoặc gom nhiều khay rồi thquý khách toán một lần.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tray_pay_mode = st.radio(
        "Chọn kiểu thquý khách toán",
        ["1 khay duy nhất", "Nhiều khay cùng lúc"],
        horizontal=True,
        key="tray_pay_mode",
        label_visibility="collapsed"
    )

    if tray_pay_mode == "Nhiều khay cùng lúc" and len(st.session_state.multi_order_rows) > 0 and st.session_state.payment_step == "scan":
        st.markdown("### 🧾 Hóa đơn nhiều khay đã lưu")
        st.markdown(
            f"""
            <div class="summary-card">
                <h3>Đã lưu {st.session_state.tray_count} khay</h3>
                <p>Quý khách có thể chụp thêm khay khác hoặc thquý khách toán tất cả các khay đã lưu.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(
            pd.DataFrame(st.session_state.multi_order_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            f"""
            <div class="total-box">
                Tổng tạm tính nhiều khay: {format_money(st.session_state.multi_total)}
            </div>
            """,
            unsafe_allow_html=True
        )

        col_multi_pay, col_multi_clear = st.columns(2)

        with col_multi_pay:
            if st.button("💳 Thquý khách toán tất cả khay đã lưu", use_container_width=True):
                st.session_state.checkout_rows = st.session_state.multi_order_rows
                st.session_state.checkout_total = st.session_state.multi_total
                st.session_state.payment_step = "checkout"
                st.rerun()

        with col_multi_clear:
            if st.button("🗑 Xóa hóa đơn nhiều khay", use_container_width=True):
                st.session_state.multi_order_rows = []
                st.session_state.multi_total = 0
                st.session_state.tray_count = 0
                clear_current_scan()
                st.rerun()

    # =========================
    # BƯỚC THANH TOÁN
    # =========================
    if st.session_state.payment_step == "checkout":
        st.markdown("### 💳 Thquý khách toán")

        order_rows = st.session_state.checkout_rows
        total = st.session_state.checkout_total

        if len(order_rows) == 0:
            st.warning("Chưa có hóa đơn để thquý khách toán.")
            if st.button("⬅ Quay lại nhận diện"):
                clear_result()
                st.rerun()
            st.stop()

        st.markdown(
            """
            <div class="payment-card">
                <h3>Chi tiết hóa đơn</h3>
                <p>Kiểm tra dquý khách sách món và chọn hình thức thquý khách toán.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        df_checkout = pd.DataFrame(order_rows)
        st.dataframe(df_checkout, use_container_width=True, hide_index=True)

        original_total = int(total)

        st.markdown(
            f"""
            <div class="total-box">
                Tổng hóa đơn: {format_money(original_total)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="member-card">
                <h3>🎁 Tích điểm & đổi điểm thành viên</h3>
                <p>Quý khách nhập số điện thoại hoặc mã thành viên để cộng điểm và dùng điểm giảm tiền.</p>
                <p>
                    <b>Tích điểm:</b> {format_money(POINT_MONEY_RATE)} = 1 điểm<br>
                    <b>Đổi điểm:</b> 1 điểm = {format_money(POINT_REDEEM_VALUE)} giảm trực tiếp vào hóa đơn
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="member-input-card">
                <h3>👤 Thông tin thành viên</h3>
                <p>Quý khách nhập số điện thoại hoặc mã thành viên để tích điểm và dùng điểm giảm trực tiếp trên hóa đơn.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_member_id, col_member_name = st.columns(2)

        with col_member_id:
            member_id = st.text_input(
                "Số điện thoại / Mã thành viên",
                value=st.session_state.member_id,
                placeholder="Ví dụ: 0912345678",
                key="checkout_member_id"
            )

        with col_member_name:
            member_name = st.text_input(
                "Tên khách hàng",
                value=st.session_state.member_name,
                placeholder="Ví dụ: Nguyễn Văn A",
                key="checkout_member_name"
            )

        st.session_state.member_id = member_id
        st.session_state.member_name = member_name

        member_id_clean = normalize_member_id(member_id)
        used_points = 0
        discount_money = 0
        payable_total = original_total

        if member_id_clean:
            member_info = get_member_info(member_id_clean)
            current_points = int(member_info.get("points", 0))
            max_discount_money = int(original_total * POINT_MAX_DISCOUNT_RATE)
            max_points_by_bill = max_discount_money // POINT_REDEEM_VALUE
            max_redeem_points = int(max(0, min(current_points, max_points_by_bill)))

            if max_redeem_points > 0:
                used_points = st.number_input(
                    "Dùng điểm để giảm tiền",
                    min_value=0,
                    max_value=max_redeem_points,
                    value=0,
                    step=1,
                    help="1 điểm = 1.000đ. Mỗi hóa đơn được giảm tối đa 50% bằng điểm.",
                    key="use_member_points"
                )
            else:
                st.caption("Thành viên chưa đủ điểm để đổi hoặc hóa đơn quá thấp để áp dụng điểm.")

            discount_money = int(used_points) * POINT_REDEEM_VALUE
            payable_total = max(0, original_total - discount_money)
            earned_preview = int(payable_total // POINT_MONEY_RATE)
            after_points = current_points - int(used_points) + earned_preview

            st.markdown(
                f"""
                <div class="point-box">
                    Điểm hiện có: {current_points} ⭐ &nbsp; | &nbsp;
                    Dùng điểm: -{int(used_points)} ⭐ &nbsp; | &nbsp;
                    Giảm: {format_money(discount_money)}<br>
                    Điểm cộng thêm: +{earned_preview} ⭐ &nbsp; | &nbsp;
                    Sau thquý khách toán: {after_points} ⭐
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            earned_preview = 0
            st.info("Không nhập mã thành viên thì vẫn thquý khách toán bình thường, nhưng đơn này sẽ không được cộng điểm hoặc đổi điểm.")

        if discount_money > 0:
            col_old_total, col_discount, col_payable = st.columns(3)
            with col_old_total:
                st.metric("Tổng hóa đơn", format_money(original_total))
            with col_discount:
                st.metric("Giảm bằng điểm", f"- {format_money(discount_money)}")
            with col_payable:
                st.metric("Còn phải trả", format_money(payable_total))
        else:
            payable_total = original_total

        st.markdown(
            """
            <div class="payment-card">
                <h3>💵 Chọn phương thức thquý khách toán</h3>
                <p>Quý khách chọn tiền mặt hoặc chuyển khoản cho hóa đơn hiện tại.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        payment_method = st.radio(
            "Chọn phương thức thquý khách toán:",
            ["Tiền mặt", "Chuyển khoản"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if payment_method == "Tiền mặt":
            st.success(f"Khách thquý khách toán bằng tiền mặt: {format_money(payable_total)}. Sau khi thu tiền, bấm Hoàn tất thquý khách toán.")
        else:
            st.markdown(
                """
                <div class="qr-card">
                    <h3>QR chuyển khoản</h3>
                    <p>Khách quét mã QR bên dưới để thquý khách toán.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            qr_img, qr_content = create_payment_qr(order_rows, payable_total)

            if qr_img is not None:
                st.image(qr_img, caption="QR thquý khách toán", width=280)
            else:
                qr_url = "https://api.qrserver.com/v1/create-qr-code/?size=280x280&data=" + quote(qr_content)
                st.image(qr_url, caption="QR thquý khách toán", width=280)

            with st.expander("Nội dung QR"):
                st.code(qr_content)


        col_pay_back, col_finish = st.columns(2)

        with col_pay_back:
            if st.button("⬅ Quay lại hóa đơn", use_container_width=True):
                st.session_state.payment_step = "scan"
                st.rerun()

        with col_finish:
            if st.button("✅ Hoàn tất thquý khách toán", use_container_width=True):
                points_info = update_member_points_after_payment(
                    st.session_state.member_id,
                    st.session_state.member_name,
                    payable_total,
                    used_points=used_points
                )

                success_message = f"Thquý khách toán hoàn tất. Số tiền đã thquý khách toán: {format_money(payable_total)}."

                if points_info is not None:
                    extra_message = f" Đã cộng {points_info['earned_points']} điểm."
                    if int(points_info.get("used_points", 0)) > 0:
                        extra_message = (
                            f" Đã dùng {points_info['used_points']} điểm để giảm "
                            f"{format_money(points_info['discount_money'])}. "
                            f"Đã cộng {points_info['earned_points']} điểm."
                        )
                    success_message += extra_message

                clear_result()
                st.session_state.payment_success_message = success_message
                st.session_state.page = "payment"
                st.rerun()

        st.stop()

    food_model = load_food_model()

    st.markdown("### Chọn ảnh khay cơm")

    st.markdown(
        """
        <div class="input-card">
            <h3>📸 Chụp hoặc tải ảnh khay cơm</h3>
            <p>
            Quý khách có thể chọn ảnh có sẵn hoặc chụp trực tiếp. Khi chụp, đặt khay nằm giữa khung hình,
            chụp thẳng từ trên xuống để OpenCV cắt 5 ô chính xác hơn.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = None
    camera_file = None

    tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📸 Chụp trực tiếp"])

    with tab_upload:
        st.markdown(
            """
            <div class="camera-guide">
            Cách dễ dùng nhất trên điện thoại: bấm nút tải ảnh rồi chọn Camera hoặc Thư viện ảnh.
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Chọn ảnh khay cơm",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

    with tab_camera:
        st.markdown(
            """
            <div class="camera-guide">
            Bấm nút chụp bên dưới. Nút chụp đã được phóng to để dễ thao tác hơn.
            </div>
            """,
            unsafe_allow_html=True
        )
        camera_file = st.camera_input(
            "Đưa khay vào giữa khung hình rồi chụp",
            key="camera_input_big"
        )

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
                "Nếu box bị lệch, quý khách nên chụp ảnh thẳng từ trên xuống và để khay nằm giữa ảnh."
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

        if tray_pay_mode == "1 khay duy nhất":
            if st.button("💳 Thquý khách toán khay này", use_container_width=True):
                st.session_state.checkout_rows = bill_rows
                st.session_state.checkout_total = total
                st.session_state.payment_step = "checkout"
                st.rerun()
        else:
            current_tray_number = st.session_state.tray_count + 1
            current_rows_with_tray = add_tray_column(bill_rows, current_tray_number)

            col_add_tray, col_pay_all = st.columns(2)

            with col_add_tray:
                if st.button("➕ Lưu khay này / chụp khay tiếp theo", use_container_width=True):
                    st.session_state.multi_order_rows.extend(current_rows_with_tray)
                    st.session_state.multi_total += total
                    st.session_state.tray_count = current_tray_number
                    clear_current_scan()
                    st.success(f"Đã lưu khay {current_tray_number}. Quý khách có thể chụp khay tiếp theo.")
                    st.rerun()

            with col_pay_all:
                if st.button("💳 Thquý khách toán khay này + các khay đã lưu", use_container_width=True):
                    all_rows = st.session_state.multi_order_rows + current_rows_with_tray
                    all_total = st.session_state.multi_total + total
                    st.session_state.checkout_rows = all_rows
                    st.session_state.checkout_total = all_total
                    st.session_state.payment_step = "checkout"
                    st.rerun()


# =========================
# PHÂN HỆ 2: THỰC ĐƠN
# =========================
elif st.session_state.page == "menu":
    st.markdown('<div class="main-title">📋 Thực đơn căn tin</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dquý khách sách món ăn và giá bán</div>',
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


