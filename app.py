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

LEFT_STICKER_B64 = "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"
RIGHT_STICKER_B64 = "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"


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
    page_title="Bụng Đói Canteen",
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


    /* ===== FIX KHUNG Ô NHẬP THÀNH VIÊN: KHÔNG BỊ LỒNG KHUNG / CẮT VIỀN ===== */
    div[data-testid="stTextInput"] {
        margin-bottom: 14px !important;
    }

    div[data-testid="stTextInput"] label p {
        color: #7c2d12 !important;
        font-size: 18px !important;
        font-weight: 950 !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        border: 2px solid #f97316 !important;
        border-radius: 14px !important;
        color: #111827 !important;
        font-size: 18px !important;
        font-weight: 750 !important;
        min-height: 56px !important;
        padding: 12px 16px !important;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.18) !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border: 3px solid #ea580c !important;
        box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.18) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9ca3af !important;
        font-weight: 700 !important;
    }


    /* ===== FIX CUỐI CÙNG CHO Ô NHẬP: STYLE ĐÚNG WRAPPER, BỎ KHUNG LỖI ===== */
    div[data-testid="stTextInput"] {
        margin-bottom: 14px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="input"] {
        background: #ffffff !important;
        border: 2px solid #f97316 !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.16) !important;
        overflow: hidden !important;
    }

    div[data-testid="stTextInput"] input {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
        color: #111827 !important;
        font-size: 18px !important;
        font-weight: 750 !important;
        min-height: 54px !important;
        padding: 12px 16px !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
        border: 3px solid #ea580c !important;
        box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.18) !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9ca3af !important;
        font-weight: 700 !important;
    }



    /* ===== STICKER TRANG CHÍNH ===== */
    .home-sticker-row {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: -8px;
        padding: 2px 6px 0 6px;
        box-sizing: border-box;
    }

    .home-sticker-row img {
        width: 120px;
        height: 120px;
        object-fit: contain;
        border-radius: 20px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        background: transparent;
    }

    @media (max-width: 768px) {
        .home-sticker-row img {
            width: 84px;
            height: 84px;
        }
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
    lines = ["Bụng Đói Canteen", f"Tong tien: {format_money(total)}", "Chi tiet mon:"]

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



def render_home_stickers():
    st.markdown(
        f"""
        <div class="home-sticker-row">
            <img src="data:image/jpeg;base64,{LEFT_STICKER_B64}" alt="left sticker">
            <img src="data:image/jpeg;base64,{RIGHT_STICKER_B64}" alt="right sticker">
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":
    render_home_stickers()
    st.markdown('<div class="main-title">🍱 Bụng Đói Canteen</div>', unsafe_allow_html=True)
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
            <div class="mode-title">💳 Chọn kiểu thanh toán</div>
            <div class="mode-desc">Quý khách chọn thanh toán cho 1 khay hiện tại hoặc gom nhiều khay rồi thanh toán một lần.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tray_pay_mode = st.radio(
        "Chọn kiểu thanh toán",
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
                <p>Quý khách có thể chụp thêm khay khác hoặc thanh toán tất cả các khay đã lưu.</p>
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
            if st.button("💳 Thanh toán tất cả khay đã lưu", use_container_width=True):
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
        st.markdown("### 💳 Thanh toán")

        order_rows = st.session_state.checkout_rows
        total = st.session_state.checkout_total

        if len(order_rows) == 0:
            st.warning("Chưa có hóa đơn để thanh toán.")
            if st.button("⬅ Quay lại nhận diện"):
                clear_result()
                st.rerun()
            st.stop()

        st.markdown(
            """
            <div class="payment-card">
                <h3>Chi tiết hóa đơn</h3>
                <p>Kiểm tra danh sách món và chọn hình thức thanh toán.</p>
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
                    Sau thanh toán: {after_points} ⭐
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            earned_preview = 0
            st.info("Không nhập mã thành viên thì vẫn thanh toán bình thường, nhưng đơn này sẽ không được cộng điểm hoặc đổi điểm.")

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
                <h3>💵 Chọn phương thức thanh toán</h3>
                <p>Quý khách chọn tiền mặt hoặc chuyển khoản cho hóa đơn hiện tại.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        payment_method = st.radio(
            "Chọn phương thức thanh toán:",
            ["Tiền mặt", "Chuyển khoản"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if payment_method == "Tiền mặt":
            st.success(f"Khách thanh toán bằng tiền mặt: {format_money(payable_total)}. Sau khi thu tiền, bấm Hoàn tất thanh toán.")
        else:
            st.markdown(
                """
                <div class="qr-card">
                    <h3>QR chuyển khoản</h3>
                    <p>Khách quét mã QR bên dưới để thanh toán.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            qr_img, qr_content = create_payment_qr(order_rows, payable_total)

            if qr_img is not None:
                st.image(qr_img, caption="QR thanh toán", width=280)
            else:
                qr_url = "https://api.qrserver.com/v1/create-qr-code/?size=280x280&data=" + quote(qr_content)
                st.image(qr_url, caption="QR thanh toán", width=280)

            with st.expander("Nội dung QR"):
                st.code(qr_content)


        col_pay_back, col_finish = st.columns(2)

        with col_pay_back:
            if st.button("⬅ Quay lại hóa đơn", use_container_width=True):
                st.session_state.payment_step = "scan"
                st.rerun()

        with col_finish:
            if st.button("✅ Hoàn tất thanh toán", use_container_width=True):
                points_info = update_member_points_after_payment(
                    st.session_state.member_id,
                    st.session_state.member_name,
                    payable_total,
                    used_points=used_points
                )

                success_message = f"Thanh toán hoàn tất. Số tiền đã thanh toán: {format_money(payable_total)}."

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
            if st.button("💳 Thanh toán khay này", use_container_width=True):
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
                if st.button("💳 Thanh toán khay này + các khay đã lưu", use_container_width=True):
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


