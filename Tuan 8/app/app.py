import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score

# ─── CẤU HÌNH GIAO DIỆN STREAMLIT ─────────────────────────────────────
st.set_page_config(
    page_title="Đồ Án Tốt Nghiệp - Hội Đồng Dự Báo BTC",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style CSS tối giản, hiện đại (Tương thích Dark/Light Mode)
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c23;
        border: 1px solid #2e3440;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #00e676;
    }
    .metric-label {
        font-size: 14px;
        color: #8892b0;
    }
</style>
""", unsafe_allow_html=True)

# ─── HÀM LOAD ARTIFACTS MÔ HÌNH ───────────────────────────────────────
@st.cache_resource
def load_models_and_artifacts():
    try:
        clf_rf = joblib.load("random_forest_model.joblib")
        clf_svm = joblib.load("svm_model.joblib")
        scaler = joblib.load("scaler.joblib")
        
        # Load TensorFlow GRU Model
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        model_gru = load_model("improved_gru_model.keras")
        
        with open("feature_cols.json", "r") as f:
            feature_cols = json.load(f)
        return clf_rf, clf_svm, model_gru, scaler, feature_cols
    except Exception as e:
        st.error(f"❌ Không thể nạp mô hình! Hãy kiểm tra xem bạn đã paste đủ 5 file từ Colab chưa. Lỗi: {e}")
        return None, None, None, None, None

# Tải mô hình
clf_rf, clf_svm, model_gru, scaler, feature_cols = load_models_and_artifacts()

# Dừng chương trình nếu thiếu mô hình
if clf_rf is None or model_gru is None:
    st.title("🪙 Đồ Án Tốt Nghiệp - Hệ Thống Dự Báo Xu Hướng BTC")
    st.error("⚠️ Hệ thống không tìm thấy đầy đủ các file mô hình học máy cục bộ trong thư mục Project!")
    st.info("💡 Hãy đảm bảo bạn đã tải đủ 5 file từ Google Colab về thư mục 'd:\\Đồ án tốt nghiệp\\Project\\' rồi F5 chạy lại.")
    st.stop()

# ─── TIÊU ĐỀ GIAO DIỆN ────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: #f39c12; margin-bottom: 5px;'>🪙 Đồ Án Tốt Nghiệp - Hội Đồng Dự Báo BTC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0; font-size: 16px; margin-bottom: 25px;'>Giao diện nạp dữ liệu độc lập (X_test / y_test) và so sánh trực quan hiệu năng 3 mô hình</p>", unsafe_allow_html=True)

# ─── BẢNG ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR) VỚI 2 NƠI TẢI FILE ──────────────
st.sidebar.markdown("<h2 style='color: #f39c12;'>⚙️ Bảng Điều Khiển</h2>", unsafe_allow_html=True)
st.sidebar.info("Vui lòng tải lên đồng thời cả 2 file X_test (chứa features) và y_test (chứa target) để chạy mô phỏng.")

st.sidebar.markdown("### 📥 1. Tải lên file X_test (Không chứa nhãn)")
file_x = st.sidebar.file_uploader(
    "Chọn file CSV của X_test:",
    type=["csv"],
    key="uploader_x"
)

st.sidebar.markdown("### 📥 2. Tải lên file y_test (Chỉ chứa nhãn)")
file_y = st.sidebar.file_uploader(
    "Chọn file CSV của y_test:",
    type=["csv"],
    key="uploader_y"
)

# ─── XỬ LÝ DỮ LIỆU KHI CẢ 2 FILE ĐỀU ĐƯỢC TẢI LÊN ──────────────────────
if file_x is not None and file_y is not None:
    # Đọc dữ liệu
    df_x = pd.read_csv(file_x)
    df_y = pd.read_csv(file_y)
    
    # Kiểm tra tính tương thích về số dòng dữ liệu đầu vào
    if len(df_x) != len(df_y):
        st.error(f"❌ Lỗi: Số lượng dòng của X_test ({len(df_x):,}) và y_test ({len(df_y):,}) không khớp nhau!")
        st.stop()
        
    st.success(f"🎉 Tải file thành công! Tìm thấy {len(df_x):,} dòng dữ liệu kiểm thử.")
    
    # Tự động tính hl_range nếu thiếu
    if 'hl_range' not in df_x.columns:
        df_x['hl_range'] = (df_x['high'] - df_x['low']) / df_x['close']
        
    # Kiểm tra xem X_test có đủ đặc trưng không
    missing_cols = [col for col in feature_cols if col not in df_x.columns]
    if missing_cols:
        st.error(f"❌ File X_test thiếu các cột đặc trưng bắt buộc: {missing_cols}")
        st.stop()
        
    # ─── PHÂN TÍCH VÀ CHIA CỬA SỔ TRƯỢT (TIMESTEP = 16) ─────────────────
    TIMESTEP = 16
    
    X_raw = df_x[feature_cols].values
    
    # Đọc cột target từ file y_test (cột đầu tiên)
    y_raw = df_y.iloc[:, 0].values.astype(int)
    
    # Chuẩn hóa
    X_scaled = scaler.transform(X_raw)
    
    X_window = []
    y_window = []
    
    for i in range(len(X_scaled) - TIMESTEP):
        X_window.append(X_scaled[i : i + TIMESTEP])
        y_window.append(y_raw[i + TIMESTEP - 1])
        
    X_window = np.array(X_window)
    y_window = np.array(y_window)
    
    # Làm phẳng dữ liệu cho RF/SVM
    X_flat = X_window.reshape(X_window.shape[0], -1)
    
    # ─── CHẠY DỰ BÁO TỪ 3 MÔ HÌNH ───────────────────────────────────────
    with st.spinner("⚡ Đang gọi các mô hình chạy suy luận dự báo..."):
        # GRU Predict
        y_pred_gru_prob = model_gru.predict(X_window)
        y_pred_gru = np.argmax(y_pred_gru_prob, axis=1)
        
        # Random Forest Predict
        y_pred_rf = clf_rf.predict(X_flat)
        
        # SVM Predict
        y_pred_svm = clf_svm.predict(X_flat)
        
    # Tính điểm số
    acc_gru = accuracy_score(y_window, y_pred_gru)
    acc_rf  = accuracy_score(y_window, y_pred_rf)
    acc_svm = accuracy_score(y_window, y_pred_svm)
    
    f1_gru = f1_score(y_window, y_pred_gru, average='weighted')
    f1_rf  = f1_score(y_window, y_pred_rf, average='weighted')
    f1_svm = f1_score(y_window, y_pred_svm, average='weighted')
    
    # ─── PHÂN KHU 1: SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY COMPARISON CHART) ───
    st.markdown("## 📊 1. Biểu Đồ So Sánh Độ Chính Xác (Accuracy & F1-Score)")
    
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=["Mô hình GRU", "Random Forest", "SVM RBF"],
        y=[acc_gru * 100, acc_rf * 100, acc_svm * 100],
        text=[f"{acc_gru*100:.2f}%", f"{acc_rf*100:.2f}%", f"{acc_svm*100:.2f}%"],
        textposition='auto',
        marker_color=['#3498db', '#2ecc71', '#e74c3c']
    ))
    fig_acc.update_layout(
        title="So Sánh Chỉ Số Độ Chính Xác (Accuracy %)",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#8892b0")
    )
    fig_acc.update_yaxes(title_text="Độ chính xác (%)", range=[0, 100], gridcolor='#2e3440')
    
    col_chart, col_metric = st.columns([2, 1])
    with col_chart:
        st.plotly_chart(fig_acc, use_container_width=True)
    with col_metric:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧠 GRU (Accuracy)</div>
            <div class="metric-value">{acc_gru*100:.2f}%</div>
        </div>
        <div class="metric-card" style="margin-top:10px;">
            <div class="metric-label">🌲 Random Forest (Accuracy)</div>
            <div class="metric-value">{acc_rf*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── PHÂN KHU 2: PHÂN PHỐI DỰ BÁO TỪNG MÔ HÌNH (PIE CHARTS) ──────────
    st.markdown("## 🍕 2. Phân Phối Kết Quả Dự Báo Của Từng Mô Hình")
    
    # Tính số lượng phân phối nhãn
    unique_vals = [0, 1, 2]
    labels_text = ["Giảm (0)", "Đi ngang (1)", "Tăng (2)"]
    colors_pie = ["#e74c3c", "#95a5a6", "#2ecc71"]
    
    counts_gru = [np.sum(y_pred_gru == val) for val in unique_vals]
    counts_rf  = [np.sum(y_pred_rf == val) for val in unique_vals]
    counts_svm = [np.sum(y_pred_svm == val) for val in unique_vals]
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        fig_p1 = go.Figure(data=[go.Pie(labels=labels_text, values=counts_gru, hole=.3, marker=dict(colors=colors_pie))])
        fig_p1.update_layout(title="Dự đoán GRU (Tỷ lệ %)", height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#8892b0"), showlegend=False)
        st.plotly_chart(fig_p1, use_container_width=True)
        
    with col_p2:
        fig_p2 = go.Figure(data=[go.Pie(labels=labels_text, values=counts_rf, hole=.3, marker=dict(colors=colors_pie))])
        fig_p2.update_layout(title="Dự đoán Random Forest", height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#8892b0"), showlegend=False)
        st.plotly_chart(fig_p2, use_container_width=True)
        
    with col_p3:
        fig_p3 = go.Figure(data=[go.Pie(labels=labels_text, values=counts_svm, hole=.3, marker=dict(colors=colors_pie))])
        fig_p3.update_layout(title="Dự đoán SVM RBF", height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#8892b0"), showlegend=False)
        st.plotly_chart(fig_p3, use_container_width=True)

    # ─── PHÂN KHU 3: SƠ ĐỒ SO SÁNH KẾT QUẢ DỰ BÁO (GROUPED BAR CHART) ───
    st.markdown("## 📈 3. So Sánh Dự Đoán Của Cả 3 Mô Hình vs Thực Tế (y_test)")
    
    counts_actual = [np.sum(y_window == val) for val in unique_vals]
    
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(x=labels_text, y=counts_actual, name='Thực tế (y_test)', marker_color='#95a5a6'))
    fig_compare.add_trace(go.Bar(x=labels_text, y=counts_gru, name='Dự đoán GRU', marker_color='#3498db'))
    fig_compare.add_trace(go.Bar(x=labels_text, y=counts_rf, name='Dự đoán Random Forest', marker_color='#2ecc71'))
    fig_compare.add_trace(go.Bar(x=labels_text, y=counts_svm, name='Dự đoán SVM RBF', marker_color='#e74c3c'))
    
    fig_compare.update_layout(
        barmode='group',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#8892b0")
    )
    fig_compare.update_yaxes(title_text="Số lượng phiên dự đoán", gridcolor='#2e3440')
    fig_compare.update_xaxes(gridcolor='#2e3440')
    
    st.plotly_chart(fig_compare, use_container_width=True)

    # ─── PHÂN KHU 4: BẢNG CHI TIẾT ──────────────────────────────────────
    with st.expander("📝 Xem bảng chi tiết kết quả đối chiếu từng phiên"):
        LABEL_MAP_INV = {0: "Giảm (0)", 1: "Đi ngang (1)", 2: "Tăng (2)"}
        
        comparison_df = pd.DataFrame({
            "Phiên thứ (Index)": np.arange(len(y_window)),
            "Nhãn thực tế (y_test)": [LABEL_MAP_INV[val] for val in y_window],
            "Dự đoán GRU": [LABEL_MAP_INV[val] for val in y_pred_gru],
            "Dự đoán Random Forest": [LABEL_MAP_INV[val] for val in y_pred_rf],
            "Dự đoán SVM": [LABEL_MAP_INV[val] for val in y_pred_svm]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ─── GIAO DIỆN CHỜ TẢI FILE ───────────────────────────────────────────
else:
    st.info("👈 Vui lòng tải lên cả 2 file X_test.csv và y_test.csv từ thanh điều khiển bên trái để bắt đầu mô phỏng dự báo!")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        ### Cấu trúc file X_test (.csv) mẫu:
        Chứa dữ liệu các đặc trưng giá & cảm xúc (KHÔNG có cột nhãn):
        *   `open, high, low, close, volume`
        *   `tweet_count, vader_influence, bert_influence`
        *   `MA_16_ratio, return_30m_lag1, return_30m_lag2, oc_change, rolling_volatility`
        """)
    with col_info2:
        st.markdown("""
        ### Cấu trúc file y_test (.csv) mẫu:
        Chỉ chứa duy nhất 1 cột chứa nhãn xu hướng thực tế của phiên kế tiếp:
        *   Cột nhãn: `trend_label` (hoặc cột đầu tiên)
        *   Giá trị: `0` (Giảm), `1` (Đi ngang), `2` (Tăng)
        """)