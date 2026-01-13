"""
Analytics Integration Module
Copy-paste code ini ke app.py Anda untuk menambahkan Learning Analytics Dashboard

INSTRUKSI INTEGRASI:
====================

1. Import modules di bagian atas app.py:
   ---
   from analytics_engine import AnalyticsEngine
   from analytics_dashboard import (
       create_metrics_summary_cards,
       plot_accuracy_trend_comparison,
       plot_mastery_heatmap,
       plot_response_time_distribution,
       plot_emotion_distribution,
       plot_learning_velocity,
       plot_progress_radar,
       create_insights_box
   )
   ---

2. Update init_state() function untuk menambahkan page "analytics":
   ---
   ss.setdefault("page", "start")  # existing
   # Tidak perlu tambahan, analytics bisa diakses dari page mana saja
   ---

3. Di student_interface(), tambahkan button di sidebar:
   ---
   # Di bagian sidebar (setelah navigation menu)
   st.sidebar.markdown("---")
   if st.sidebar.button("📊 Lihat Analytics", use_container_width=True):
       st.session_state.page = "analytics"
       st.rerun()
   ---

4. Di student_interface(), tambahkan elif untuk page analytics:
   ---
   elif st.session_state.page == "analytics":
       show_analytics_dashboard()
   ---

5. Tambahkan fungsi show_analytics_dashboard() di bawah:
"""

import streamlit as st
import os
from datetime import datetime

# Import modules (pastikan sudah ada di imports app.py)
from analytics_engine import AnalyticsEngine
from analytics_dashboard import (
    create_metrics_summary_cards,
    plot_accuracy_trend_comparison,
    plot_mastery_heatmap,
    plot_response_time_distribution,
    plot_emotion_distribution,
    plot_learning_velocity,
    plot_progress_radar,
    create_insights_box
)


def show_analytics_dashboard():
    """
    Main function untuk analytics dashboard page
    Tambahkan function ini ke app.py Anda
    """
    
    # Check if user is logged in
    if not st.session_state.get("user_id"):
        st.warning("⚠️ Silakan login terlebih dahulu untuk melihat analytics.")
        if st.button("Kembali ke Login"):
            st.session_state.page = "start"
            st.rerun()
        return
    
    user_id = st.session_state.user_id
    
    # Path to events.csv (adjust based on your structure)
    # Jika struktur Anda: webapp/data/events.csv
    events_path = os.path.join(os.path.dirname(__file__), "data", "events.csv")
    
    # Initialize analytics engine
    engine = AnalyticsEngine(events_path)
    
    # Load data
    if not engine.load_data():
        st.error("❌ Tidak dapat memuat data analytics. File events.csv mungkin tidak ada atau kosong.")
        if st.button("Kembali"):
            st.session_state.page = "material"
            st.rerun()
        return
    
    # Check if user has data
    user_df = engine.get_user_data(user_id)
    if user_df.empty:
        st.info("📊 Belum ada data analytics untuk Anda. Mulai mengerjakan soal untuk melihat progress!")
        if st.button("Mulai Latihan"):
            st.session_state.page = "material"
            st.rerun()
        return
    
    # Get date range of user data
    min_date = user_df['ts'].min()
    max_date = user_df['ts'].max()
    days_available = (max_date - min_date).days
    
    # ===== HEADER =====
    st.title(f"📊 Learning Analytics Dashboard")
    st.markdown(f"### Halo, **{user_id}**! 👋")
    st.caption(f"Data tersedia: {min_date.strftime('%d %b %Y')} - {max_date.strftime('%d %b %Y')} ({days_available} hari)")
    
    st.markdown("---")
    
    # ===== TIME PERIOD SELECTOR =====
    st.subheader("🕐 Pengaturan Perbandingan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Slider untuk comparison days
        max_comparison = min(30, days_available)
        
        if max_comparison < 2:
            st.warning("⚠️ Data Anda belum cukup untuk perbandingan. Kembali lagi setelah beberapa hari!")
            comparison_days = 1
        else:
            comparison_days = st.slider(
                "Compare performa dengan berapa hari yang lalu?",
                min_value=1,
                max_value=max_comparison,
                value=min(7, max_comparison),
                help="Pilih berapa hari ke belakang untuk membandingkan performa"
            )
    
    with col2:
        window_size = st.selectbox(
            "Ukuran window periode",
            options=[3, 7, 14],
            index=1,
            help="Berapa hari data per periode yang dibandingkan"
        )
    
    # Info boxes
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        period_a_end = max_date.strftime('%d %b')
        period_a_start = (max_date - pd.Timedelta(days=window_size)).strftime('%d %b')
        st.info(f"📅 **Periode Sekarang:** {period_a_start} - {period_a_end}")
    
    with col_info2:
        period_b_end = (max_date - pd.Timedelta(days=comparison_days)).strftime('%d %b')
        period_b_start = (max_date - pd.Timedelta(days=comparison_days + window_size)).strftime('%d %b')
        st.success(f"📅 **Periode Pembanding:** {period_b_start} - {period_b_end}")
    
    st.markdown("---")
    
    # ===== COMPUTE METRICS =====
    with st.spinner("⏳ Menganalisis data..."):
        metrics_current, metrics_past, delta = engine.compare_periods(
            user_id=user_id,
            comparison_days=comparison_days,
            window_size=window_size
        )
        
        daily_trends = engine.get_daily_trends(user_id, days_back=30)
        
        insights = engine.generate_insights(metrics_current, metrics_past, delta)
    
    # ===== SUMMARY METRICS CARDS =====
    st.subheader("📈 Ringkasan Performa")
    st.markdown(
        create_metrics_summary_cards(metrics_current, metrics_past, delta),
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # ===== AUTO INSIGHTS =====
    st.markdown(create_insights_box(insights), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== VISUALIZATIONS =====
    st.subheader("📊 Visualisasi Detail")
    
    # Tabs untuk organize charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tren & Progress",
        "🎯 Penguasaan Konsep",
        "⏱️ Time & Velocity",
        "😊 Emosi & Overall"
    ])
    
    with tab1:
        st.plotly_chart(
            plot_accuracy_trend_comparison(metrics_current, metrics_past, daily_trends),
            use_container_width=True
        )
        
        st.markdown("##### 📊 Statistik Detail")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Soal Dikerjakan",
                metrics_current.get('total_questions', 0),
                f"{delta.get('total_questions', {}).get('value', 0):.0f}"
            )
            st.metric(
                "Hari Aktif",
                metrics_current.get('active_days', 0),
                f"{delta.get('active_days', {}).get('value', 0):.0f}"
            )
        with col2:
            st.metric(
                "Soal Benar",
                metrics_current.get('correct_count', 0),
                f"{delta.get('correct_count', {}).get('value', 0):.0f}"
            )
            st.metric(
                "Soal Salah",
                metrics_current.get('wrong_count', 0),
                f"{delta.get('wrong_count', {}).get('value', 0):.0f}"
            )
    
    with tab2:
        st.plotly_chart(
            plot_mastery_heatmap(metrics_current, metrics_past, delta),
            use_container_width=True
        )
        
        # Detail mastery table
        if metrics_current.get('mastery_per_concept'):
            st.markdown("##### 📋 Tabel Detail Penguasaan")
            mastery_data = []
            for concept, data in metrics_current['mastery_per_concept'].items():
                past_data = metrics_past.get('mastery_per_concept', {}).get(concept, {})
                delta_val = data['accuracy'] - past_data.get('accuracy', 0)
                
                mastery_data.append({
                    'Konsep': concept,
                    'Akurasi Sekarang': f"{data['accuracy']:.1f}%",
                    'Akurasi Sebelumnya': f"{past_data.get('accuracy', 0):.1f}%",
                    'Perubahan': f"{'↑' if delta_val > 0 else '↓'} {abs(delta_val):.1f}%",
                    'Total Soal': data['count'],
                    'Benar': data['correct']
                })
            
            st.dataframe(pd.DataFrame(mastery_data), use_container_width=True)
    
    with tab3:
        st.plotly_chart(
            plot_response_time_distribution(metrics_current, metrics_past),
            use_container_width=True
        )
        
        st.plotly_chart(
            plot_learning_velocity(metrics_current, metrics_past, delta),
            use_container_width=True
        )
        
        # Additional time metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Waktu Belajar",
                f"{metrics_current.get('total_time_spent', 0):.0f}s",
                f"{delta.get('total_time_spent', {}).get('value', 0):.0f}s"
            )
        with col2:
            st.metric(
                "Avg Soal per Hari",
                f"{metrics_current.get('avg_questions_per_day', 0):.1f}",
                f"{delta.get('avg_questions_per_day', {}).get('value', 0):.1f}"
            )
    
    with tab4:
        st.plotly_chart(
            plot_emotion_distribution(metrics_current, metrics_past),
            use_container_width=True
        )
        
        st.plotly_chart(
            plot_progress_radar(metrics_current, metrics_past),
            use_container_width=True
        )
        
        # Emotion summary
        st.markdown("##### 😊 Ringkasan Kondisi Emosional")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Sekarang:** {metrics_current.get('dominant_emotion', 'N/A')}")
        with col2:
            st.info(f"**Sebelumnya:** {metrics_past.get('dominant_emotion', 'N/A')}")
    
    st.markdown("---")
    
    # ===== NAVIGATION BUTTONS =====
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Kembali ke Materi", use_container_width=True):
            st.session_state.page = "material"
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col3:
        # Export feature (future enhancement)
        st.button("📄 Export Report (Coming Soon)", disabled=True, use_container_width=True)


# ===== SIDEBAR BUTTON CODE =====
# Paste ini di bagian sidebar app.py Anda, dalam student_interface():
"""
def add_analytics_sidebar_button():
    '''
    Tambahkan ini di sidebar student_interface()
    '''
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Analytics")
    
    if st.sidebar.button("📈 Lihat Dashboard", use_container_width=True, key="analytics_btn"):
        st.session_state.page = "analytics"
        st.rerun()
    
    st.sidebar.caption("Track your learning progress!")
"""


# ===== UPDATE student_interface() =====
# Tambahkan elif ini di routing page student_interface():
"""
    elif st.session_state.page == "analytics":
        show_analytics_dashboard()
"""


if __name__ == "__main__":
    st.info("""
    📋 **Instruksi Integrasi:**
    
    1. Copy `analytics_engine.py` ke folder `webapp/`
    2. Copy `analytics_dashboard.py` ke folder `webapp/`
    3. Copy function `show_analytics_dashboard()` dari file ini ke `app.py`
    4. Tambahkan imports di bagian atas `app.py`
    5. Tambahkan sidebar button di `student_interface()`
    6. Tambahkan `elif st.session_state.page == "analytics"` di routing
    7. Done! ✅
    
    Test dengan: `streamlit run app.py`
    """)