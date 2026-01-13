"""
Session-Based Analytics Dashboard
Replace or add this function to your app.py

This version compares sessions instead of days, so users can see
progress even if they do multiple quizzes on the same day.
"""

import streamlit as st
import pandas as pd
import os
from analytics_engine import AnalyticsEngine
from analytics_dashboard import (
    plot_mastery_heatmap,
    plot_response_time_distribution,
    plot_emotion_distribution,
    plot_progress_radar
)

# Buat fungsi pengganti sederhana
def create_metrics_summary_cards(metrics_current, metrics_past, delta):
    """Simple replacement - returns empty string"""
    return ""

def create_insights_box(insights):
    """Simple replacement - returns insights as bullet points"""
    return "<ul>" + "".join([f"<li>{i}</li>" for i in insights]) + "</ul>"

def student_analytics_dashboard_session_based(user_id: str):
    """
    Session-based analytics dashboard untuk student
    Compare sesi terakhir vs sesi ke-N sebelumnya
    
    Args:
        user_id: Student ID yang sedang login
    """
    
    # Path to events.csv
    events_path = os.path.join(os.path.dirname(__file__), "data", "events.csv")
    
    # Initialize engine
    engine = AnalyticsEngine(events_path)
    
    # Load data
    if not engine.load_data():
        st.error("❌ Tidak dapat memuat data analytics. File events.csv mungkin tidak ada.")
        if st.button("Kembali"):
            st.session_state.page = "material"
            st.rerun()
        return
    
    # Get user sessions
    sessions = engine.get_user_sessions(user_id)
    
    if len(sessions) == 0:
        st.info("📊 Belum ada sesi latihan yang tercatat. Selesaikan minimal 1 quiz untuk melihat analytics!")
        if st.button("Mulai Latihan"):
            st.session_state.page = "material"
            st.rerun()
        return
    
    if len(sessions) < 2:
        st.warning("⏳ Baru ada 1 sesi latihan. Selesaikan 1 sesi lagi untuk melihat perbandingan progress!")
        
        # Show current session stats
        st.subheader("📊 Statistik Sesi Terakhir")
        current_session = sessions[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sesi #", current_session['session_number'])
        col2.metric("Level", int(current_session['level']))
        col3.metric("Akurasi", f"{current_session['accuracy']:.1f}%")
        col4.metric("Durasi", f"{current_session['duration']:.0f}s")
        
        st.info(f"⏰ Waktu: {current_session['start_ts'].strftime('%d %b %H:%M')} - {current_session['end_ts'].strftime('%H:%M')}")
        
        if st.button("Mulai Latihan Berikutnya"):
            st.session_state.page = "material"
            st.rerun()
        return
    
    # ===== HEADER =====
    st.title("📊 Learning Analytics Dashboard")
    st.markdown(f"### Halo, **{user_id}**! 👋")
    st.caption(f"Total sesi latihan: {len(sessions)}")
    
    st.markdown("---")
    
    # ===== SESSION SELECTOR =====
    st.subheader("🕐 Pengaturan Perbandingan")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Slider untuk session comparison
        max_sessions_back = min(30, len(sessions) - 1)
        
        sessions_back = st.slider(
            "Bandingkan sesi terakhir dengan sesi ke-N sebelumnya",
            min_value=1,
            max_value=max_sessions_back,
            value=1,
            help="Pilih berapa sesi ke belakang untuk membandingkan progress"
        )
    
    with col2:
        st.metric("Total Sesi", len(sessions))
    
    # Calculate which sessions being compared
    current_session_num = sessions[-1]['session_number']
    past_session_num = sessions[-(sessions_back + 1)]['session_number']
    
    # Info boxes
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        current = sessions[-1]
        st.info(f"""
        **📅 Sesi #{current_session_num} (Terakhir)**  
        ⏰ {current['start_ts'].strftime('%d %b %H:%M')} - {current['end_ts'].strftime('%H:%M')}  
        📊 Level {int(current['level'])}
        """)
    
    with col_info2:
        past = sessions[-(sessions_back + 1)]
        st.success(f"""
        **📅 Sesi #{past_session_num} (Pembanding)**  
        ⏰ {past['start_ts'].strftime('%d %b %H:%M')} - {past['end_ts'].strftime('%H:%M')}  
        📊 Level {int(past['level'])}
        """)
    
    st.markdown("---")
    
    # ===== COMPUTE METRICS =====
    with st.spinner("⏳ Menganalisis data..."):
        metrics_current, metrics_past, delta = engine.compare_sessions(
            user_id=user_id,
            sessions_back=sessions_back
        )
        
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
    
    # ===== SESSION HISTORY TABLE =====
    with st.expander("📋 Lihat Riwayat Semua Sesi", expanded=False):
        session_history = []
        for s in sessions:
            session_history.append({
                'Sesi #': s['session_number'],
                'Tanggal': s['start_ts'].strftime('%d %b %Y'),
                'Waktu': f"{s['start_ts'].strftime('%H:%M')} - {s['end_ts'].strftime('%H:%M')}",
                'Level': int(s['level']),
                'Akurasi': f"{s['accuracy']:.1f}%",
                'Durasi': f"{s['duration']:.0f}s",
                'Emosi': s['emotion_state'],
                'Struggle': s['struggle_pred']
            })
        
        st.dataframe(
            pd.DataFrame(session_history),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # ===== VISUALIZATIONS =====
    st.subheader("📊 Visualisasi Detail")
    
    # Tabs untuk organize charts
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tren & Progress",
        "🎯 Penguasaan Konsep",
        "⏱️ Time & Performance",
        "😊 Emosi & Overall"
    ])
    
    with tab1:
        # Session progression chart
        st.markdown("##### 📈 Progression Across Sessions")
        
        # Build session progression data
        session_prog = pd.DataFrame([{
            'Sesi': s['session_number'],
            'Akurasi': s['accuracy'],
            'Level': s['level']
        } for s in sessions])
        
        import plotly.graph_objects as go
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=session_prog['Sesi'],
            y=session_prog['Akurasi'],
            mode='lines+markers',
            name='Akurasi',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        # Highlight compared sessions
        fig.add_trace(go.Scatter(
            x=[past_session_num, current_session_num],
            y=[metrics_past.get('accuracy', 0), metrics_current.get('accuracy', 0)],
            mode='markers',
            name='Sesi Dibandingkan',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title='Akurasi per Sesi',
            xaxis_title='Nomor Sesi',
            yaxis_title='Akurasi (%)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("##### 📊 Statistik Detail")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Soal (Sesi Ini)",
                metrics_current.get('total_questions', 0),
                f"{delta.get('total_questions', {}).get('value', 0):.0f}"
            )
            st.metric(
                "Durasi Sesi",
                f"{metrics_current.get('session_duration', 0):.0f}s",
                f"{delta.get('session_duration', {}).get('value', 0):.0f}s" 
                if 'session_duration' in delta else None
            )
        with col2:
            st.metric(
                "Soal Benar",
                metrics_current.get('correct_count', 0),
                f"{delta.get('correct_count', {}).get('value', 0):.0f}" 
                if 'correct_count' in delta else None
            )
            st.metric(
                "Soal Salah",
                metrics_current.get('wrong_count', 0),
                f"{delta.get('wrong_count', {}).get('value', 0):.0f}"
                if 'wrong_count' in delta else None
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
            
            st.dataframe(pd.DataFrame(mastery_data), use_container_width=True, hide_index=True)
    
    with tab3:
        st.plotly_chart(
            plot_response_time_distribution(metrics_current, metrics_past),
            use_container_width=True
        )
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Avg Response Time",
                f"{metrics_current.get('avg_response_time', 0):.1f}s",
                f"{delta.get('avg_response_time', {}).get('value', 0):.1f}s"
            )
        with col2:
            st.metric(
                "Median Time",
                f"{metrics_current.get('median_response_time', 0):.1f}s"
            )
        with col3:
            st.metric(
                "Max Time",
                f"{metrics_current.get('max_response_time', 0):.1f}s"
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
        st.markdown("##### 😊 Kondisi Emosional")
        col1, col2 = st.columns(2)
        with col1:
            emotion_now = metrics_current.get('dominant_emotion', 'N/A')
            emoji = {"CONFIDENT": "😎", "CONFUSED": "😕", "FRUSTRATED": "😤", 
                     "ANXIOUS": "😰", "NEUTRAL": "😐"}.get(emotion_now, "😊")
            st.info(f"{emoji} **Sesi Sekarang:** {emotion_now}")
        with col2:
            emotion_past = metrics_past.get('dominant_emotion', 'N/A')
            emoji = {"CONFIDENT": "😎", "CONFUSED": "😕", "FRUSTRATED": "😤", 
                     "ANXIOUS": "😰", "NEUTRAL": "😐"}.get(emotion_past, "😊")
            st.info(f"{emoji} **Sesi Sebelumnya:** {emotion_past}")
    
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
        if st.button("📝 Latihan Lagi", use_container_width=True):
            st.session_state.page = "material"
            st.rerun()


# ===== USAGE IN app.py =====
"""
Replace your existing analytics function call with:

elif st.session_state.page == "analytics":
    student_analytics_dashboard_session_based(st.session_state.user_id)

OR keep both options and let user choose:

elif st.session_state.page == "analytics":
    mode = st.sidebar.radio("Mode", ["Per Sesi", "Per Hari"])
    if mode == "Per Sesi":
        student_analytics_dashboard_session_based(st.session_state.user_id)
    else:
        student_analytics_dashboard(st.session_state.user_id)  # old function
"""