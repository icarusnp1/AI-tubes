"""
Student Learning Analytics Dashboard
Provides detailed performance insights with time-based and session-based comparison
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================
# EXISTING FUNCTIONS (TIME-BASED ANALYTICS)
# ============================================

def load_user_events(user_id: str, events_csv: str) -> pd.DataFrame:
    """Load and filter events for specific user"""
    if not os.path.exists(events_csv):
        return pd.DataFrame()
    
    df = pd.read_csv(events_csv)
    df['ts'] = pd.to_datetime(df['ts'])
    df_user = df[df['user_id'] == user_id].copy()
    df_user = df_user.sort_values('ts')
    
    return df_user


def get_time_range_data(df: pd.DataFrame, days_ago: int) -> tuple:
    """
    Split data into current period and comparison period
    Returns: (current_df, comparison_df)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    latest_ts = df['ts'].max()
    cutoff_date = latest_ts - timedelta(days=days_ago)
    
    # Current period: data setelah cutoff
    current_df = df[df['ts'] > cutoff_date].copy()
    
    # Comparison period: data sebelum cutoff (dengan durasi yang sama)
    comparison_start = cutoff_date - timedelta(days=days_ago)
    comparison_df = df[(df['ts'] >= comparison_start) & (df['ts'] <= cutoff_date)].copy()
    
    return current_df, comparison_df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate key performance metrics from events"""
    if df.empty:
        return {
            "total_questions": 0,
            "avg_accuracy": 0.0,
            "total_time_minutes": 0.0,
            "avg_response_time": 0.0,
            "struggle_rate": 0.0,
            "hint_usage_rate": 0.0,
            "current_level": 1,
            "sessions_completed": 0,
            "concepts_practiced": 0,
            "confident_rate": 0.0,
            "frustrated_rate": 0.0
        }
    
    # Filter relevant events
    answers = df[df['event'] == 'SUBMIT_ANSWER']
    set_results = df[df['event'] == 'SET_RESULT']
    
    metrics = {
        "total_questions": len(answers),
        "avg_accuracy": answers['is_correct'].mean() if len(answers) > 0 else 0.0,
        "total_time_minutes": answers['response_time'].sum() / 60.0 if len(answers) > 0 else 0.0,
        "avg_response_time": answers['response_time'].mean() if len(answers) > 0 else 0.0,
        "struggle_rate": (set_results['struggle_pred'] == 'STRUGGLE').mean() if len(set_results) > 0 else 0.0,
        "hint_usage_rate": answers['used_hint'].mean() if len(answers) > 0 else 0.0,
        "current_level": df['level'].iloc[-1] if len(df) > 0 else 1,
        "sessions_completed": len(set_results),
        "concepts_practiced": answers['concept_tag'].nunique() if len(answers) > 0 else 0,
        "confident_rate": (set_results['emotion_state'] == 'CONFIDENT').mean() if len(set_results) > 0 else 0.0,
        "frustrated_rate": (set_results['emotion_state'] == 'FRUSTRATED').mean() if len(set_results) > 0 else 0.0
    }
    
    return metrics


def create_comparison_metrics_cards(current: dict, comparison: dict) -> dict:
    """
    Create comparison data for metric cards
    Returns dict with metric_name: (current_value, change_percent, is_improved)
    """
    cards = {}
    
    # Define metrics with improvement direction (True = higher is better)
    metric_configs = {
        "Akurasi": ("avg_accuracy", True, "%"),
        "Total Soal": ("total_questions", True, ""),
        "Waktu Belajar": ("total_time_minutes", True, " mnt"),
        "Sesi Selesai": ("sessions_completed", True, ""),
        "Konsep Dipraktikkan": ("concepts_practiced", True, ""),
        "Rate Percaya Diri": ("confident_rate", True, "%"),
        "Rate Kesulitan": ("struggle_rate", False, "%"),
        "Penggunaan Hint": ("hint_usage_rate", False, "%")
    }
    
    for display_name, (key, higher_is_better, unit) in metric_configs.items():
        curr_val = current.get(key, 0)
        comp_val = comparison.get(key, 0)
        
        # Calculate change
        if comp_val == 0:
            change_pct = 0.0 if curr_val == 0 else 100.0
        else:
            change_pct = ((curr_val - comp_val) / comp_val) * 100
        
        # Determine if improved
        if higher_is_better:
            is_improved = change_pct > 0
        else:
            is_improved = change_pct < 0
        
        # Format value based on unit
        if unit == "%":
            display_val = f"{curr_val * 100:.1f}%"
        elif unit == " mnt":
            display_val = f"{curr_val:.1f}{unit}"
        else:
            display_val = f"{curr_val:.0f}{unit}"
        
        cards[display_name] = {
            "value": display_val,
            "change": change_pct,
            "improved": is_improved,
            "raw_current": curr_val,
            "raw_comparison": comp_val
        }
    
    return cards


def create_accuracy_trend_chart(df: pd.DataFrame, days_ago: int) -> go.Figure:
    """Create line chart showing accuracy trend over time"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Belum ada data", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    set_results = df[df['event'] == 'SET_RESULT'].copy()
    
    if set_results.empty:
        fig = go.Figure()
        fig.add_annotation(text="Belum ada set soal yang diselesaikan", 
                          xref="paper", yref="paper", x=0.5, y=0.5, 
                          showarrow=False, font=dict(size=16))
        return fig
    
    set_results['date'] = set_results['ts'].dt.date
    
    # Daily aggregation
    daily_acc = set_results.groupby('date')['accuracy'].mean().reset_index()
    daily_acc['accuracy_pct'] = daily_acc['accuracy'] * 100
    
    # Create figure with moving average
    fig = go.Figure()
    
    # Actual accuracy
    fig.add_trace(go.Scatter(
        x=daily_acc['date'],
        y=daily_acc['accuracy_pct'],
        mode='lines+markers',
        name='Akurasi Harian',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8)
    ))
    
    # Moving average (3-day)
    if len(daily_acc) >= 3:
        daily_acc['ma3'] = daily_acc['accuracy_pct'].rolling(window=3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=daily_acc['date'],
            y=daily_acc['ma3'],
            mode='lines',
            name='Rata-rata 3 Hari',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
    
    # Target line
    fig.add_hline(y=80, line_dash="dot", line_color="orange", 
                  annotation_text="Target 80%", annotation_position="right")
    
    fig.update_layout(
        title=f"Tren Akurasi ({days_ago} Hari Terakhir)",
        xaxis_title="Tanggal",
        yaxis_title="Akurasi (%)",
        hovermode='x unified',
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    return fig


def create_concept_mastery_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap showing mastery level per concept"""
    answers = df[df['event'] == 'SUBMIT_ANSWER'].copy()
    
    if answers.empty or answers['concept_tag'].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="Belum ada data konsep", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    # Calculate accuracy per concept
    concept_stats = answers.groupby('concept_tag').agg(
        total=('is_correct', 'count'),
        correct=('is_correct', 'sum')
    ).reset_index()
    
    concept_stats['accuracy'] = (concept_stats['correct'] / concept_stats['total']) * 100
    concept_stats = concept_stats.sort_values('accuracy', ascending=False)
    
    # Color scale: red (low) -> yellow (medium) -> green (high)
    fig = go.Figure(data=go.Bar(
        y=concept_stats['concept_tag'],
        x=concept_stats['accuracy'],
        orientation='h',
        marker=dict(
            color=concept_stats['accuracy'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            colorbar=dict(title="Akurasi (%)")
        ),
        text=concept_stats['accuracy'].round(1),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Penguasaan per Konsep Matematika",
        xaxis_title="Akurasi (%)",
        yaxis_title="Konsep",
        height=max(400, len(concept_stats) * 40),
        xaxis=dict(range=[0, 105])
    )
    
    return fig


def create_emotion_distribution_chart(df: pd.DataFrame, days_ago: int) -> go.Figure:
    """Create pie chart showing emotion state distribution"""
    set_results = df[df['event'] == 'SET_RESULT'].copy()
    
    if set_results.empty or set_results['emotion_state'].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="Belum ada data emosi", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    emotion_counts = set_results['emotion_state'].value_counts()
    
    # Color mapping for emotions
    color_map = {
        'CONFIDENT': '#10b981',
        'NEUTRAL': '#6b7280',
        'CONFUSED': '#f59e0b',
        'FRUSTRATED': '#ef4444',
        'ANXIOUS': '#8b5cf6'
    }
    
    colors = [color_map.get(e, '#6b7280') for e in emotion_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=emotion_counts.index,
        values=emotion_counts.values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title=f"Distribusi Kondisi Emosional ({days_ago} Hari Terakhir)",
        height=400
    )
    
    return fig


def create_response_time_boxplot(df: pd.DataFrame) -> go.Figure:
    """Create boxplot showing response time distribution per level"""
    answers = df[df['event'] == 'SUBMIT_ANSWER'].copy()
    
    if answers.empty:
        fig = go.Figure()
        fig.add_annotation(text="Belum ada data response time", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    # Filter outliers (response time > 60 seconds)
    answers_filtered = answers[answers['response_time'] <= 60].copy()
    
    fig = go.Figure()
    
    for level in sorted(answers_filtered['level'].unique()):
        level_data = answers_filtered[answers_filtered['level'] == level]['response_time']
        
        fig.add_trace(go.Box(
            y=level_data,
            name=f'Level {level}',
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Distribusi Response Time per Level",
        yaxis_title="Response Time (detik)",
        xaxis_title="Level",
        height=400,
        showlegend=False
    )
    
    return fig


def create_learning_activity_timeline(df: pd.DataFrame, days_ago: int) -> go.Figure:
    """Create timeline showing learning activity intensity"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Belum ada aktivitas", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    df_copy = df.copy()
    df_copy['date'] = df_copy['ts'].dt.date
    df_copy['hour'] = df_copy['ts'].dt.hour
    
    # Count activities per day
    daily_activity = df_copy.groupby('date').size().reset_index(name='activity_count')
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_activity['date'],
        y=daily_activity['activity_count'],
        marker=dict(
            color=daily_activity['activity_count'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Aktivitas")
        ),
        text=daily_activity['activity_count'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Timeline Aktivitas Belajar ({days_ago} Hari Terakhir)",
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Aktivitas",
        height=400
    )
    
    return fig


def create_struggle_vs_confident_timeline(df: pd.DataFrame) -> go.Figure:
    """Create stacked area chart showing struggle vs confident over time"""
    set_results = df[df['event'] == 'SET_RESULT'].copy()
    
    if set_results.empty:
        fig = go.Figure()
        fig.add_annotation(text="Belum ada data sesi", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig
    
    set_results['date'] = set_results['ts'].dt.date
    
    # Count emotions per day
    daily_emotion = set_results.groupby(['date', 'emotion_state']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    # Add traces for each emotion
    if 'CONFIDENT' in daily_emotion.columns:
        fig.add_trace(go.Scatter(
            x=daily_emotion.index,
            y=daily_emotion['CONFIDENT'],
            mode='lines',
            name='Confident',
            fill='tonexty',
            line=dict(color='#10b981')
        ))
    
    if 'FRUSTRATED' in daily_emotion.columns:
        fig.add_trace(go.Scatter(
            x=daily_emotion.index,
            y=daily_emotion['FRUSTRATED'],
            mode='lines',
            name='Frustrated',
            fill='tonexty',
            line=dict(color='#ef4444')
        ))
    
    if 'CONFUSED' in daily_emotion.columns:
        fig.add_trace(go.Scatter(
            x=daily_emotion.index,
            y=daily_emotion['CONFUSED'],
            mode='lines',
            name='Confused',
            fill='tonexty',
            line=dict(color='#f59e0b')
        ))
    
    fig.update_layout(
        title="Perkembangan Kondisi Emosional",
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Sesi",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def generate_insights(current: dict, comparison: dict, cards: dict) -> list:
    """Generate AI-like insights based on metrics"""
    insights = []
    
    # Insight 1: Accuracy trend
    acc_card = cards.get("Akurasi", {})
    if acc_card.get("improved"):
        if acc_card.get("change", 0) > 10:
            insights.append(("✨", "Hebat!", f"Akurasi kamu meningkat {acc_card['change']:.1f}%! Pertahankan konsistensi ini."))
        else:
            insights.append(("👍", "Bagus!", f"Akurasi kamu meningkat {acc_card['change']:.1f}%. Terus tingkatkan!"))
    elif acc_card.get("change", 0) < -10:
        insights.append(("⚠️", "Perhatian", f"Akurasi turun {abs(acc_card['change']):.1f}%. Coba review konsep yang sulit."))
    
    # Insight 2: Activity level
    questions_card = cards.get("Total Soal", {})
    curr_q = questions_card.get("raw_current", 0)
    if curr_q > 30:
        insights.append(("🔥", "Produktif!", f"Kamu sudah mengerjakan {curr_q:.0f} soal. Luar biasa!"))
    elif curr_q < 10:
        insights.append(("📚", "Saran", "Coba tingkatkan frekuensi latihan untuk hasil optimal."))
    
    # Insight 3: Struggle rate
    struggle_card = cards.get("Rate Kesulitan", {})
    if struggle_card.get("raw_current", 0) > 0.5:
        insights.append(("💪", "Tantangan", "Kamu sering mengalami kesulitan. Jangan ragu pakai hint dan remedial!"))
    
    # Insight 4: Confident rate
    confident_card = cards.get("Rate Percaya Diri", {})
    if confident_card.get("raw_current", 0) > 0.7:
        insights.append(("🎯", "Konsisten", "Tingkat percaya diri tinggi! Kamu siap naik level."))
    
    # Insight 5: Hint usage
    hint_card = cards.get("Penggunaan Hint", {})
    if hint_card.get("raw_current", 0) > 0.6:
        insights.append(("💡", "Tips", "Kamu sering pakai hint. Coba pahami pola soal agar lebih mandiri."))
    
    return insights


# ============================================
# NEW FUNCTIONS (SESSION-BASED ANALYTICS)
# ============================================

def create_metrics_summary_cards(
    metrics_current: Dict,
    metrics_past: Dict,
    delta: Dict
) -> str:
    """
    Create HTML cards untuk ringkasan metrics (for session-based)
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics  
        delta: Delta metrics
        
    Returns:
        HTML string
    """
    def format_delta(key, decimals=1):
        """Helper untuk format delta dengan arrow dan color"""
        d = delta.get(key, {})
        value = d.get('value', 0)
        sentiment = d.get('sentiment', 'neutral')
        
        if abs(value) < 0.01:
            return '<span style="color: gray;">→ 0</span>'
        
        arrow = "↑" if d.get('direction') == 'up' else "↓" if d.get('direction') == 'down' else "→"
        color = "green" if sentiment == "positive" else ("red" if sentiment == "negative" else "gray")
        
        return f'<span style="color: {color};">{arrow} {abs(value):.{decimals}f}</span>'
    
    html = f"""
    <style>
        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1f77b4;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-compare {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}
    </style>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
        <div class="metric-card">
            <div class="metric-label">📊 Akurasi</div>
            <div class="metric-value">{metrics_current.get('accuracy', 0):.1f}%</div>
            <div class="metric-compare">
                Sebelumnya: {metrics_past.get('accuracy', 0):.1f}% {format_delta('accuracy')}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">⏱️ Avg Response Time</div>
            <div class="metric-value">{metrics_current.get('avg_response_time', 0):.1f}s</div>
            <div class="metric-compare">
                Sebelumnya: {metrics_past.get('avg_response_time', 0):.1f}s {format_delta('avg_response_time')}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">📝 Total Soal</div>
            <div class="metric-value">{int(metrics_current.get('total_questions', 0))}</div>
            <div class="metric-compare">
                Sebelumnya: {int(metrics_past.get('total_questions', 0))} {format_delta('total_questions', 0)}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">🎯 Sesi Latihan</div>
            <div class="metric-value">{int(metrics_current.get('total_sessions', 0))}</div>
            <div class="metric-compare">
                Sebelumnya: {int(metrics_past.get('total_sessions', 0))} {format_delta('total_sessions', 0)}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">😟 Struggle Events</div>
            <div class="metric-value">{int(metrics_current.get('struggle_count', 0))}</div>
            <div class="metric-compare">
                Sebelumnya: {int(metrics_past.get('struggle_count', 0))} {format_delta('struggle_count', 0)}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">💡 Hint Usage</div>
            <div class="metric-value">{metrics_current.get('hint_usage_rate', 0):.0f}%</div>
            <div class="metric-compare">
                Sebelumnya: {metrics_past.get('hint_usage_rate', 0):.0f}% {format_delta('hint_usage_rate', 0)}
            </div>
        </div>
    </div>
    """
    
    return html


def plot_accuracy_trend_comparison(
    metrics_current: Dict,
    metrics_past: Dict,
    daily_trends: pd.DataFrame = None
) -> go.Figure:
    """
    Plot line chart comparison accuracy trends
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics
        daily_trends: Optional daily trend dataframe
        
    Returns:
        Plotly figure
    """
    if daily_trends is not None and not daily_trends.empty:
        # Use actual daily data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_trends['date'],
            y=daily_trends['accuracy'],
            mode='lines+markers',
            name='Akurasi Harian',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Add average lines
        avg_current = metrics_current.get('accuracy', 0)
        avg_past = metrics_past.get('accuracy', 0)
        
        fig.add_hline(
            y=avg_current,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Rata-rata Sekarang: {avg_current:.1f}%",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=avg_past,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Rata-rata Sebelumnya: {avg_past:.1f}%",
            annotation_position="right"
        )
        
    else:
        # Simple bar comparison
        fig = go.Figure()
        
        categories = ['Periode Sekarang', 'Periode Sebelumnya']
        values = [
            metrics_current.get('accuracy', 0),
            metrics_past.get('accuracy', 0)
        ]
        
        colors = ['#1f77b4', '#ff7f0e']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='📈 Tren Akurasi',
        xaxis_title='',
        yaxis_title='Akurasi (%)',
        yaxis=dict(range=[0, 105]),
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def plot_mastery_heatmap(
    metrics_current: Dict,
    metrics_past: Dict,
    delta: Dict
) -> go.Figure:
    """
    Plot heatmap penguasaan konsep
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics
        delta: Delta metrics
        
    Returns:
        Plotly figure
    """
    mastery_changes = delta.get('mastery_changes', {})
    
    if not mastery_changes:
        # Empty state
        fig = go.Figure()
        fig.add_annotation(
            text="Belum ada data penguasaan konsep",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300)
        return fig
    
    # Prepare data
    concepts = list(mastery_changes.keys())
    current_values = [mastery_changes[c]['current'] for c in concepts]
    past_values = [mastery_changes[c]['past'] for c in concepts]
    deltas = [mastery_changes[c]['delta'] for c in concepts]
    
    # Sort by delta (improvement first)
    sorted_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)
    concepts = [concepts[i] for i in sorted_indices]
    current_values = [current_values[i] for i in sorted_indices]
    past_values = [past_values[i] for i in sorted_indices]
    deltas = [deltas[i] for i in sorted_indices]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Sekarang',
        x=concepts,
        y=current_values,
        marker_color='#1f77b4',
        text=[f"{v:.0f}%" for v in current_values],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Sebelumnya',
        x=concepts,
        y=past_values,
        marker_color='#ff7f0e',
        text=[f"{v:.0f}%" for v in past_values],
        textposition='outside'
    ))
    
    # Add delta annotations
    for i, (concept, delta_val) in enumerate(zip(concepts, deltas)):
        arrow = "↑" if delta_val > 0 else "↓" if delta_val < 0 else "→"
        color = "green" if delta_val > 0 else "red" if delta_val < 0 else "gray"
        
        fig.add_annotation(
            x=concept,
            y=max(current_values[i], past_values[i]) + 5,
            text=f"{arrow} {abs(delta_val):.0f}%",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            borderpad=2
        )
    
    fig.update_layout(
        title='🎯 Penguasaan Konsep (Perbandingan)',
        xaxis_title='Konsep Matematika',
        yaxis_title='Akurasi (%)',
        yaxis=dict(range=[0, 110]),
        barmode='group',
        height=500,
        hovermode='x unified',
        xaxis_tickangle=-45
    )
    
    return fig


def plot_response_time_distribution(
    metrics_current: Dict,
    metrics_past: Dict
) -> go.Figure:
    """
    Plot distribusi response time comparison
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    categories = ['Avg Time', 'Median Time', 'Max Time']
    
    current_times = [
        metrics_current.get('avg_response_time', 0),
        metrics_current.get('median_response_time', 0),
        metrics_current.get('max_response_time', 0)
    ]
    
    past_times = [
        metrics_past.get('avg_response_time', 0),
        metrics_past.get('median_response_time', 0),
        metrics_past.get('max_response_time', 0)
    ]
    
    fig.add_trace(go.Bar(
        name='Sekarang',
        x=categories,
        y=current_times,
        marker_color='#1f77b4',
        text=[f"{v:.1f}s" for v in current_times],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Sebelumnya',
        x=categories,
        y=past_times,
        marker_color='#ff7f0e',
        text=[f"{v:.1f}s" for v in past_times],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='⏱️ Response Time Distribution',
        xaxis_title='',
        yaxis_title='Time (seconds)',
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_emotion_distribution(
    metrics_current: Dict,
    metrics_past: Dict
) -> go.Figure:
    """
    Plot emotion state distribution comparison
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics
        
    Returns:
        Plotly figure
    """
    emotion_current = metrics_current.get('emotion_distribution', {})
    emotion_past = metrics_past.get('emotion_distribution', {})
    
    if not emotion_current and not emotion_past:
        fig = go.Figure()
        fig.add_annotation(
            text="Belum ada data emosi",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=300)
        return fig
    
    # Get all emotion states
    all_emotions = set(list(emotion_current.keys()) + list(emotion_past.keys()))
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=('Sekarang', 'Sebelumnya')
    )
    
    # Current emotions
    if emotion_current:
        fig.add_trace(
            go.Pie(
                labels=list(emotion_current.keys()),
                values=list(emotion_current.values()),
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set2)
            ),
            row=1, col=1
        )
    
    # Past emotions
    if emotion_past:
        fig.add_trace(
            go.Pie(
                labels=list(emotion_past.keys()),
                values=list(emotion_past.values()),
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set2)
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text='😊 Distribusi Kondisi Emosional',
        height=400,
        showlegend=True
    )
    
    return fig


def plot_learning_velocity(
    metrics_current: Dict,
    metrics_past: Dict,
    delta: Dict
) -> go.Figure:
    """
    Plot learning velocity metrics
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics
        delta: Delta metrics
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    metrics = [
        'Soal/Hari',
        'Sesi/Hari',
        'Hari Aktif'
    ]
    
    current_values = [
        metrics_current.get('avg_questions_per_day', 0),
        metrics_current.get('total_sessions', 0) / max(1, metrics_current.get('active_days', 1)),
        metrics_current.get('active_days', 0)
    ]
    
    past_values = [
        metrics_past.get('avg_questions_per_day', 0),
        metrics_past.get('total_sessions', 0) / max(1, metrics_past.get('active_days', 1)),
        metrics_past.get('active_days', 0)
    ]
    
    fig.add_trace(go.Bar(
        name='Sekarang',
        x=metrics,
        y=current_values,
        marker_color='#2ca02c',
        text=[f"{v:.1f}" for v in current_values],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Sebelumnya',
        x=metrics,
        y=past_values,
        marker_color='#d62728',
        text=[f"{v:.1f}" for v in past_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='🚀 Learning Velocity',
        xaxis_title='',
        yaxis_title='Count',
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_progress_radar(
    metrics_current: Dict,
    metrics_past: Dict
) -> go.Figure:
    """
    Plot radar chart untuk overall progress comparison
    
    Args:
        metrics_current: Current period metrics
        metrics_past: Past period metrics
        
    Returns:
        Plotly figure
    """
    categories = [
        'Akurasi',
        'Kecepatan',
        'Konsistensi',
        'Mandiri',
        'Stabilitas'
    ]
    
    # Normalize metrics to 0-100 scale
    current_values = [
        metrics_current.get('accuracy', 0),
        max(0, 100 - (metrics_current.get('avg_response_time', 0) * 5)),  # Lower time = better
        min(100, metrics_current.get('avg_questions_per_day', 0) * 10),
        max(0, 100 - metrics_current.get('hint_usage_rate', 0)),  # Lower hint = better
        max(0, 100 - metrics_current.get('struggle_rate', 0))  # Lower struggle = better
    ]
    
    past_values = [
        metrics_past.get('accuracy', 0),
        max(0, 100 - (metrics_past.get('avg_response_time', 0) * 5)),
        min(100, metrics_past.get('avg_questions_per_day', 0) * 10),
        max(0, 100 - metrics_past.get('hint_usage_rate', 0)),
        max(0, 100 - metrics_past.get('struggle_rate', 0))
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=current_values + [current_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Sekarang',
        line_color='#1f77b4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=past_values + [past_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Sebelumnya',
        line_color='#ff7f0e',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='🎯 Progress Radar',
        height=500
    )
    
    return fig


def create_insights_box(insights: List[str]) -> str:
    """
    Create HTML box untuk display insights
    
    Args:
        insights: List of insight strings
        
    Returns:
        HTML string
    """
    html = """
    <style>
        .insights-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .insights-title {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .insight-item {
            background: rgba(255,255,255,0.1);
            border-left: 4px solid white;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 5px;
            backdrop-filter: blur(10px);
        }
    </style>
    
    <div class="insights-box">
        <div class="insights-title">💡 Insight Otomatis</div>
    """
    
    for insight in insights:
        html += f'<div class="insight-item">{insight}</div>\n'
    
    html += "</div>"
    
    return html