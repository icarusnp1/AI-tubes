"""
Analytics Engine for Adaptive Learning System
Handles data processing, metrics calculation, and comparison logic
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


class AnalyticsEngine:
    """
    Core engine untuk learning analytics dengan time-based comparison
    """
    
    def __init__(self, events_csv_path: str):
        """
        Initialize analytics engine
        
        Args:
            events_csv_path: Path to events.csv file
        """
        self.events_path = events_csv_path
        self.df = None
        
    def load_data(self) -> bool:
        """
        Load events.csv into dataframe
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.events_path):
            return False
            
        try:
            self.df = pd.read_csv(self.events_path)
            
            # Parse timestamp
            self.df['ts'] = pd.to_datetime(self.df['ts'], errors='coerce')
            
            # Drop rows with invalid timestamps
            self.df = self.df.dropna(subset=['ts'])
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_user_data(self, user_id: str) -> pd.DataFrame:
        """
        Filter data untuk user tertentu
        
        Args:
            user_id: Student ID
            
        Returns:
            Filtered dataframe
        """
        if self.df is None:
            return pd.DataFrame()
        
        return self.df[self.df['user_id'] == user_id].copy()
    
    def filter_by_period(
        self, 
        df: pd.DataFrame, 
        end_date: datetime, 
        days_back: int
    ) -> pd.DataFrame:
        """
        Filter data untuk periode waktu tertentu
        
        Args:
            df: Input dataframe
            end_date: End of period
            days_back: Number of days to look back
            
        Returns:
            Filtered dataframe
        """
        start_date = end_date - timedelta(days=days_back)
        mask = (df['ts'] >= start_date) & (df['ts'] <= end_date)
        return df[mask].copy()
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute learning metrics dari dataframe
        
        Args:
            df: Input dataframe (already filtered by user and time)
            
        Returns:
            Dictionary of metrics
        """
        if df.empty:
            return self._empty_metrics()
        
        # Filter events
        answers = df[df['event'] == 'SUBMIT_ANSWER'].copy()
        results = df[df['event'] == 'SET_RESULT'].copy()
        
        metrics = {}
        
        # Basic counts
        metrics['total_questions'] = len(answers)
        metrics['total_sessions'] = len(results)
        
        # Accuracy
        if len(answers) > 0:
            metrics['accuracy'] = answers['is_correct'].mean() * 100
            metrics['correct_count'] = answers['is_correct'].sum()
            metrics['wrong_count'] = len(answers) - metrics['correct_count']
        else:
            metrics['accuracy'] = 0
            metrics['correct_count'] = 0
            metrics['wrong_count'] = 0
        
        # Response time
        if len(answers) > 0:
            valid_times = answers['response_time'].dropna()
            if len(valid_times) > 0:
                metrics['avg_response_time'] = valid_times.mean()
                metrics['median_response_time'] = valid_times.median()
                metrics['max_response_time'] = valid_times.max()
            else:
                metrics['avg_response_time'] = 0
                metrics['median_response_time'] = 0
                metrics['max_response_time'] = 0
        else:
            metrics['avg_response_time'] = 0
            metrics['median_response_time'] = 0
            metrics['max_response_time'] = 0
        
        # Struggle tracking
        if len(results) > 0:
            struggle_mask = results['struggle_pred'] == 'STRUGGLE'
            metrics['struggle_count'] = struggle_mask.sum()
            metrics['struggle_rate'] = (struggle_mask.sum() / len(results)) * 100
        else:
            metrics['struggle_count'] = 0
            metrics['struggle_rate'] = 0
        
        # Hint usage
        if len(answers) > 0:
            metrics['hint_usage_rate'] = (answers['used_hint'].sum() / len(answers)) * 100
            metrics['explanation_usage_rate'] = (answers['used_explanation'].sum() / len(answers)) * 100
        else:
            metrics['hint_usage_rate'] = 0
            metrics['explanation_usage_rate'] = 0
        
        # Emotion states (from SET_RESULT)
        if len(results) > 0:
            emotion_counts = results['emotion_state'].value_counts().to_dict()
            metrics['emotion_distribution'] = emotion_counts
            metrics['dominant_emotion'] = results['emotion_state'].mode()[0] if len(results) > 0 else "N/A"
        else:
            metrics['emotion_distribution'] = {}
            metrics['dominant_emotion'] = "N/A"
        
        # Level progress
        if len(df) > 0:
            metrics['current_level'] = df['level'].max()
            metrics['level_distribution'] = df['level'].value_counts().to_dict()
        else:
            metrics['current_level'] = 0
            metrics['level_distribution'] = {}
        
        # Mastery per concept
        metrics['mastery_per_concept'] = self._compute_mastery_per_concept(answers)
        
        # Time spent
        metrics['total_time_spent'] = answers['response_time'].sum()
        
        # Daily activity
        if len(df) > 0:
            metrics['active_days'] = df['ts'].dt.date.nunique()
            metrics['avg_questions_per_day'] = metrics['total_questions'] / max(1, metrics['active_days'])
        else:
            metrics['active_days'] = 0
            metrics['avg_questions_per_day'] = 0
        
        return metrics
    
    def _compute_mastery_per_concept(self, answers: pd.DataFrame) -> Dict:
        """
        Compute accuracy per concept tag
        
        Args:
            answers: Dataframe of SUBMIT_ANSWER events
            
        Returns:
            Dictionary of {concept_tag: accuracy_percentage}
        """
        if answers.empty:
            return {}
        
        # Filter out rows without concept_tag
        answers = answers[answers['concept_tag'].notna()].copy()
        
        if answers.empty:
            return {}
        
        mastery = {}
        for concept in answers['concept_tag'].unique():
            concept_answers = answers[answers['concept_tag'] == concept]
            if len(concept_answers) > 0:
                accuracy = concept_answers['is_correct'].mean() * 100
                mastery[concept] = {
                    'accuracy': accuracy,
                    'count': len(concept_answers),
                    'correct': concept_answers['is_correct'].sum()
                }
        
        return mastery
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_questions': 0,
            'total_sessions': 0,
            'accuracy': 0,
            'correct_count': 0,
            'wrong_count': 0,
            'avg_response_time': 0,
            'median_response_time': 0,
            'max_response_time': 0,
            'struggle_count': 0,
            'struggle_rate': 0,
            'hint_usage_rate': 0,
            'explanation_usage_rate': 0,
            'emotion_distribution': {},
            'dominant_emotion': "N/A",
            'current_level': 0,
            'level_distribution': {},
            'mastery_per_concept': {},
            'total_time_spent': 0,
            'active_days': 0,
            'avg_questions_per_day': 0
        }
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """
        Extract session data untuk user dari events
        Session = grup events yang diakhiri dengan SET_RESULT
        
        Args:
            user_id: Student ID
            
        Returns:
            List of session dicts dengan metadata
        """
        user_df = self.get_user_data(user_id)
        
        if user_df.empty:
            return []
        
        # Find SET_RESULT events (marks end of session)
        result_events = user_df[user_df['event'] == 'SET_RESULT'].copy()
        
        if result_events.empty:
            return []
        
        # Sort by timestamp
        result_events = result_events.sort_values('ts')
        
        sessions = []
        
        for idx, row in result_events.iterrows():
            session_end_ts = row['ts']
            
            # Get all events before this SET_RESULT (same session)
            # Find START_SET or previous SET_RESULT as session start
            prev_results = result_events[result_events['ts'] < session_end_ts]
            
            if not prev_results.empty:
                session_start_ts = prev_results['ts'].max()
            else:
                # First session - start from earliest event
                session_start_ts = user_df['ts'].min()
            
            # Filter events in this session
            session_mask = (user_df['ts'] > session_start_ts) & (user_df['ts'] <= session_end_ts)
            session_df = user_df[session_mask].copy()
            
            # Build session metadata
            session_info = {
                'session_number': len(sessions) + 1,
                'start_ts': session_start_ts,
                'end_ts': session_end_ts,
                'duration': (session_end_ts - session_start_ts).total_seconds(),
                'level': row.get('level', 0),
                'events_df': session_df,
                'accuracy': row.get('accuracy', 0) if pd.notna(row.get('accuracy')) else 0,
                'struggle_pred': row.get('struggle_pred', 'N/A'),
                'emotion_state': row.get('emotion_state', 'N/A')
            }
            
            sessions.append(session_info)
        
        return sessions
    
    def compare_sessions(
        self, 
        user_id: str, 
        sessions_back: int = 1
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Compare performance antara sesi terakhir vs sesi ke-N sebelumnya
        
        Args:
            user_id: Student ID
            sessions_back: Berapa sesi ke belakang untuk comparison (1-30)
            
        Returns:
            Tuple of (metrics_current, metrics_past, delta_metrics)
        """
        sessions = self.get_user_sessions(user_id)
        
        if len(sessions) < 2:
            # Not enough sessions for comparison
            return self._empty_metrics(), self._empty_metrics(), {}
        
        # Current session = latest
        current_session = sessions[-1]
        
        # Past session = N sessions ago
        past_index = -(sessions_back + 1)
        
        if abs(past_index) > len(sessions):
            # Requested session doesn't exist, use oldest
            past_session = sessions[0]
        else:
            past_session = sessions[past_index]
        
        # Compute metrics for each session
        metrics_current = self.compute_metrics(current_session['events_df'])
        metrics_past = self.compute_metrics(past_session['events_df'])
        
        # Add session metadata
        metrics_current['session_number'] = current_session['session_number']
        metrics_current['session_start'] = current_session['start_ts'].strftime('%Y-%m-%d %H:%M')
        metrics_current['session_end'] = current_session['end_ts'].strftime('%Y-%m-%d %H:%M')
        metrics_current['session_duration'] = current_session['duration']
        
        metrics_past['session_number'] = past_session['session_number']
        metrics_past['session_start'] = past_session['start_ts'].strftime('%Y-%m-%d %H:%M')
        metrics_past['session_end'] = past_session['end_ts'].strftime('%Y-%m-%d %H:%M')
        metrics_past['session_duration'] = past_session['duration']
        
        # Calculate deltas
        delta = self._calculate_delta(metrics_current, metrics_past)
        
        return metrics_current, metrics_past, delta
    
    def compare_periods(
        self, 
        user_id: str, 
        comparison_days: int,
        window_size: int = 7
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Compare performance antara 2 periode waktu
        
        Args:
            user_id: Student ID
            comparison_days: Berapa hari ke belakang untuk comparison (1-30)
            window_size: Ukuran window untuk setiap periode (default 7 hari)
            
        Returns:
            Tuple of (metrics_current, metrics_past, delta_metrics)
        """
        user_df = self.get_user_data(user_id)
        
        if user_df.empty:
            return self._empty_metrics(), self._empty_metrics(), {}
        
        # Get latest timestamp
        latest_ts = user_df['ts'].max()
        
        # Period A (Current): last `window_size` days
        period_a_end = latest_ts
        period_a_df = self.filter_by_period(user_df, period_a_end, window_size)
        
        # Period B (Past): `comparison_days` to `comparison_days + window_size` ago
        period_b_end = latest_ts - timedelta(days=comparison_days)
        period_b_df = self.filter_by_period(user_df, period_b_end, window_size)
        
        # Compute metrics
        metrics_current = self.compute_metrics(period_a_df)
        metrics_past = self.compute_metrics(period_b_df)
        
        # Calculate deltas
        delta = self._calculate_delta(metrics_current, metrics_past)
        
        # Add period info
        metrics_current['period_start'] = (period_a_end - timedelta(days=window_size)).strftime('%Y-%m-%d')
        metrics_current['period_end'] = period_a_end.strftime('%Y-%m-%d')
        
        metrics_past['period_start'] = (period_b_end - timedelta(days=window_size)).strftime('%Y-%m-%d')
        metrics_past['period_end'] = period_b_end.strftime('%Y-%m-%d')
        
        return metrics_current, metrics_past, delta
    
    def _calculate_delta(self, current: Dict, past: Dict) -> Dict:
        """
        Calculate delta between current and past metrics
        
        Args:
            current: Current period metrics
            past: Past period metrics
            
        Returns:
            Dictionary of deltas with directions
        """
        delta = {}
        
        # Numeric deltas
        numeric_keys = [
            'accuracy', 'avg_response_time', 'total_questions', 
            'struggle_count', 'struggle_rate', 'hint_usage_rate',
            'total_sessions', 'avg_questions_per_day'
        ]
        
        for key in numeric_keys:
            curr_val = current.get(key, 0)
            past_val = past.get(key, 0)
            
            diff = curr_val - past_val
            
            # Determine if positive change is good or bad
            positive_is_good = key not in ['avg_response_time', 'struggle_count', 'struggle_rate', 'hint_usage_rate']
            
            if positive_is_good:
                direction = "up" if diff > 0 else ("down" if diff < 0 else "same")
                sentiment = "positive" if diff > 0 else ("negative" if diff < 0 else "neutral")
            else:
                direction = "down" if diff < 0 else ("up" if diff > 0 else "same")
                sentiment = "positive" if diff < 0 else ("negative" if diff > 0 else "neutral")
            
            delta[key] = {
                'value': diff,
                'percentage': (diff / past_val * 100) if past_val != 0 else 0,
                'direction': direction,
                'sentiment': sentiment
            }
        
        # Mastery delta
        delta['mastery_changes'] = self._compare_mastery(
            current.get('mastery_per_concept', {}),
            past.get('mastery_per_concept', {})
        )
        
        return delta
    
    def _compare_mastery(self, current: Dict, past: Dict) -> Dict:
        """
        Compare mastery levels per concept
        
        Args:
            current: Current mastery dict
            past: Past mastery dict
            
        Returns:
            Dictionary of mastery changes per concept
        """
        changes = {}
        
        # Get all concepts
        all_concepts = set(list(current.keys()) + list(past.keys()))
        
        for concept in all_concepts:
            curr_acc = current.get(concept, {}).get('accuracy', 0)
            past_acc = past.get(concept, {}).get('accuracy', 0)
            
            diff = curr_acc - past_acc
            
            changes[concept] = {
                'current': curr_acc,
                'past': past_acc,
                'delta': diff,
                'direction': "up" if diff > 0 else ("down" if diff < 0 else "same")
            }
        
        return changes
    
    def get_daily_trends(
        self, 
        user_id: str, 
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Get daily aggregated metrics for trend visualization
        
        Args:
            user_id: Student ID
            days_back: Number of days to look back
            
        Returns:
            Dataframe with daily metrics
        """
        user_df = self.get_user_data(user_id)
        
        if user_df.empty:
            return pd.DataFrame()
        
        # Filter last N days
        latest_ts = user_df['ts'].max()
        start_date = latest_ts - timedelta(days=days_back)
        
        filtered_df = user_df[user_df['ts'] >= start_date].copy()
        
        # Extract date
        filtered_df['date'] = filtered_df['ts'].dt.date
        
        # Get answer events
        answers = filtered_df[filtered_df['event'] == 'SUBMIT_ANSWER'].copy()
        
        if answers.empty:
            return pd.DataFrame()
        
        # Group by date
        daily = answers.groupby('date').agg({
            'is_correct': ['sum', 'count', 'mean'],
            'response_time': 'mean',
            'used_hint': 'sum'
        }).reset_index()
        
        # Flatten columns
        daily.columns = ['date', 'correct', 'total', 'accuracy', 'avg_time', 'hints_used']
        
        # Convert accuracy to percentage
        daily['accuracy'] = daily['accuracy'] * 100
        
        return daily
    
    def get_date_range(self, df: pd.DataFrame) -> Tuple[datetime, datetime]:
        """
        Get earliest and latest timestamp dari dataframe
        
        Args:
            df: Input dataframe (with 'ts' column)
            
        Returns:
            Tuple of (earliest_date, latest_date)
        """
        if df.empty:
            now = datetime.now()
            return now, now
        
        # Pastikan kolom ts ada
        if 'ts' not in df.columns:
            now = datetime.now()
            return now, now
        
        # Ambil min dan max timestamp
        earliest = df['ts'].min()
        latest = df['ts'].max()
        
        # Handle jika ada NaT values
        if pd.isna(earliest) or pd.isna(latest):
            now = datetime.now()
            return now, now
        
        return earliest, latest
    
    def generate_insights(
        self, 
        metrics_current: Dict, 
        metrics_past: Dict, 
        delta: Dict
    ) -> List[str]:
        """
        Generate auto insights berdasarkan comparison
        
        Args:
            metrics_current: Current metrics
            metrics_past: Past metrics
            delta: Delta metrics
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Accuracy insights
        acc_delta = delta.get('accuracy', {})
        if acc_delta.get('value', 0) > 5:
            insights.append(f"🎉 Akurasi naik {acc_delta['value']:.1f}% - Excellent progress!")
        elif acc_delta.get('value', 0) < -5:
            insights.append(f"⚠️ Akurasi turun {abs(acc_delta['value']):.1f}% - Butuh fokus lebih!")
        
        # Response time insights
        time_delta = delta.get('avg_response_time', {})
        if time_delta.get('value', 0) < -2:
            insights.append(f"⚡ Response time makin cepat {abs(time_delta['value']):.1f}s - Kamu makin confident!")
        
        # Struggle insights
        struggle_delta = delta.get('struggle_count', {})
        if struggle_delta.get('value', 0) < 0:
            insights.append(f"💪 Struggle events berkurang {abs(int(struggle_delta['value']))} kali - Keep it up!")
        elif struggle_delta.get('value', 0) > 2:
            insights.append(f"🤔 Struggle events bertambah - Mungkin perlu review materi lagi")
        
        # Activity insights
        questions_delta = delta.get('total_questions', {})
        if questions_delta.get('value', 0) > 10:
            insights.append(f"📈 Kamu mengerjakan {int(questions_delta['value'])} soal lebih banyak - Great consistency!")
        
        # Mastery insights
        mastery_changes = delta.get('mastery_changes', {})
        improved_concepts = [c for c, d in mastery_changes.items() if d['delta'] > 10]
        if improved_concepts:
            concepts_str = ", ".join([f"`{c}`" for c in improved_concepts[:2]])
            insights.append(f"🎯 Penguasaan meningkat signifikan di: {concepts_str}")
        
        # Hint usage insights
        hint_delta = delta.get('hint_usage_rate', {})
        if hint_delta.get('value', 0) < -10:
            insights.append(f"🌟 Penggunaan hint berkurang {abs(hint_delta['value']):.0f}% - Kamu makin mandiri!")
        
        # Emoji based on overall performance
        if metrics_current.get('accuracy', 0) >= 80:
            insights.insert(0, "⭐ Overall performance excellent!")
        elif metrics_current.get('accuracy', 0) < 60:
            insights.insert(0, "💡 Ada ruang untuk improvement - jangan menyerah!")
        
        return insights if insights else ["📊 Data tersedia untuk analisis lebih lanjut."]