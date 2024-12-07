import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Driver Behavior Analysis",
    page_icon="🚗",
    layout="wide"
)

import cv2
import numpy as np
import tempfile
import os
from video_processor import VideoProcessor
import time
import traceback

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'high_risk_segments' not in st.session_state:
    st.session_state.high_risk_segments = []
if 'video_stats' not in st.session_state:
    st.session_state.video_stats = None

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    /* Dark theme variables */
    :root {
        --background-color: #1E1E1E;
        --text-color: #FFFFFF;
        --card-background: #2D2D2D;
        --border-color: #404040;
        --accent-color: #0066FF;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --danger-color: #F44336;
    }

    /* Main container */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 2rem;
    }

    /* Cards */
    .stCard {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Metrics container */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, var(--card-background), #353535);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }

    /* Risk levels */
    .risk-low {
        color: var(--success-color);
        font-weight: bold;
    }
    .risk-moderate {
        color: var(--warning-color);
        font-weight: bold;
    }
    .risk-severe {
        color: var(--danger-color);
        font-weight: bold;
    }

    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)

def process_video(uploaded_file):
    """Process the uploaded video file."""
    tfile = None
    cap = None
    try:
        # Create temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()  # Close the file handle
        
        # Initialize video processor
        st.session_state.processor = VideoProcessor()
        
        # Open video file
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps
        
        # Initialize progress bar and stats
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize metrics
        total_drowsy_time = 0
        total_distracted_time = 0
        high_risk_segments = []
        current_segment = None
        frame_count = 0
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame, metrics, _ = st.session_state.processor.process_frame(frame)
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Update status
            frame_count += 1
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Track high-risk segments
            is_high_risk = (metrics['behaviors']['drowsy'] > 0.3 or 
                          metrics['behaviors']['distracted'] > 0.3)
            
            if is_high_risk:
                frame_time = frame_count / fps
                if current_segment is None:
                    current_segment = {'start': frame_time, 'end': frame_time}
                else:
                    current_segment['end'] = frame_time
            elif current_segment is not None:
                if current_segment['end'] - current_segment['start'] >= 1.0:  # Minimum 1 second
                    high_risk_segments.append(current_segment)
                current_segment = None
            
            # Update total times
            total_drowsy_time += metrics['behaviors']['drowsy'] / fps
            total_distracted_time += metrics['behaviors']['distracted'] / fps
        
        # Add final segment if exists
        if current_segment is not None and current_segment['end'] - current_segment['start'] >= 1.0:
            high_risk_segments.append(current_segment)
        
        # Store results in session state
        st.session_state.high_risk_segments = high_risk_segments
        st.session_state.video_stats = {
            'duration': duration,
            'total_drowsy_time': total_drowsy_time,
            'total_distracted_time': total_distracted_time,
            'high_risk_segments': len(high_risk_segments)
        }
        
        return True
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        st.code(traceback.format_exc())
        return False
        
    finally:
        # Clean up resources
        if cap is not None:
            cap.release()
        if tfile is not None:
            try:
                os.unlink(tfile.name)
            except Exception as e:
                st.warning(f"Warning: Could not delete temporary file: {str(e)}")

def display_high_risk_segments():
    """Display high-risk segments and statistics."""
    if not st.session_state.video_stats:
        return
    
    stats = st.session_state.video_stats
    duration = stats['duration']
    
    # Calculate risk percentages
    drowsy_percent = (stats['total_drowsy_time'] / duration) * 100
    distracted_percent = (stats['total_distracted_time'] / duration) * 100
    total_risk_percent = ((stats['total_drowsy_time'] + stats['total_distracted_time']) / duration) * 100
    
    # Determine overall risk level
    risk_level = (
        'SEVERE' if total_risk_percent >= 20 else
        'MODERATE' if total_risk_percent >= 10 else
        'LOW'
    )
    
    # Display overall risk level
    risk_color = (
        'risk-severe' if risk_level == 'SEVERE' else
        'risk-moderate' if risk_level == 'MODERATE' else
        'risk-low'
    )
    
    st.markdown(f"""
    <div class="stCard">
        <h2>Analysis Results</h2>
        <div class="metrics-container">
            <div class="metric-card">
                <h3>Overall Risk Level</h3>
                <p class="{risk_color}">{risk_level}</p>
            </div>
            <div class="metric-card">
                <h3>Total Duration</h3>
                <p>{duration:.1f} seconds</p>
            </div>
            <div class="metric-card">
                <h3>High Risk Events</h3>
                <p>{stats['high_risk_segments']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stCard">
            <h3>Drowsiness Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Drowsy Time", f"{stats['total_drowsy_time']:.1f}s")
        st.metric("Drowsy Percentage", f"{drowsy_percent:.1f}%")
        
    with col2:
        st.markdown("""
        <div class="stCard">
            <h3>Distraction Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Distracted Time", f"{stats['total_distracted_time']:.1f}s")
        st.metric("Distracted Percentage", f"{distracted_percent:.1f}%")
    
    # Display high-risk segments
    if st.session_state.high_risk_segments:
        st.markdown("""
        <div class="stCard">
            <h3>High Risk Segments</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, segment in enumerate(st.session_state.high_risk_segments, 1):
            duration = segment['end'] - segment['start']
            st.markdown(f"""
            <div class="metric-card">
                <h4>Segment {i}</h4>
                <p>Start: {segment['start']:.1f}s</p>
                <p>End: {segment['end']:.1f}s</p>
                <p>Duration: {duration:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function."""
    st.title("🚗 Driver Behavior Analysis")
    st.markdown("""
    <div class="stCard">
        <h3>Upload a video for analysis</h3>
        <p>The system will analyze driver behavior and identify high-risk segments.</p>
        <ul>
            <li>Drowsiness detection (eyes closed > 1 second)</li>
            <li>Distraction detection (looking away > 1 second)</li>
            <li>Real-time risk assessment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                success = process_video(uploaded_file)
                if success:
                    st.success("Video processing complete!")
                    display_high_risk_segments()

if __name__ == "__main__":
    main()
