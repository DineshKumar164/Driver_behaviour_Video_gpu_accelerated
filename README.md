# 🚗 Driver Behavior Safety System

A real-time driver monitoring system that uses computer vision and machine learning to detect drowsiness and distraction, helping prevent accidents and improve road safety.

## ✨ Features

- **Real-time Detection**
  - Drowsiness detection (eyes closed > 0.5 seconds)
  - Distraction detection (looking away > 0.5 seconds)
  - Face tracking and analysis
  - Immediate visual alerts

- **Advanced Analytics**
  - Risk level assessment (Low/Medium/High)
  - Behavior statistics and trends
  - High-risk segment identification
  - Duration-based analysis

- **Modern UI/UX**
  - Dark theme interface
  - Real-time progress tracking
  - Interactive metrics display
  - Responsive design
  - Beautiful gradient cards

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/driver-behavior-safety.git
   cd driver-behavior-safety
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Upload a video**
   - Click "Choose a video file"
   - Select any MP4, AVI, or MOV file
   - Click "Process Video"
   - View real-time analysis results

## 🛠️ Technical Details

### System Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)
- Webcam for live monitoring (optional)

### Key Components
- **Video Processor**
  - OpenCV with CUDA acceleration
  - Haar Cascade Classifiers
  - Real-time frame processing
  - Multi-threaded analysis

- **Detection Parameters**
  ```python
  DROWSINESS_THRESHOLD = 0.5  # seconds
  DISTRACTION_THRESHOLD = 0.5  # seconds
  RISK_LEVELS = {
      'HIGH': > 20%,
      'MEDIUM': 10-20%,
      'LOW': < 10%
  }
  ```

### Performance Optimizations
- GPU acceleration for video processing
- Batch frame processing
- Efficient memory management
- Parallel stream processing

## 📊 Analysis Metrics

The system tracks and analyzes:
- Total drowsy time
- Total distracted time
- Number of high-risk events
- Risk level distribution
- Behavior patterns
- Face position and movement

## 🎯 Use Cases

1. **Professional Drivers**
   - Fleet management
   - Safety compliance
   - Performance monitoring

2. **Safety Research**
   - Behavior analysis
   - Risk assessment
   - Pattern identification

3. **Training Programs**
   - Driver education
   - Safety awareness
   - Best practices demonstration

## 🔒 Privacy & Security

- No video data is stored permanently
- All processing done locally
- Temporary files automatically cleaned
- No external API dependencies

## 🌟 Future Enhancements

- [ ] Live webcam monitoring
- [ ] Multi-camera support
- [ ] Cloud synchronization
- [ ] Mobile app integration
- [ ] Advanced ML models
- [ ] Custom alert systems

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community
- Streamlit framework
- Python ecosystem
- Computer vision researchers

## 📧 Contact

Your Name - your.email@example.com
Project Link: [https://github.com/yourusername/driver-behavior-safety](https://github.com/yourusername/driver-behavior-safety)

---
Made with ❤️ for safer roads
