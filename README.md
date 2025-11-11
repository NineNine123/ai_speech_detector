# Clone the repo
git clone https://github.com/ninenine123/ai_speech_detector.git
cd ai_speech_detector

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run web_app/app.py
