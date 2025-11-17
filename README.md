# RTL Language Transcription App

A Streamlit application for displaying Arabic and other right-to-left (RTL) languages, with audio recording capabilities and integration with Hoppepper transcription platform.

## Features

- ✅ **RTL Language Support**: Properly displays Arabic, Hebrew, Persian, Urdu, and other RTL languages
- ✅ **Text Database**: SQLite database for storing and managing texts
- ✅ **uploads to S3**: upon user submission of a recording, the audio uploads to the cloud in text/audio pairs
- ✅ **Audio Recording**: Built-in audio recorder for capturing user readings
- ✅ **Recording History**: Track all recordings per text

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Hoppepper API credentials (optional):
   - Create a `.streamlit/secrets.toml` file:
   ```toml
   HOPPEPPER_API_KEY = "your_api_key_here"
   HOPPEPPER_API_URL = "https://api.hoppepper.com"
   ```
   - Or set environment variables:
   ```bash
   export HOPPEPPER_API_KEY="your_api_key_here"
   export HOPPEPPER_API_URL="https://api.hoppepper.com"
   ```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Add texts:
   - Use the sidebar to add sample texts
   - Or programmatically add texts to the database

3. Load texts:
   - Click "Load Texts from Database" in the sidebar

4. Record audio:
   - Navigate through texts using Previous/Next buttons
   - Click the microphone to record
   - Save recording locally or upload to Hoppepper

## Database Structure

The app uses SQLite with two main tables:

- **texts**: Stores text content, language, and RTL flag
- **recordings**: Stores audio file paths, Hoppepper job IDs, and status


## Customization

- Modify `HOPPEPPER_API_URL` if using a different endpoint
- Adjust RTL styling in the text display section
- Add more languages by updating the language selector

## Notes

- Audio files are saved in the `recordings/` directory
- Database file `texts.db` is created automatically
- RTL text rendering uses CSS direction and text-align properties

