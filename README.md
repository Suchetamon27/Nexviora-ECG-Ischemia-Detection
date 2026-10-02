# Nexviora-ECG-Ischemia-Detection

This project provides a live ECG streaming and analysis pipeline capable of automatically detecting anomalies like myocardial ischemia in real-time. It uses a backend built with FastAPI, a frontend utilizing WebSockets and HTML5 Canvas, and leverages a local Ollama model (`qwen2.5vl:3b`) to perform inference on streaming ECG data.

## Features
- Reads streaming ECG data from a hardware serial port (or simulated mock data).
- Plots real-time ECG signals via a Web UI with live timestamps.
- Slices streaming ECG data into windows and feeds them as visual plots to an AI vision model.
- Analyzes ECG graphs for ischemia markers (e.g., ST-segment deviation, T-wave abnormalities) using the `qwen2.5vl:3b` Ollama model.
- Displays live Ischemia Detection reports, categorized by severity (Normal, Indeterminate, Warning) along with baseline vitals.

## Prerequisites
1. **Python 3.12+**
2. **Ollama:** You must have [Ollama](https://ollama.com/) installed and running locally.
3. **Qwen2.5-VL Model:** Pull the `qwen2.5vl:3b` model via Ollama.
   ```bash
   ollama pull qwen2.5vl:3b
   ```

## Setup Instructions
1. Navigate to the `backend` directory.
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (if not already done).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install the dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Starting the Server
1. Ensure the Ollama service is running in the background.
2. From the root of the project, activate the virtual environment and start the FastAPI server:
   ```bash
   PYTHONPATH=backend backend/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   *Alternatively, navigate into the `backend` folder and run `uvicorn app.main:app --host 0.0.0.0 --port 8000`.*
3. Once the server is running, open your web browser and navigate to:
   ```
   http://localhost:8000
   ```
   The frontend UI will automatically connect to the backend via WebSockets, and you will see the live ECG data being plotted and analyzed!
