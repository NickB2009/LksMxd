# Getting Started with Morphology Scout

Follow these instructions to set up and run the facial analysis platform on your local machine.

## Prerequisites
- **Python 3.8+**
- **Node.js 16+** and **npm**

---

## 1. Backend Setup (FastAPI)

The backend handles the facial morphology calculations and image processing.

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**:
   ```bash
   python3 main.py
   ```
   *The API will be available at `http://localhost:8000`.*

---

## 2. Frontend Setup (Vite + React)

The frontend provides the interactive dashboard and file upload interface.

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   *The application will be available at `http://localhost:5173`.*

---

## Usage
1. Ensure the **Backend** is running first.
2. Open your browser to the **Frontend** URL.
3. Upload a clear, frontal reference photo to begin the analysis.
