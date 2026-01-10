#!/bin/bash
echo "🚀 Launching Morphology Scout Platform..."

# Kill any existing processes on ports 8000 (API) and 5173 (Frontend)
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start Backend
echo "📈 Starting Python Analysis Engine..."
cd backend
source ../.venv/bin/activate
nohup python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend running (PID: $BACKEND_PID)"
cd ..

# Start Frontend
echo "🎨 Starting React Dashboard..."
cd frontend
nohup npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend running (PID: $FRONTEND_PID)"
cd ..

echo "✅ Deployment Complete!"
echo "   - API: http://localhost:8000"
echo "   - UI:  http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop servers."

# Keep script running to trap exit
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
