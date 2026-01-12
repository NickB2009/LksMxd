import { useState, useEffect } from 'react'
import './index.css'
import { Layout, CheckCircle2, XCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import FaceAnalyzer from './components/FaceAnalyzer';

function App() {
  const [view, setView] = useState('home'); // 'home' | 'analyze'
  const [serverStatus, setServerStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await fetch('http://localhost:8000/');
        if (res.ok) {
          setServerStatus('online');
          setRestarting(false);
        } else {
          setServerStatus('offline');
        }
      } catch (err) {
        setServerStatus('offline');
      }
    };
    checkServer();
    const interval = setInterval(checkServer, restarting ? 2000 : 10000);
    return () => clearInterval(interval);
  }, [restarting]);

  const handleRestart = async () => {
    setRestarting(true);
    setServerStatus('checking');
    try {
      await fetch('http://localhost:8000/restart', { method: 'POST' });
    } catch (err) {
      // It will fail because the server closes the connection on restart
      console.log("Server restarting...");
    }
  };

  return (
    <div className="app-container" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{
        padding: '1.5rem',
        borderBottom: '1px solid hsl(var(--border-subtle))',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'hsla(var(--bg-app), 0.8)',
        backdropFilter: 'blur(10px)',
        position: 'sticky',
        top: 0,
        zIndex: 10
      }}>
        <div
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}
          onClick={() => setView('home')}
        >
          <div style={{ width: 24, height: 24, background: 'hsl(var(--accent-primary))', borderRadius: '50%' }}></div>
          <h1 style={{ fontSize: '1.25rem', margin: 0 }}>MORPHOLOGY <span style={{ color: 'hsl(var(--txt-secondary))', fontWeight: 400 }}>// SCOUT</span></h1>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              padding: '4px 12px',
              borderRadius: '20px',
              background: serverStatus === 'online' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              color: serverStatus === 'online' ? '#22c55e' : '#ef4444',
              border: `1px solid ${serverStatus === 'online' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`
            }}>
              {serverStatus === 'online' ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
              {restarting ? 'Restarting...' : (serverStatus === 'online' ? 'Engine Online' : 'Engine Offline')}
            </div>

            {serverStatus === 'online' && (
              <button
                onClick={handleRestart}
                disabled={restarting}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'hsl(var(--txt-secondary))',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  fontSize: '0.75rem',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  transition: 'all 0.2s ease',
                  opacity: restarting ? 0.5 : 1
                }}
                onMouseOver={(e) => e.currentTarget.style.color = 'white'}
                onMouseOut={(e) => e.currentTarget.style.color = 'hsl(var(--txt-secondary))'}
              >
                <RefreshCw size={12} className={restarting ? 'animate-spin' : ''} />
                Restart
              </button>
            )}
          </div>

          <nav style={{ display: 'flex', gap: '2rem', fontSize: '0.9rem', color: 'hsl(var(--txt-secondary))' }}>
            <span onClick={() => setView('home')} style={{ cursor: 'pointer', color: view === 'home' ? 'white' : 'inherit' }}>Dashboard</span>
            <span style={{ cursor: 'pointer' }}>Candidates</span>
            <span style={{ cursor: 'pointer' }}>Market Profiles</span>
          </nav>
        </div>
      </header>

      <main style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
        <div className="container">
          {view === 'home' ? (
            <div style={{ maxWidth: '600px', margin: '4rem auto', textAlign: 'center' }}>
              <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Facial Analysis Engine</h2>
              <p className="text-muted" style={{ fontSize: '1.1rem', marginBottom: '2rem', lineHeight: 1.6 }}>
                Advanced morphology extraction and statistical market fit evaluation.
              </p>

              {serverStatus === 'offline' && (
                <div style={{
                  background: 'rgba(239, 68, 68, 0.05)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  padding: '1.5rem',
                  borderRadius: 'var(--radius-md)',
                  marginBottom: '2rem',
                  textAlign: 'left'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444', marginBottom: '0.5rem', fontWeight: 600 }}>
                    <AlertTriangle size={18} />
                    Backend API is not running
                  </div>
                  <p style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'hsl(var(--txt-secondary))' }}>
                    The analysis engine must be active to process images. Run the following command in your terminal:
                  </p>
                  <code style={{
                    display: 'block',
                    background: '#000',
                    padding: '1rem',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.85rem',
                    color: 'hsl(var(--accent-primary))',
                    fontFamily: 'monospace'
                  }}>
                    bash start.sh
                  </code>
                </div>
              )}

              <button
                className="btn"
                onClick={() => setView('analyze')}
                disabled={serverStatus === 'offline'}
                style={{
                  padding: '1rem 2rem',
                  fontSize: '1.1rem',
                  background: serverStatus === 'offline' ? 'hsl(var(--bg-panel))' : 'linear-gradient(135deg, hsl(var(--accent-primary)), hsl(280, 80%, 60%))',
                  boxShadow: serverStatus === 'offline' ? 'none' : '0 0 20px hsla(var(--accent-primary), 0.3)',
                  opacity: serverStatus === 'offline' ? 0.6 : 1,
                  cursor: serverStatus === 'offline' ? 'not-allowed' : 'pointer'
                }}
              >
                {serverStatus === 'offline' ? 'Wait for Engine...' : 'Launch Analysis'}
              </button>
            </div>
          ) : (
            <div className="fade-in">
              <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
                <h2 style={{ marginBottom: '0.5rem' }}>Morphology & Market Fit</h2>
                <p className="text-muted">AI-powered facial structure decomposition</p>
              </div>
              <FaceAnalyzer />
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
