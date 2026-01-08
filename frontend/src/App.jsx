import { useState } from 'react'
import './index.css'
import { Layout } from 'lucide-react';
import FaceAnalyzer from './components/FaceAnalyzer';

function App() {
  const [view, setView] = useState('home'); // 'home' | 'analyze'

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
        <nav style={{ display: 'flex', gap: '2rem', fontSize: '0.9rem', color: 'hsl(var(--txt-secondary))' }}>
          <span onClick={() => setView('home')} style={{ cursor: 'pointer', color: view === 'home' ? 'white' : 'inherit' }}>Dashboard</span>
          <span style={{ cursor: 'pointer' }}>Candidates</span>
          <span style={{ cursor: 'pointer' }}>Market Profiles</span>
        </nav>
      </header>

      <main style={{ flex: 1, padding: '2rem', overflowY: 'auto' }}>
        <div className="container">
          {view === 'home' ? (
            <div style={{ maxWidth: '600px', margin: '4rem auto', textAlign: 'center' }}>
              <h2 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>Facial Analysis Engine</h2>
              <p className="text-muted" style={{ fontSize: '1.1rem', marginBottom: '2rem', lineHeight: 1.6 }}>
                Advanced morphology extraction and statistical market fit evaluation.
                <br />
                Upload a frontal portrait to begin analysis.
              </p>

              <button
                className="btn"
                onClick={() => setView('analyze')}
                style={{
                  padding: '1rem 2rem',
                  fontSize: '1.1rem',
                  background: 'linear-gradient(135deg, hsl(var(--accent-primary)), hsl(280, 80%, 60%))',
                  boxShadow: '0 0 20px hsla(var(--accent-primary), 0.3)'
                }}
              >
                Launch Analysis
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
