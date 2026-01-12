import { useState } from 'react';
import { Scan, Sparkles, Eye, Scissors, ChevronRight, Activity, Layout } from 'lucide-react';
import FeatureCard from './FeatureCard';
import ScoreGauge from './ScoreGauge';

export default function MorphologyReport({ data }) {
    const [activeTab, setActiveTab] = useState('eyes');
    const [showAdvanced, setShowAdvanced] = useState(false);

    if (!data || !data.overall) return null;

    const { overall, proportions, eyes, nose, jawline, detailed } = data;

    const sections = [
        { id: 'eyes', label: 'Eyes & Gaze', icon: Eye, color: 'hsl(210, 80%, 60%)' },
        { id: 'nose', label: 'Nasal Structure', icon: Activity, color: 'hsl(280, 70%, 60%)' },
        { id: 'jawline', label: 'Jaw & Chin', icon: Scissors, color: 'hsl(150, 70%, 50%)' },
        { id: 'proportions', label: 'Face Harmony', icon: Layout, color: 'hsl(30, 90%, 60%)' }
    ];

    return (
        <div className="fade-in" style={{ marginTop: '2rem', animationDelay: '0.2s' }}>
            {/* Header / Summary */}
            <div className="card glass" style={{
                background: 'linear-gradient(135deg, hsla(var(--accent-primary), 0.1), hsla(var(--bg-panel), 0.3))',
                padding: '2.5rem',
                marginBottom: '2rem',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid hsla(var(--accent-primary), 0.2)',
                display: 'flex',
                alignItems: 'center',
                gap: '3rem',
                flexWrap: 'wrap'
            }}>
                <div style={{ flex: '0 0 auto' }}>
                    <ScoreGauge
                        value={overall.harmonyScore}
                        ideal={85}
                        label="Harmony Score"
                        size={160}
                    />
                </div>

                <div style={{ flex: '1 1 400px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                        <Sparkles size={24} color="hsl(var(--accent-primary))" />
                        <h2 style={{ margin: 0, fontSize: '1.75rem', letterSpacing: '-0.02em' }}>Aesthetic Assessment</h2>
                        <span className="badge" style={{ fontSize: '0.7rem', opacity: 0.8 }}>CLAHE-PRECISION ACTIVE</span>
                    </div>

                    <p style={{
                        fontSize: '1.1rem',
                        lineHeight: 1.6,
                        color: 'hsl(var(--txt-primary))',
                        margin: '0 0 1.5rem 0',
                        fontWeight: 500
                    }}>
                        {overall.verdict}
                    </p>

                    <div style={{ display: 'flex', gap: '2rem' }}>
                        <div>
                            <div className="text-muted" style={{ fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Symmetry Index</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{overall.symmetryScore}%</div>
                        </div>
                        <div>
                            <div className="text-muted" style={{ fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Proportion Ratio</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{proportions.phiRatio}</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Feature Exploration */}
            <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '2rem' }} className="mobile-stack">
                {/* Sidebar Navigation */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {sections.map(section => (
                        <button
                            key={section.id}
                            onClick={() => setActiveTab(section.id)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '1rem',
                                padding: '1rem 1.25rem',
                                borderRadius: 'var(--radius-md)',
                                background: activeTab === section.id ? 'hsla(var(--accent-primary), 0.15)' : 'transparent',
                                border: activeTab === section.id ? '1px solid hsla(var(--accent-primary), 0.3)' : '1px solid transparent',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                                textAlign: 'left',
                                width: '100%',
                                color: activeTab === section.id ? 'hsl(var(--accent-primary))' : 'inherit'
                            }}
                        >
                            <section.icon size={20} />
                            <span style={{ fontWeight: 600, flex: 1 }}>{section.label}</span>
                            {activeTab === section.id && <ChevronRight size={16} />}
                        </button>
                    ))}
                </div>

                {/* Content Area */}
                <div className="card glass" style={{ padding: '2rem' }}>
                    {activeTab === 'eyes' && <FeatureCard feature="eyes" data={eyes} />}
                    {activeTab === 'nose' && <FeatureCard feature="nose" data={nose} />}
                    {activeTab === 'jawline' && <FeatureCard feature="jawline" data={jawline} />}
                    {activeTab === 'proportions' && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                <Layout size={24} color="hsl(var(--accent-primary))" />
                                <h3 style={{ margin: 0 }}>Neoclassical Proportions</h3>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                                <div className="card" style={{ background: 'hsla(var(--bg-panel), 0.5)', padding: '1.5rem' }}>
                                    <div className="text-muted" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>Phi Ratio (Length/Width)</div>
                                    <div style={{ fontSize: '2rem', fontWeight: 800 }}>{proportions.phiRatio}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'hsl(var(--accent-primary))' }}>Ideal: 1.618</div>
                                </div>
                                <div className="card" style={{ background: 'hsla(var(--bg-panel), 0.5)', padding: '1.5rem' }}>
                                    <div className="text-muted" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>fWHR (Compactness)</div>
                                    <div style={{ fontSize: '2rem', fontWeight: 800 }}>{proportions.fWHR}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'hsl(var(--accent-primary))' }}>Ideal: 1.8 - 2.0</div>
                                </div>
                            </div>

                            <div style={{ marginTop: '1rem' }}>
                                <h4 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: 'hsl(var(--txt-secondary))', marginBottom: '1rem' }}>Vertical Thirds Balance</h4>
                                <div style={{ display: 'flex', height: '140px', gap: '8px' }}>
                                    <ThirdBar label="Upper (Hairline)" pct={proportions.thirds.upper} color="#666" />
                                    <ThirdBar label="Mid (Brow-Nose)" pct={proportions.thirds.mid} color="hsl(var(--accent-primary))" />
                                    <ThirdBar label="Lower (Nose-Chin)" pct={proportions.thirds.lower} color="#666" />
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem', fontSize: '0.85rem' }} className="text-muted">
                                    <span>{proportions.thirds.upper}%</span>
                                    <span>{proportions.thirds.mid}%</span>
                                    <span>{proportions.thirds.lower}%</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Advanced Raw Data (Collapsible) */}
            <div style={{ marginTop: '3rem', borderTop: '1px solid hsl(var(--border-subtle))', paddingTop: '1rem' }}>
                <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    style={{ background: 'transparent', border: 'none', color: 'hsl(var(--txt-secondary))', fontSize: '0.9rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                >
                    {showAdvanced ? 'Hide' : 'Show'} Full Morphometric Specs
                </button>
                {showAdvanced && (
                    <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
                        {Object.entries(detailed).map(([key, value]) => (
                            <div key={key} style={{ background: 'hsla(var(--bg-panel), 0.3)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)' }}>
                                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'hsl(var(--txt-secondary))', marginBottom: '0.25rem' }}>{key}</div>
                                <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>{value}</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

function ThirdBar({ label, pct, color }) {
    return (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{
                flex: 1,
                background: 'hsla(var(--bg-panel), 0.5)',
                borderRadius: 'var(--radius-sm)',
                position: 'relative',
                overflow: 'hidden'
            }}>
                <div style={{
                    position: 'absolute',
                    bottom: 0, left: 0, width: '100%',
                    height: `${pct * 2.5}%`, // Scaling for visual impact
                    background: `linear-gradient(to top, ${color}, transparent)`,
                    opacity: 0.8,
                    transition: 'height 1s ease-out'
                }} />
            </div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, textAlign: 'center', color }}>{label.split(' ')[0]}</div>
        </div>
    );
}

