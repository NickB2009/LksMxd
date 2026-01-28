import { useState } from 'react';
import { Activity, Layout, GitMerge, Zap, Scale, ChevronDown, ChevronUp } from 'lucide-react';
import ScoreGauge from './ScoreGauge';

export default function PSLReport({ data }) {
    const [showDetails, setShowDetails] = useState(false);

    if (!data || !data.harmony) return null;

    const { harmony, features } = data;
    const { score, explanations } = harmony;

    // Helpers for display
    const formatPct = (val) => Math.round(val * 100);
    const formatScore = (val) => Math.round(val);

    return (
        <div className="fade-in" style={{ marginTop: '2rem' }}>
            {/* Top Card: Harmony Score & Verdict */}
            <div className="card glass" style={{
                padding: '3rem',
                marginBottom: '2rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: '1.5rem',
                background: 'linear-gradient(135deg, hsla(var(--bg-panel), 0.5), hsla(var(--accent-primary), 0.05))'
            }}>
                <ScoreGauge value={score} ideal={80} label="Facial Harmony" size={200} />

                <div style={{ textAlign: 'center', maxWidth: '600px' }}>
                    <h2 style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>
                        {score >= 80 ? 'Exceptional Geometric Harmony' :
                            score >= 60 ? 'High Geometric Harmony' :
                                score >= 40 ? 'Moderate Harmony' : 'Developing Harmony'}
                    </h2>
                    <p className="text-muted">
                        Probabilistic inference based on {Object.keys(features).length} normalized geometric features.
                    </p>
                </div>
            </div>

            {/* Explanation Layer: Contributors */}
            <div className="grid-stack" style={{ gap: '2rem', marginBottom: '2rem' }}>
                <div className="card glass" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                        <Zap size={20} color="hsl(var(--accent-primary))" />
                        <h3 style={{ margin: 0 }}>Harmony Contributors</h3>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <ContributorRow label="Symmetry" value={explanations.symmetry} icon={GitMerge}
                            desc="Left-Right structural alignment" />
                        <ContributorRow label="Proportions" value={explanations.proportions} icon={Layout}
                            desc="Relationships between facial thirds and fifths" />
                        <ContributorRow label="Golden Ratio" value={explanations.golden_ratio} icon={Scale}
                            desc="Adherence to divine proportion (1.618)" />
                        <ContributorRow label="Angular Harmony" value={explanations.angles} icon={Activity}
                            desc="Alignment of jaw, brow, and canthal tilts" />
                        <ContributorRow label="Balance" value={explanations.balance} icon={Scale}
                            desc="Distribution of facial mass" />
                    </div>
                </div>
            </div>

            {/* Detailed Normalized Features (Collapsible) */}
            <div className="card glass" style={{ padding: '0' }}>
                <button
                    onClick={() => setShowDetails(!showDetails)}
                    style={{
                        width: '100%',
                        padding: '1.5rem 2rem',
                        background: 'transparent',
                        border: 'none',
                        color: 'inherit',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        cursor: 'pointer',
                        fontSize: '1rem',
                        fontWeight: 500
                    }}
                >
                    <span>Raw Geometric Features (Normalized 0.0 - 1.0)</span>
                    {showDetails ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </button>

                {showDetails && (
                    <div style={{ padding: '0 2rem 2rem 2rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem', borderTop: '1px solid hsl(var(--border-subtle))', paddingTop: '1.5rem' }}>
                        {Object.entries(features).map(([key, val]) => (
                            <div key={key} style={{ background: 'hsla(var(--bg-app), 0.5)', padding: '1rem', borderRadius: 'var(--radius-sm)' }}>
                                <div style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: 'hsl(var(--txt-secondary))', marginBottom: '0.25rem' }}>
                                    {key.replace('_', ' ')}
                                </div>
                                <div style={{ fontSize: '1.25rem', fontWeight: 'bold', fontFamily: 'monospace', color: val > 0.8 ? 'hsl(var(--accent-primary))' : 'inherit' }}>
                                    {val.toFixed(3)}
                                </div>
                                <div style={{
                                    width: '100%', height: '4px', background: 'hsl(var(--bg-panel))',
                                    marginTop: '0.5rem', borderRadius: '2px', overflow: 'hidden'
                                }}>
                                    <div style={{ width: `${val * 100}%`, height: '100%', background: 'hsl(var(--accent-primary))' }} />
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

function ContributorRow({ label, value, icon: Icon, desc }) {
    // Value here is a weighted contribution score roughly proportional to importance
    // We can normalize it for display or just show the bar relative to max possible ~3.0
    // Let's assume max reasonable contribution is around 2.5-3.0 for display scaling
    const barWidth = Math.min(100, Math.max(5, (value / 2.5) * 100));

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 100px', alignItems: 'center', gap: '1rem' }}>
            <div style={{
                width: '40px', height: '40px', borderRadius: '8px',
                background: 'hsla(var(--bg-panel), 0.5)',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
                <Icon size={20} className="text-muted" />
            </div>

            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                    <span style={{ fontWeight: 600 }}>{label}</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'hsl(var(--txt-secondary))', marginBottom: '0.5rem' }}>
                    {desc}
                </div>
                <div style={{
                    width: '100%', height: '6px', background: 'hsla(var(--bg-panel), 0.8)',
                    borderRadius: '3px', overflow: 'hidden'
                }}>
                    <div style={{
                        width: `${barWidth}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, hsl(var(--accent-primary)), hsl(280, 80%, 60%))',
                        borderRadius: '3px'
                    }} />
                </div>
            </div>

            <div style={{ textAlign: 'right', fontWeight: 600, fontSize: '1.1rem' }}>
                +{value.toFixed(2)}
            </div>
        </div>
    )
}
