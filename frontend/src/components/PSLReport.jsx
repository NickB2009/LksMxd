import { useState } from 'react';
import { Activity, Layout, GitMerge, Zap, Scale, ChevronDown, ChevronUp, ShieldAlert, Sparkles, Binary, Link as LinkIcon } from 'lucide-react';
import ScoreGauge from './ScoreGauge';

export default function PSLReport({ data }) {
    const [showDetails, setShowDetails] = useState(false);

    if (!data || !data.harmony) return null;

    const { harmony, features } = data;
    const { score, explanations, dual } = harmony;

    // Helpers for display
    const formatPct = (val) => Math.round(val * 100);
    const formatScore = (val) => Math.round(val);

    return (
        <div className="fade-in" style={{ marginTop: '2rem' }}>
            {/* Top Section: Dual Scores & Composite */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }} className="mobile-stack">

                {/* Left: Component Scores */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div className="card glass" style={{ padding: '1.5rem', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'hsl(210, 80%, 60%)' }}>
                            <Layout size={20} />
                            <span style={{ fontWeight: 600, textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>Natural Harmony</span>
                        </div>
                        <div style={{ fontSize: '2.5rem', fontWeight: 800 }}>{dual.natural}</div>
                        <div className="text-muted" style={{ fontSize: '0.9rem' }}>Alignment & Symmetry</div>
                        {dual.natural >= 80 && (
                            <span className="badge" style={{ marginTop: '0.5rem', alignSelf: 'flex-start', background: 'hsl(210, 80%, 60%)', color: 'white' }}>Natural Beauty</span>
                        )}
                    </div>

                    <div className="card glass" style={{ padding: '1.5rem', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'hsl(280, 80%, 60%)' }}>
                            <Sparkles size={20} />
                            <span style={{ fontWeight: 600, textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>Expressive Harmony</span>
                        </div>
                        <div style={{ fontSize: '2.5rem', fontWeight: 800 }}>{dual.expressive}</div>
                        <div className="text-muted" style={{ fontSize: '0.9rem' }}>Coherent Distinctiveness</div>
                        {dual.expressive >= 80 && (
                            <span className="badge" style={{ marginTop: '0.5rem', alignSelf: 'flex-start', background: 'hsl(280, 80%, 60%)', color: 'white' }}>Model Tier</span>
                        )}
                    </div>
                </div>

                {/* Right: Composite Gauge */}
                <div className="card glass" style={{
                    padding: '2rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexDirection: 'column',
                    background: 'linear-gradient(135deg, hsla(var(--bg-panel), 0.5), hsla(var(--accent-primary), 0.05))'
                }}>
                    <ScoreGauge value={score} ideal={85} label="Global Composite" size={180} />
                    <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                        <div style={{ fontWeight: 600, fontSize: '1.2rem' }}>
                            {score >= 85 ? 'Exceptional' :
                                score >= 70 ? 'High' :
                                    score >= 50 ? 'Moderate' : 'Low'}
                        </div>
                    </div>
                </div>
            </div>

            {/* Explanation Layer: The 3 Forces */}
            <div className="grid-stack" style={{ gap: '2rem', marginBottom: '2rem' }}>
                <div className="card glass" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                        <Zap size={20} color="hsl(var(--accent-primary))" />
                        <h3 style={{ margin: 0 }}>Metric Contributions</h3>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <ContributorRow
                            label="Natural Harmony"
                            value={explanations["Natural Harmony"]}
                            icon={Layout}
                            color="hsl(210, 80%, 60%)"
                            desc="Recursive alignment with neoclassical norms & symmetry."
                        />
                        <ContributorRow
                            label="Expressive Bonus"
                            value={explanations["Expressive Bonus"]}
                            icon={Sparkles}
                            color="hsl(280, 80%, 60%)"
                            desc="Reward for structured, coherent distinctiveness (Model Tier)."
                        />
                        <ContributorRow
                            label="Coherence Gate"
                            value={explanations["Coherence Factor"]}
                            icon={LinkIcon}
                            color="hsl(160, 80%, 40%)"
                            desc="Multiplier enforcing agreement between Harmony and Expressiveness."
                        />
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

function ContributorRow({ label, value, icon: Icon, desc, color }) {
    // Value represents relative "Push". 
    // We scale it for visual impact. 
    // Max push is roughly weighted (e.g. 9.0), so value might be around 2-3.
    // Display as a relative bar.

    // Handle negative chaos
    const isNegative = value < 0;
    const absVal = Math.abs(value);
    const barWidth = Math.min(100, Math.max(5, (absVal / 2.0) * 100));

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 100px', alignItems: 'center', gap: '1rem' }}>
            <div style={{
                width: '40px', height: '40px', borderRadius: '8px',
                background: 'hsla(var(--bg-panel), 0.5)',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
                <Icon size={20} style={{ color: color || 'inherit' }} />
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
                        background: color || 'hsl(var(--accent-primary))',
                        borderRadius: '3px',
                        opacity: 0.8
                    }} />
                </div>
            </div>

            <div style={{ textAlign: 'right', fontWeight: 600, fontSize: '1.1rem', color: isNegative ? 'hsl(0, 80%, 60%)' : 'inherit' }}>
                {isNegative ? '' : '+'}{value.toFixed(2)}
            </div>
        </div>
    )
}
