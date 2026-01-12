import { useState } from 'react';
import { Scan, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import FeatureCard from './FeatureCard';
import ScoreGauge from './ScoreGauge';

export default function MorphologyReport({ data }) {
    const [showDetails, setShowDetails] = useState(false);

    if (!data) return null;

    // Calculate overall score based on key metrics with harmony weighting
    const calculateOverallScore = () => {
        const scores = [];
        const weights = [];

        // Symmetry (25% weight)
        if (data.detailed?.['Symmetry Index']) {
            const symScore = parseFloat(data.detailed['Symmetry Index']);
            scores.push(symScore);
            weights.push(0.25);
        }

        // Golden Ratio Adherence (20% weight)
        if (data.phiRatio) {
            const phiDiff = Math.abs(data.phiRatio - 1.618);
            const phiTolerance = 1.618 * 0.20; // 20% tolerance
            const phiScore = 100 * Math.exp(-Math.pow(phiDiff / phiTolerance, 2));
            scores.push(phiScore);
            weights.push(0.20);
        }

        // fWHR (15% weight) - Elite range: 1.8-2.2
        if (data.fWHR) {
            const fwhrDiff = Math.min(Math.abs(data.fWHR - 1.9), Math.abs(data.fWHR - 2.0));
            const fwhrTolerance = 1.9 * 0.25;
            const fwhrScore = 100 * Math.exp(-Math.pow(fwhrDiff / fwhrTolerance, 2));
            scores.push(fwhrScore);
            weights.push(0.15);
        }

        // Canthal Tilt (15% weight) - Elite: 7-10°
        if (data.canthalTilt) {
            const tiltDiff = Math.abs(data.canthalTilt - 8.0);
            const tiltTolerance = 8.0 * 0.40;
            const tiltScore = 100 * Math.exp(-Math.pow(tiltDiff / tiltTolerance, 2));
            scores.push(tiltScore);
            weights.push(0.15);
        }

        // Jaw Definition (15% weight) - Elite: 0.88-0.95
        if (data.jawToCheek) {
            const jawDiff = Math.abs(data.jawToCheek - 0.90);
            const jawTolerance = 0.90 * 0.30;
            const jawScore = 100 * Math.exp(-Math.pow(jawDiff / jawTolerance, 2));
            scores.push(jawScore);
            weights.push(0.15);
        }

        // Midface Ratio (10% weight) - Elite: 0.97-1.11
        if (data.psl?.midface_ratio_psl) {
            const midfaceDiff = Math.abs(data.psl.midface_ratio_psl - 1.0);
            const midfaceTolerance = 1.0 * 0.20;
            const midfaceScore = 100 * Math.exp(-Math.pow(midfaceDiff / midfaceTolerance, 2));
            scores.push(midfaceScore);
            weights.push(0.10);
        }

        // Weighted average
        if (scores.length === 0) return 70;

        const totalWeight = weights.reduce((a, b) => a + b, 0);
        const weightedSum = scores.reduce((sum, score, i) => sum + score * weights[i], 0);
        return Math.round(weightedSum / totalWeight);
    };

    const overallScore = calculateOverallScore();

    const getScoreLabel = (score) => {
        if (score >= 80) return { label: 'Exceptional', color: 'hsl(120, 70%, 50%)' };
        if (score >= 70) return { label: 'Above Average', color: 'hsl(90, 70%, 50%)' };
        if (score >= 60) return { label: 'Good', color: 'hsl(60, 70%, 50%)' };
        if (score >= 50) return { label: 'Average', color: 'hsl(30, 70%, 50%)' };
        return { label: 'Below Average', color: 'hsl(0, 70%, 50%)' };
    };

    const scoreInfo = getScoreLabel(overallScore);

    return (
        <div className="fade-in" style={{ marginTop: '2rem', animationDelay: '0.2s' }}>
            {/* At-a-Glance Summary */}
            <div className="card glass" style={{
                background: 'linear-gradient(135deg, hsla(var(--accent-primary), 0.1), hsla(var(--bg-panel), 0.3))',
                padding: '2rem',
                marginBottom: '1.5rem',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid hsla(var(--accent-primary), 0.2)'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                    <Sparkles size={28} color="hsl(var(--accent-primary))" />
                    <h3 style={{ margin: 0, fontSize: '1.5rem' }}>Overall Assessment</h3>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', flexWrap: 'wrap' }}>
                    {/* Score Circle */}
                    <div style={{ flex: '0 0 auto' }}>
                        <ScoreGauge
                            value={overallScore}
                            ideal={80}
                            label="Overall Score"
                            size={140}
                        />
                    </div>

                    {/* Top Strengths */}
                    <div style={{ flex: '1 1 300px' }}>
                        <h4 style={{
                            fontSize: '0.95rem',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                            color: 'hsl(var(--txt-secondary))',
                            marginBottom: '0.75rem'
                        }}>
                            Key Strengths
                        </h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {data.detailed?.['Symmetry Index'] && parseFloat(data.detailed['Symmetry Index']) > 90 && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{
                                        width: '6px',
                                        height: '6px',
                                        borderRadius: '50%',
                                        background: 'hsl(var(--accent-primary))'
                                    }} />
                                    <span style={{ fontSize: '0.95rem' }}>High Facial Symmetry ({data.detailed['Symmetry Index']})</span>
                                </div>
                            )}
                            {data.psl?.eye_size_ratio > 0.017 && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{
                                        width: '6px',
                                        height: '6px',
                                        borderRadius: '50%',
                                        background: 'hsl(var(--accent-primary))'
                                    }} />
                                    <span style={{ fontSize: '0.95rem' }}>Large, Prominent Eyes</span>
                                </div>
                            )}
                            {data.jawToCheek > 0.85 && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{
                                        width: '6px',
                                        height: '6px',
                                        borderRadius: '50%',
                                        background: 'hsl(var(--accent-primary))'
                                    }} />
                                    <span style={{ fontSize: '0.95rem' }}>Strong Jaw Definition</span>
                                </div>
                            )}
                            {data.canthalTilt > 4 && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <div style={{
                                        width: '6px',
                                        height: '6px',
                                        borderRadius: '50%',
                                        background: 'hsl(var(--accent-primary))'
                                    }} />
                                    <span style={{ fontSize: '0.95rem' }}>Positive Canthal Tilt</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Feature Cards */}
            <div className="card glass" style={{ padding: '2rem' }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    marginBottom: '2rem',
                    paddingBottom: '1rem',
                    borderBottom: '1px solid hsl(var(--border-subtle))'
                }}>
                    <Scan size={24} color="hsl(var(--accent-primary))" />
                    <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Feature Analysis</h3>
                </div>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                    gap: '1.5rem'
                }}>
                    <FeatureCard feature="eyes" data={data} />
                    <FeatureCard feature="nose" data={data} />
                    <FeatureCard feature="jaw" data={data} />
                </div>

                {/* Toggle Advanced Details */}
                <div style={{ marginTop: '2rem', textAlign: 'center' }}>
                    <button
                        onClick={() => setShowDetails(!showDetails)}
                        className="btn"
                        style={{
                            background: 'transparent',
                            border: '1px solid hsl(var(--border-subtle))',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.5rem'
                        }}
                    >
                        {showDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                        {showDetails ? 'Hide' : 'Show'} Advanced Metrics
                    </button>
                </div>

                {/* Collapsible Advanced Section */}
                {showDetails && (
                    <div style={{
                        marginTop: '2rem',
                        padding: '1.5rem',
                        background: 'hsla(var(--bg-app), 0.5)',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid hsla(255,255,255,0.05)'
                    }}>
                        <h4 style={{
                            fontSize: '0.95rem',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                            color: 'hsl(var(--txt-secondary))',
                            marginBottom: '1rem'
                        }}>
                            All Measurements
                        </h4>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                            gap: '0.75rem',
                            fontSize: '0.9rem'
                        }}>
                            {data.detailed && Object.entries(data.detailed).map(([key, value]) => (
                                <div key={key} style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    padding: '0.5rem',
                                    background: 'hsla(var(--bg-panel), 0.3)',
                                    borderRadius: 'var(--radius-sm)'
                                }}>
                                    <span className="text-muted" style={{ fontSize: '0.85rem' }}>
                                        {key}
                                    </span>
                                    <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                        {value}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
