import { Ruler, Scan, Activity, Eye, LayoutTemplate, Sparkles, Scale } from 'lucide-react';

export default function MorphologyReport({ data }) {
    if (!data) return null;

    const { thirds } = data;
    const f = (n) => typeof n === 'number' ? n.toFixed(2) : n;

    return (
        <div className="card glass fade-in" style={{ marginTop: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem', borderBottom: '1px solid hsl(var(--border-subtle))', paddingBottom: '1rem' }}>
                <Scan size={24} color="hsl(var(--accent-primary))" />
                <div>
                    <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Scientific Morphology Report</h3>
                    <p className="text-muted" style={{ margin: 0, fontSize: '0.85rem' }}>Objectively measured against Neoclassical Canons.</p>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>

                {/* GOLDEN RATIO & HARMONY */}
                <div style={{ background: 'hsla(var(--bg-panel), 0.3)', padding: '1.5rem', borderRadius: 'var(--radius-sm)' }}>
                    <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: 0, color: 'hsl(var(--accent-primary))' }}>
                        <Scale size={18} /> Golden Ratio & Harmony
                    </h4>

                    <ComparisonRow
                        label="Facial Ratio (Phi)"
                        value={data.phiRatio}
                        ideal="1.618"
                        desc="Length / Width"
                    />
                    <ComparisonRow
                        label="Eye Spacing (5ths)"
                        value={data.eyeSpacingRatio}
                        ideal="1.0"
                        desc="Intercanthal / Eye Width"
                    />
                    <ComparisonRow
                        label="Mouth-Nose Ratio"
                        value={data.mouthNoseRatio}
                        ideal="1.618"
                        desc="Mouth Width / Nose Width"
                    />
                </div>

                {/* VERTICAL THIRDS */}
                <div style={{ background: 'hsla(var(--bg-panel), 0.3)', padding: '1.5rem', borderRadius: 'var(--radius-sm)' }}>
                    <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: 0, color: 'hsl(var(--accent-primary))' }}>
                        <LayoutTemplate size={18} /> Rule of Thirds
                    </h4>

                    <div style={{ display: 'flex', height: '120px', gap: '4px', marginTop: '1rem', marginBottom: '1rem' }}>
                        <ThirdBar label="Upper (Hairline)" pct={thirds?.upper || 33.3} color="#888" />
                        <ThirdBar label="Mid (Brow-Nose)" pct={thirds?.mid || 33.3} color="hsl(var(--accent-primary))" />
                        <ThirdBar label="Lower (Nose-Chin)" pct={thirds?.lower || 33.3} color="#888" />
                    </div>
                    <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'hsl(var(--txt-secondary))' }}>
                        Ideal: 33.3% / 33.3% / 33.3%
                    </div>
                </div>

                {/* STRUCTURAL */}
                <div style={{ background: 'hsla(var(--bg-panel), 0.3)', padding: '1.5rem', borderRadius: 'var(--radius-sm)' }}>
                    <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: 0, color: 'hsl(var(--accent-primary))' }}>
                        <Ruler size={18} /> Structural Features
                    </h4>
                    <div className="metric-row">
                        <span>fWHR (Width/Height)</span>
                        <strong>{f(data.fWHR)}</strong>
                    </div>
                    <div className="metric-row">
                        <span>Jaw-Cheek Ratio</span>
                        <strong>{f(data.jawToCheek)}</strong>
                    </div>
                    <div className="metric-row">
                        <span>Canthal Tilt</span>
                        <strong>{f(data.canthalTilt)}°</strong>
                    </div>
                </div>
            </div>
        </div>
    );
}

function ComparisonRow({ label, value, ideal, desc }) {
    const val = parseFloat(value);
    const idl = parseFloat(ideal);
    const diff = Math.abs(val - idl);
    // 5% tolerance is "Perfect", 10% "Good"
    const isPerfect = diff / idl < 0.05;
    const isGood = diff / idl < 0.15;

    return (
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid hsla(255,255,255,0.05)', paddingBottom: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ fontWeight: 500 }}>{label}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.8rem', color: 'hsl(var(--txt-secondary))' }}>Ideal: {ideal}</span>
                    <strong style={{ color: isPerfect ? 'hsl(var(--accent-primary))' : isGood ? '#fff' : '#888' }}>
                        {value}
                    </strong>
                </div>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'hsl(var(--txt-secondary))' }}>{desc}</div>
        </div>
    )
}

function ThirdBar({ label, pct, color }) {
    if (!pct) return null;
    return (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '4px' }}>{pct.toFixed(1)}%</span>
            <div style={{ width: '100%', height: `${pct}%`, background: color, borderRadius: '4px', opacity: 0.8, minHeight: '20%' }}></div>
            <span style={{ fontSize: '0.7rem', color: 'hsl(var(--txt-secondary))', marginTop: '4px' }}>{label}</span>
        </div>
    )
}
