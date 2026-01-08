import { Table, ClipboardList, Microchip, FileText } from 'lucide-react';

export default function DetailedAnalysis({ data }) {
    if (!data || !data.detailed) return null;

    const { detailed, verdict } = data;

    return (
        <div className="card glass fade-in" style={{ marginTop: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', borderBottom: '1px solid hsl(var(--border-subtle))', paddingBottom: '1rem' }}>
                <ClipboardList size={20} color="hsl(var(--accent-primary))" />
                <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Full Morphometric Specs</h3>
            </div>

            {/* VERDICT BLOCK */}
            <div style={{ marginBottom: '2rem', background: 'hsla(var(--accent-primary), 0.05)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid hsl(var(--accent-primary))' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'hsl(var(--accent-primary))', fontWeight: 600 }}>
                    <Microchip size={16} /> AI Verdict
                </div>
                <p style={{ margin: 0, lineHeight: 1.6, fontSize: '0.95rem' }}>
                    {verdict}
                </p>
            </div>

            {/* DENSE GRID */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                {Object.entries(detailed).map(([key, value]) => (
                    <div key={key} style={{
                        background: 'hsla(var(--bg-panel), 0.3)',
                        padding: '0.75rem',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid hsla(255,255,255,0.05)'
                    }}>
                        <div style={{ fontSize: '0.75rem', color: 'hsl(var(--txt-secondary))', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
                            {key}
                        </div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 600, fontFamily: 'monospace' }}>
                            {value}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
