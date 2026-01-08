import { TrendingUp, Fingerprint, Layers, Sparkles } from 'lucide-react';

export default function MarketFitReport({ rarity, marketFit, potential }) {
    if (!rarity || !marketFit) return null;

    return (
        <div className="card glass fade-in" style={{ marginTop: '2rem', animationDelay: '0.4s' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', borderBottom: '1px solid hsl(var(--border-subtle))', paddingBottom: '1rem' }}>
                <TrendingUp size={20} color="hsl(var(--accent-primary))" />
                <h3 style={{ margin: 0, fontSize: '1.25rem' }}>Strategic Market Analysis</h3>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                {/* Rarity Engine Output */}
                <div>
                    <h4 style={{ color: 'hsl(var(--txt-secondary))', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Fingerprint size={16} /> Rarity Index
                    </h4>

                    <div style={{ background: 'hsla(var(--bg-app), 0.5)', padding: '1.5rem', borderRadius: 'var(--radius-md)', textAlign: 'center', border: '1px solid hsl(var(--border-subtle))' }}>
                        <div style={{ fontSize: '3rem', fontWeight: 800, lineHeight: 1, marginBottom: '0.5rem', background: 'linear-gradient(to right, #fff, #aaa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            {rarity.score}<span style={{ fontSize: '1.5rem', opacity: 0.5 }}>/10</span>
                        </div>
                        <div style={{ fontSize: '1.1rem', color: 'hsl(var(--accent-primary))', fontWeight: 500, marginBottom: '0.5rem' }}>
                            {rarity.label}
                        </div>

                        <div style={{ textAlign: 'left', marginTop: '1.5rem', fontSize: '0.9rem' }}>
                            {Object.entries(rarity.details).map(([trait, data]) => (
                                <div key={trait} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <span className="text-muted" style={{ textTransform: 'capitalize' }}>{trait.replace(/([A-Z])/g, ' $1').trim()}</span>
                                    <span style={{
                                        color: parseFloat(data.rarity) > 5 ? 'hsl(var(--accent-primary))' : 'inherit',
                                        fontWeight: parseFloat(data.rarity) > 5 ? 600 : 400
                                    }}>
                                        {data.label} ({data.rarity})
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Market Fit Output */}
                <div>
                    <h4 style={{ color: 'hsl(var(--txt-secondary))', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Layers size={16} /> Market Suitability
                    </h4>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {Object.entries(marketFit).map(([market, data]) => {
                            const pot = potential && potential[market];
                            return (
                                <MarketBar
                                    key={market}
                                    market={market}
                                    score={data.score}
                                    description={data.description}
                                    potential={pot ? pot.potential : null}
                                />
                            );
                        })}
                    </div>

                    {potential && (
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'hsla(var(--accent-primary), 0.1)', borderRadius: 'var(--radius-sm)', display: 'flex', gap: '0.75rem', border: '1px solid hsla(var(--accent-primary), 0.2)' }}>
                            <Sparkles size={18} color="hsl(var(--accent-primary))" style={{ flexShrink: 0, marginTop: '2px' }} />
                            <div>
                                <h5 style={{ margin: '0 0 0.25rem 0', fontSize: '0.95rem', fontWeight: 600 }}>Leanness Projection</h5>
                                <p style={{ margin: 0, fontSize: '0.85rem', color: 'hsl(var(--txt-secondary))', lineHeight: 1.4 }}>
                                    Simulated scores based on reduced facial puffiness (10-15% increase in jaw definition).
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function MarketBar({ market, score, description, potential }) {
    const isHighFit = score > 80;

    return (
        <div style={{ background: 'hsla(var(--bg-panel), 0.3)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: isHighFit ? '1px solid hsla(var(--accent-primary), 0.5)' : '1px solid transparent' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ textTransform: 'capitalize', fontWeight: 600, fontSize: '1rem' }}>{market}</span>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                    {potential && potential > score && (
                        <span style={{ fontSize: '0.85rem', color: 'hsl(var(--accent-primary))', fontWeight: 500 }}>
                            ↑ {potential}%
                        </span>
                    )}
                    <span style={{ fontWeight: 700, color: isHighFit ? 'hsl(var(--accent-primary))' : 'hsl(var(--txt-secondary))' }}>
                        {score}%
                    </span>
                </div>
            </div>

            {/* Bar */}
            <div style={{ height: '6px', background: 'hsla(var(--bg-app), 0.8)', borderRadius: 'var(--radius-full)', overflow: 'hidden', marginBottom: '0.5rem', position: 'relative' }}>
                {/* Potential Bar (Ghost) */}
                {potential && (
                    <div style={{
                        position: 'absolute',
                        top: 0, left: 0, height: '100%',
                        width: `${potential}%`,
                        background: 'hsla(var(--accent-primary), 0.3)',
                        borderRadius: 'var(--radius-full)',
                    }}></div>
                )}

                {/* Actual Score Bar */}
                <div style={{
                    height: '100%',
                    width: `${score}%`,
                    background: isHighFit
                        ? 'linear-gradient(90deg, hsl(var(--accent-primary)), #fff)'
                        : 'hsl(var(--txt-secondary))',
                    borderRadius: 'var(--radius-full)',
                    position: 'relative',
                    zIndex: 2
                }}></div>
            </div>

            <p style={{ fontSize: '0.8rem', color: 'hsl(var(--txt-secondary))', margin: 0 }}>
                {description}
            </p>
        </div>
    );
}
