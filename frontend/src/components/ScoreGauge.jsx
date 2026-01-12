export default function ScoreGauge({ value, ideal, label, size = 120 }) {
    const val = parseFloat(value) || 0;
    const idl = parseFloat(ideal) || 1;

    // Metric-specific tolerance
    const getToleranceMultiplier = (l) => {
        const tolerances = {
            'Symmetry': 0.1,         // Very tight (needs >95% for top score)
            'Harmony Score': 0.15,   // Tight
            'Canthal Tilt': 0.40,
            'Gonial Angle': 0.15,
            'Phi Ratio': 0.15,
            'ESR': 0.20
        };
        return tolerances[l] || 0.25;
    };

    const toleranceMultiplier = getToleranceMultiplier(label);
    const tolerance = idl * toleranceMultiplier;

    // Gaussian curve: 100 * exp(-(diff/tolerance)^2)
    const diff = Math.abs(val - idl);
    const gaussianScore = 100 * Math.exp(-Math.pow(diff / tolerance, 2));
    const score = Math.max(0, Math.min(100, gaussianScore));

    const getColor = () => {
        if (score >= 90) return 'hsl(120, 70%, 55%)'; // Elite
        if (score >= 80) return 'hsl(100, 70%, 50%)'; // Exceptional
        if (score >= 70) return 'hsl(80, 70%, 50%)';  // Above Average
        if (score >= 60) return 'hsl(60, 70%, 50%)';  // Good
        if (score >= 50) return 'hsl(30, 70%, 50%)';  // Average
        return 'hsl(0, 70%, 50%)'; // Red - Below Average
    };

    const getLabel = () => {
        if (score >= 90) return 'Elite';
        if (score >= 80) return 'Exceptional';
        if (score >= 70) return 'Above Average';
        if (score >= 60) return 'Good';
        if (score >= 50) return 'Average';
        return 'Below Average';
    };

    const color = getColor();
    const radius = size / 2 - 10;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.5rem'
        }}>
            <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="hsla(255, 255, 255, 0.05)"
                    strokeWidth="6"
                />
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="6"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)', filter: `drop-shadow(0 0 6px ${color}40)` }}
                />
                <text
                    x={size / 2}
                    y={size / 2}
                    textAnchor="middle"
                    dy="0.3em"
                    style={{
                        transform: 'rotate(90deg)',
                        transformOrigin: 'center',
                        fontSize: size > 120 ? '1.75rem' : '1.25rem',
                        fontWeight: 800,
                        fill: 'white',
                        fontFamily: 'Inter, sans-serif'
                    }}
                >
                    {Math.round(score)}
                </text>
            </svg>
            <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: color, marginBottom: '2px' }}>
                    {getLabel()}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'hsl(var(--txt-secondary))', fontWeight: 500 }}>
                    {label}
                </div>
            </div>
        </div>
    );
}
