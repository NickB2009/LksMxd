export default function ScoreGauge({ value, ideal, label, size = 120 }) {
    const val = parseFloat(value) || 0;
    const idl = parseFloat(ideal) || 1;

    // Metric-specific tolerance (based on real supermodel variance)
    const getToleranceMultiplier = (label) => {
        const tolerances = {
            'Tilt': 0.40,           // Canthal tilt varies widely (4-10°)
            'Definition': 0.30,     // Jaw definition (0.85-1.0+)
            'Length/Width': 0.25,   // Nose L/W (1.4-1.8)
            'Size': 0.25,           // Eye size (1.5-2.0%)
            'Spacing': 0.20,        // ESR is more consistent (0.44-0.47)
            'Lower Third': 0.15,    // Lower third is fairly consistent
            'Mouth Ratio': 0.20     // Golden ratio adherence
        };
        return tolerances[label] || 0.25; // Default 25%
    };

    const toleranceMultiplier = getToleranceMultiplier(label);
    const tolerance = idl * toleranceMultiplier;

    // Use Gaussian curve for smoother degradation
    // Elite faces can deviate but still score high
    const diff = Math.abs(val - idl);
    const gaussianScore = 100 * Math.exp(-Math.pow(diff / tolerance, 2));
    const score = Math.max(0, Math.min(100, gaussianScore));

    // Recalibrated thresholds (easier to reach "Exceptional")
    const getColor = () => {
        if (score >= 80) return 'hsl(120, 70%, 50%)'; // Green - Exceptional
        if (score >= 70) return 'hsl(90, 70%, 50%)';  // Yellow-green - Above Average
        if (score >= 60) return 'hsl(60, 70%, 50%)';  // Yellow - Good
        if (score >= 50) return 'hsl(30, 70%, 50%)';  // Orange - Average
        return 'hsl(0, 70%, 50%)'; // Red - Below Average
    };

    const getLabel = () => {
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
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="hsla(255, 255, 255, 0.1)"
                    strokeWidth="8"
                />
                {/* Progress circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="8"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dashoffset 0.5s ease' }}
                />
                {/* Center text */}
                <text
                    x={size / 2}
                    y={size / 2}
                    textAnchor="middle"
                    dy="0.3em"
                    style={{
                        transform: 'rotate(90deg)',
                        transformOrigin: 'center',
                        fontSize: '1.5rem',
                        fontWeight: 700,
                        fill: color
                    }}
                >
                    {Math.round(score)}
                </text>
            </svg>
            <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: color }}>
                    {getLabel()}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'hsl(var(--txt-secondary))' }}>
                    {label}
                </div>
            </div>
        </div>
    );
}
