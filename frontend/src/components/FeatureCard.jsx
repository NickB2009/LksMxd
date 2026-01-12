import { Eye, Activity, Scissors, CheckCircle2, AlertCircle } from 'lucide-react';
import ScoreGauge from './ScoreGauge';

export default function FeatureCard({ feature, data }) {
    if (!data) return null;

    const renderEyes = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <Eye size={24} color="hsl(var(--accent-primary))" />
                <h3 style={{ margin: 0 }}>Eyes & Gaze</h3>
            </div>

            <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                <ScoreGauge value={data.harmonyIndex} ideal={90} size={120} label="Eye Harmony" />

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <MetricRow label="Canthal Tilt" value={`${data.canthalTilt}°`} desc={data.canthalTilt > 4 ? "Positive / Hunter" : "Neutral / Ideal"} />
                    <MetricRow label="Eye Spacing (ESR)" value={data.eyeSpacingRatio} desc="Ideal: 0.46 - 0.48" />
                    <MetricRow label="Eye Area Ratio" value={`${(data.eyeAreaRatio * 100).toFixed(2)}%`} desc="Ideal: 1.5% - 2.0%" />
                    <MetricRow label="Eye Symmetry" value={`${data.symmetry}%`} desc="Central alignment" />
                </div>
            </div>
        </div>
    );

    const renderNose = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <Activity size={24} color="hsl(var(--accent-primary))" />
                <h3 style={{ margin: 0 }}>Nasal Structure</h3>
            </div>

            <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                <ScoreGauge value={data.symmetry} ideal={100} size={120} label="Symmetry" />

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <MetricRow label="Nose L/W Ratio" value={data.noseRatio} desc={data.noseRatio > 1.6 ? "Narrow / Refined" : "Balanced"} />
                    <MetricRow label="Facial Width Ratio" value={`${(data.noseToFaceWidth * 100).toFixed(1)}%`} desc="Ideal: ~25% of bizygoma" />
                    <MetricRow label="Nasal Symmetry" value={`${data.symmetry}%`} desc="Central alignment" />
                </div>
            </div>
        </div>
    );

    const renderJaw = () => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <Scissors size={24} color="hsl(var(--accent-primary))" />
                <h3 style={{ margin: 0 }}>Jawline & Chin</h3>
            </div>

            <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
                <ScoreGauge value={data.symmetry} ideal={100} size={120} label="Symmetry" />

                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <MetricRow label="Gonial Angle" value={`${data.gonialAngle}°`} desc={data.gonialAngle < 125 ? "Strong / Masculine" : "Balanced"} />
                    <MetricRow label="Jaw-to-Cheek Ratio" value={data.jawToCheekRatio} desc="Ideal: 0.85 - 0.95" />
                    <MetricRow label="Chin verticality" value={data.chinPhiltrumRatio} desc="Ideal: 2.0 - 2.5" />
                    <MetricRow label="Lower Third Ratio" value={data.lowerThirdRatio ? data.lowerThirdRatio : "N/A"} desc="Ideal: 2.0 (Moridani)" />
                </div>
            </div>
        </div>
    );

    return (
        <div className="fade-in">
            {feature === 'eyes' && renderEyes()}
            {feature === 'nose' && renderNose()}
            {feature === 'jawline' && renderJaw()}
        </div>
    );
}

function MetricRow({ label, value, desc }) {
    return (
        <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0.75rem 0',
            borderBottom: '1px solid hsla(var(--border-subtle), 0.5)'
        }}>
            <div>
                <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>{label}</div>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>{desc}</div>
            </div>
            <div style={{ fontSize: '1.25rem', fontWeight: 700, fontFamily: 'monospace' }}>
                {value}
            </div>
        </div>
    );
}
