import ScoreGauge from './ScoreGauge';
import { Eye, Wind, Activity } from 'lucide-react';

const FEATURE_CONFIGS = {
    eyes: {
        icon: Eye,
        title: 'Eyes',
        color: 'hsl(200, 70%, 50%)',
        metrics: [
            { key: 'eye_size_ratio', label: 'Size', ideal: 0.0175, format: (v) => (v * 100).toFixed(2) + '%' },
            { key: 'esr', label: 'Spacing', ideal: 0.46 },
            { key: 'canthalTilt', label: 'Tilt', ideal: 8.0, suffix: '°' }
        ],
        getVerdict: (data) => {
            const size = data.psl?.eye_size_ratio || 0.015;
            const spacing = data.psl?.esr || 0.46;
            const tilt = data.canthalTilt || 0;

            if (size > 0.018 && tilt > 7) return "Your eyes are exceptionally large with ideal hunter tilt.";
            if (size > 0.018) return "Your eyes are exceptionally large and prominent.";
            if (tilt > 7) return "You have a strong positive canthal tilt - a highly attractive feature.";
            if (size > 0.016) return "You have well-proportioned, attractive eyes.";
            if (spacing < 0.44) return "Your eyes are closely set, creating intensity.";
            return "Your eye proportions are within normal range.";
        }
    },
    nose: {
        icon: Wind,
        title: 'Nose',
        color: 'hsl(30, 70%, 50%)',
        metrics: [
            { key: 'nose_lw_ratio', label: 'Length/Width', ideal: 1.5 },
            { key: 'mouthNoseRatio', label: 'Mouth Ratio', ideal: 1.618 }
        ],
        getVerdict: (data) => {
            const lw = data.psl?.nose_lw_ratio || 1.5;

            if (lw > 1.6) return "Your nose is long and narrow - a highly attractive feature.";
            if (lw > 1.4) return "Your nose has balanced proportions.";
            return "Your nose is on the wider side.";
        }
    },
    jaw: {
        icon: Activity,
        title: 'Jaw & Structure',
        color: 'hsl(280, 70%, 50%)',
        metrics: [
            { key: 'jawToCheek', label: 'Definition', ideal: 0.90 },      // Updated from 0.85
            { key: 'lower_third_ratio', label: 'Lower Third', ideal: 0.32 } // Updated from 0.33
        ],
        getVerdict: (data) => {
            const ratio = data.jawToCheek || 0.78;
            const lowerThird = data.psl?.lower_third_ratio || 0.33;

            if (ratio > 0.95) return "You have an exceptionally wide, model-tier jaw.";
            if (ratio > 0.88) return "You have an exceptionally defined, strong jawline.";
            if (lowerThird > 0.34) return "You have strong lower third dominance - highly masculine.";
            if (ratio > 0.82) return "Your jaw structure is well-defined and attractive.";
            return "Your jaw has softer, more delicate contours.";
        }
    }
};

export default function FeatureCard({ feature, data }) {
    const config = FEATURE_CONFIGS[feature];
    if (!config) return null;

    const Icon = config.icon;

    return (
        <div style={{
            background: 'hsla(var(--bg-panel), 0.3)',
            padding: '1.5rem',
            borderRadius: 'var(--radius-md)',
            border: '1px solid hsla(255,255,255,0.05)'
        }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                marginBottom: '1.25rem',
                paddingBottom: '1rem',
                borderBottom: '1px solid hsla(255,255,255,0.05)'
            }}>
                <Icon size={24} color={config.color} />
                <h4 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>
                    {config.title}
                </h4>
            </div>

            {/* Gauges */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-around',
                marginBottom: '1.25rem',
                flexWrap: 'wrap',
                gap: '1rem'
            }}>
                {config.metrics.map(metric => {
                    let value = data[metric.key];

                    // Navigate to psl object if needed
                    if (!value && data.psl && data.psl[metric.key]) {
                        value = data.psl[metric.key];
                    }

                    return (
                        <ScoreGauge
                            key={metric.key}
                            value={value}
                            ideal={metric.ideal}
                            label={metric.label}
                            size={100}
                        />
                    );
                })}
            </div>

            {/* Verdict */}
            <div style={{
                background: `linear-gradient(to right, ${config.color}15, transparent)`,
                padding: '0.875rem',
                borderRadius: 'var(--radius-sm)',
                borderLeft: `3px solid ${config.color}`,
                fontSize: '0.9rem',
                lineHeight: 1.5,
                color: 'hsl(var(--txt-primary))'
            }}>
                {config.getVerdict(data)}
            </div>
        </div>
    );
}
