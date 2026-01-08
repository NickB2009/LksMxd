import { useState, useRef, useEffect } from 'react';
import { Upload, Loader2, RefreshCw } from 'lucide-react';
import { detectLandmarks, loadModel } from '../engine/morphology';
import MorphologyReport from './MorphologyReport';
import MarketFitReport from './MarketFitReport';
import DetailedAnalysis from './DetailedAnalysis';

export default function FaceAnalyzer() {
    const [image, setImage] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [landmarks, setLandmarks] = useState(null);

    // Results State
    const [reportData, setReportData] = useState(null);
    const [rarityData, setRarityData] = useState(null);
    const [marketFitData, setMarketFitData] = useState(null);
    const [potentialData, setPotentialData] = useState(null);

    const [modelLoading, setModelLoading] = useState(false);

    const imgRef = useRef(null);
    const canvasRef = useRef(null);

    // Preload model on mount
    useEffect(() => {
        async function init() {
            setModelLoading(true);
            try {
                await loadModel();
            } catch (err) {
                console.error("Model init failed", err);
            } finally {
                setModelLoading(false);
            }
        }
        init();
    }, []);

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const url = URL.createObjectURL(file);
        setImage(url);

        // Reset results
        setLandmarks(null);
        setReportData(null);
        setRarityData(null);
        setMarketFitData(null);
        setPotentialData(null);

        // Auto-analyze triggered by onLoad
    };

    const runAnalysis = async () => {
        if (!imgRef.current) return;
        setAnalyzing(true);

        try {
            // 1. Visual Feedback (Client-side)
            const predictions = await detectLandmarks(imgRef.current);
            if (predictions && predictions.length > 0) {
                setLandmarks(predictions[0].keypoints);
                drawLandmarks(predictions[0].keypoints);
            }

            // 2. Deep Analysis (Server-side)
            // Get Blob from current image URL
            const blob = await fetch(image).then(r => r.blob());
            const formData = new FormData();
            formData.append('file', blob);

            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(err.detail || 'Analysis failed');
            }

            const data = await response.json();

            // Map API response to Component State
            // Map API response to Component State
            setReportData({
                ...data.morphology, // Pass all top-level metrics (phiRatio, fWHR, thirds, etc.)
                analysis: {
                    faceShape: "Classified",
                    eyeTiltCategory: data.morphology.canthalTilt > 4 ? "Positive (Hunter)" : "Neutral",
                    proportions: "Calculated"
                }
            });

            setRarityData(data.rarity);
            setMarketFitData(data.marketFit);
            setPotentialData(data.potential);

        } catch (err) {
            console.error(err);
            alert(`Analysis failed: ${err.message}`);
        } finally {
            setAnalyzing(false);
        }
    };

    const drawLandmarks = (keypoints) => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img) return;

        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Premium overlay style
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';

        keypoints.forEach((pt) => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 1, 0, 2 * Math.PI);
            ctx.fill();
        });
    };

    const reset = () => {
        setImage(null);
        setLandmarks(null);
        setReportData(null);
        setRarityData(null);
        setMarketFitData(null);
        setPotentialData(null);
    };

    return (
        <div style={{ paddingBottom: '4rem' }}>
            <div className="card glass" style={{ maxWidth: '800px', margin: '0 auto', minHeight: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                {!image ? (
                    <div style={{ textAlign: 'center', padding: '3rem' }}>
                        <div style={{
                            width: '80px', height: '80px',
                            borderRadius: '50%',
                            background: 'hsla(var(--bg-panel), 0.5)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            margin: '0 auto 1.5rem auto'
                        }}>
                            <Upload size={32} className="text-muted" />
                        </div>
                        <h3 style={{ marginBottom: '0.5rem' }}>Upload Reference Photo</h3>
                        <p className="text-muted" style={{ marginBottom: '2rem' }}>
                            Frontal face, neutral expression, good lighting.
                        </p>

                        <label className="btn" style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            cursor: modelLoading ? 'wait' : 'pointer',
                            opacity: modelLoading ? 0.7 : 1
                        }}>
                            <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} disabled={modelLoading} />
                            {modelLoading ? (
                                <>
                                    <Loader2 className="animate-spin" size={18} />
                                    Loading Engine...
                                </>
                            ) : (
                                "Select Photo"
                            )}
                        </label>
                    </div>
                ) : (
                    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div style={{ position: 'relative', borderRadius: 'var(--radius-md) var(--radius-md) 0 0', overflow: 'hidden', background: '#000', lineHeight: 0 }}>
                            <img
                                ref={imgRef}
                                src={image}
                                alt="Analysis Target"
                                style={{ width: '100%', maxHeight: '600px', objectFit: 'contain', display: 'block' }}
                                onLoad={() => runAnalysis()}
                            />
                            <canvas
                                ref={canvasRef}
                                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                            />

                            {analyzing && (
                                <div style={{
                                    position: 'absolute', inset: 0,
                                    background: 'rgba(0,0,0,0.7)',
                                    backdropFilter: 'blur(4px)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    flexDirection: 'column', gap: '1rem',
                                    zIndex: 20
                                }}>
                                    <Loader2 className="animate-spin" size={48} color="hsl(var(--accent-primary))" />
                                    <span style={{ fontWeight: 500, letterSpacing: '0.05em' }}>COMPUTING MORPHOLOGY...</span>
                                </div>
                            )}
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '0 1rem 1rem 1rem' }}>
                            <button className="btn" onClick={reset} style={{ background: 'transparent', border: '1px solid hsl(var(--border-subtle))' }}>
                                <RefreshCw size={16} style={{ marginRight: '0.5rem' }} /> New Analysis
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {reportData && (
                <>
                    <MorphologyReport data={reportData} />
                    <DetailedAnalysis data={reportData} />
                </>
            )}
            {rarityData && marketFitData && <MarketFitReport rarity={rarityData} marketFit={marketFitData} potential={potentialData} />}
        </div>
    );
}
