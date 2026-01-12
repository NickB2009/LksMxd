import { useState, useRef, useEffect } from 'react';
import { Upload, Loader2, RefreshCw } from 'lucide-react';
import { detectLandmarks, loadModel } from '../engine/morphology';
import MorphologyReport from './MorphologyReport';

export default function FaceAnalyzer() {
    const [image, setImage] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [landmarks, setLandmarks] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [modelLoading, setModelLoading] = useState(false);

    const imgRef = useRef(null);
    const canvasRef = useRef(null);

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
        setLandmarks(null);
        setAnalysisData(null);
    };

    const runAnalysis = async () => {
        if (!imgRef.current) return;
        setAnalyzing(true);

        try {
            const predictions = await detectLandmarks(imgRef.current);
            if (predictions && predictions.length > 0) {
                setLandmarks(predictions[0].keypoints);
                drawLandmarks(predictions[0].keypoints);
            }

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
            setAnalysisData(data.analysis);

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

        // Visual Synchronization: Match canvas to DISPLAY dimensions
        // This ensures dots are always visible size (not shrunk by high-res image scaling)
        const displayW = img.offsetWidth;
        const displayH = img.offsetHeight;

        canvas.width = displayW;
        canvas.height = displayH;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(100, 255, 218, 0.7)'; // High-vis cyan

        const scaleX = displayW / img.naturalWidth;
        const scaleY = displayH / img.naturalHeight;

        keypoints.forEach((pt) => {
            ctx.beginPath();
            // Scale backend coordinates (source resolution) to visual coordinates
            ctx.arc(pt.x * scaleX, pt.y * scaleY, 1.2, 0, 2 * Math.PI);
            ctx.fill();
        });
    };

    const reset = () => {
        setImage(null);
        setLandmarks(null);
        setAnalysisData(null);
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
                    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>

                        {/* 
                            Container - display: inline-block ensures the div shrinks to fit the image exactly.
                            Position relative establishes the coordinate context for the absolute canvas.
                        */}
                        <div style={{
                            position: 'relative',
                            display: 'inline-block',
                            borderRadius: 'var(--radius-md)',
                            overflow: 'hidden',
                            lineHeight: 0,
                            boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
                        }}>
                            <img
                                ref={imgRef}
                                src={image}
                                alt="Analysis Target"
                                style={{
                                    maxWidth: '100%',
                                    maxHeight: '600px',
                                    display: 'block'
                                    // No width/height forced here, let it render naturally constrained by max-
                                }}
                                onLoad={() => runAnalysis()}
                            />
                            {/* Canvas overlays perfectly because container matches image size */}
                            <canvas
                                ref={canvasRef}
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    pointerEvents: 'none'
                                }}
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

            {analysisData && (
                <MorphologyReport data={analysisData} />
            )}
        </div>
    );
}

