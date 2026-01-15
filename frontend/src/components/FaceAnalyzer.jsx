import { useState, useRef } from 'react';
import { Upload, Loader2, RefreshCw, Bug, Microscope } from 'lucide-react';
import MorphologyReport from './MorphologyReport';

export default function FaceAnalyzer() {
    const [image, setImage] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [landmarks, setLandmarks] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [debugMode, setDebugMode] = useState(false);
    const [debugStats, setDebugStats] = useState(null); // Store calc results for overlay

    const imgRef = useRef(null);
    const canvasRef = useRef(null);

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const url = URL.createObjectURL(file);
        setImage(url);
        setLandmarks(null);
        setAnalysisData(null);
    };

    // Store best-truth landmarks from calibration
    const calibrationLandmarksRef = useRef(null);

    const runCalibration = async () => {
        setAnalyzing(true);
        try {
            const res = await fetch('http://localhost:8000/debug/calibration');
            const data = await res.json();
            calibrationLandmarksRef.current = data.analysis.landmarks;
            setAnalysisData(data.analysis);
            setImage(data.image_url);
            setDebugMode(true);
        } catch (e) {
            console.error(e);
            alert("Calibration failed");
            setAnalyzing(false);
        }
    };

    const runAnalysis = async () => {
        if (!imgRef.current) return;

        // Calibration Bypass: Use mocked truth if active
        if (calibrationLandmarksRef.current) {
            setLandmarks(calibrationLandmarksRef.current);
            requestAnimationFrame(() => drawLandmarks(calibrationLandmarksRef.current));
            setAnalyzing(false);
            return;
        }

        setAnalyzing(true);

        try {
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

            // Use Backend Landmarks for perfect visualization alignment
            if (data.analysis.landmarks) {
                setLandmarks(data.analysis.landmarks);
                // Slight delay to ensure canvas is ready if needed, but synchronous call usually works
                requestAnimationFrame(() => drawLandmarks(data.analysis.landmarks));
            }

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

        // Container dimensions
        const displayW = img.offsetWidth;
        const displayH = img.offsetHeight;

        // Image natural dimensions
        const naturalW = img.naturalWidth;
        const naturalH = img.naturalHeight;

        // Calculate Rendered Dimensions (Object-Fit: Contain Logic)
        const naturalRatio = naturalW / naturalH;
        const displayRatio = displayW / displayH;

        let renderW, renderH, offX = 0, offY = 0;

        if (naturalRatio > displayRatio) {
            // Image is wider relative to container -> Horizontal fit
            renderW = displayW;
            renderH = displayW / naturalRatio;
            offY = (displayH - renderH) / 2;
        } else {
            // Image is taller relative to container -> Vertical fit
            renderH = displayH;
            renderW = displayH * naturalRatio;
            offX = (displayW - renderW) / 2;
        }

        // Update Debug Stats
        setDebugStats({
            container: `${displayW}x${displayH}`,
            natural: `${naturalW}x${naturalH}`,
            rendered: `${Math.round(renderW)}x${Math.round(renderH)}`,
            offset: `X:${Math.round(offX)} Y:${Math.round(offY)}`
        });

        canvas.width = displayW;
        canvas.height = displayH;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Debug Visuals: Render Box
        if (debugMode) {
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
            ctx.lineWidth = 2;
            ctx.strokeRect(offX, offY, renderW, renderH);

            // Crosshair
            ctx.strokeStyle = 'rgba(0, 100, 255, 0.5)';
            ctx.beginPath();
            ctx.moveTo(offX + renderW / 2, 0); ctx.lineTo(offX + renderW / 2, displayH);
            ctx.moveTo(0, offY + renderH / 2); ctx.lineTo(displayW, offY + renderH / 2);
            ctx.stroke();
        }

        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';

        keypoints.forEach((pt) => {
            ctx.beginPath();
            const x = offX + (pt.x * renderW);
            const y = offY + (pt.y * renderH);

            ctx.arc(x, y, 1.2, 0, 2 * Math.PI);
            ctx.fill();
        });
    };

    const reset = () => {
        calibrationLandmarksRef.current = null;
        setDebugMode(false);
        setDebugStats(null);
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
                            cursor: 'pointer'
                        }}>
                            <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} />
                            Select Photo
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
                                    display: 'block',
                                    margin: 0,
                                    padding: 0,
                                    width: 'auto',
                                    height: 'auto',
                                    objectFit: 'contain'
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

                            {/* Debug Overlay */}
                            {debugMode && debugStats && (
                                <div style={{
                                    position: 'absolute',
                                    top: 10,
                                    right: 10,
                                    background: 'rgba(0, 0, 0, 0.85)',
                                    color: 'lime',
                                    padding: '0.5rem',
                                    borderRadius: '4px',
                                    fontSize: '0.75rem',
                                    fontFamily: 'monospace',
                                    pointerEvents: 'none',
                                    zIndex: 10
                                }}>
                                    <div><strong>DEBUG MODE</strong></div>
                                    <div>Container: {debugStats.container}</div>
                                    <div>Natural: {debugStats.natural}</div>
                                    <div>Rendered: {debugStats.rendered}</div>
                                    <div>Offset: {debugStats.offset}</div>
                                </div>
                            )}

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

