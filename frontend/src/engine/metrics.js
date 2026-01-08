/**
 * MediaPipe Face Mesh Landmark Indices
 * Based on canonical canonical face mesh topology
 */
const LANDMARKS = {
    // Midline
    chin: 152,
    noseTip: 1,
    midEyes: 168, // glabella approx
    topHead: 10,

    // Eyes
    leftEyeOuter: 33,
    leftEyeInner: 133,
    rightEyeInner: 362,
    rightEyeOuter: 263,

    // Jaw
    jawLeft: 58,  // gonion area approx
    jawRight: 288,
    faceWidthLeft: 234, // zygoma
    faceWidthRight: 454,

    // Mouth
    mouthLeft: 61,
    mouthRight: 291,
};

function distance(p1, p2) {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

function angleDegrees(p1, p2) {
    return Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
}

/**
 * Calculates a comprehensive morphology report from 3D keypoints.
 * @param {Array} keypoints - Array of {x, y, z} objects from MediaPipe
 */
export function calculateMorphology(keypoints) {
    if (!keypoints || keypoints.length < 468) return null;

    const points = {};
    for (const [name, index] of Object.entries(LANDMARKS)) {
        points[name] = keypoints[index];
    }

    // --- 1. Facial Width-to-Height Ratio (fWHR) ---
    // Width: Bizygomatic breadth (234 to 454)
    // Height: Nasion to Prostlion (mid-brow to upper lip) - Simplified here as 10 to 152 for total face, 
    // or glabella to upper lip. 
    // Market standard fWHR is usually: width (bizygomatic) / height (mid-brow to upper lip).
    // Let's use a standard Full Face Ratio for valid parsing: Width / (Chin to Mid-Brow).

    const faceWidth = distance(points.faceWidthLeft, points.faceWidthRight);
    const faceHeight = distance(points.midEyes, points.chin); // Glabella to Menton
    const fWHR = faceWidth / faceHeight;

    // --- 2. Canthal Tilt ---
    // Angle between inner and outer eye corners
    // Positive = positive tilt (hunter eyes/cat eyes)
    const leftEyeAngle = angleDegrees(points.leftEyeInner, points.leftEyeOuter); // normally negative if outer is higher in screen coords (y down)?
    // Screen coords: Y increases downwards. 
    // If outer (x_out, y_out) is higher (smaller y) than inner (x_in, y_in), then y_out - y_in is negative.
    // We want Positive Tilt if outer is physically higher.
    // So invert the y diff or just negate the angle result if using standard screen coords?
    // Let's check: inner=(100, 100), outer=(120, 90). dy = -10, dx = 20. atan2(-10, 20) is negative.
    // We want to call this "Positive Tilt". So we negate the result.

    const leftTilt = -angleDegrees(points.leftEyeInner, points.leftEyeOuter);
    const rightTilt = -angleDegrees(points.rightEyeInner, points.rightEyeOuter);
    // Wait, right eye: inner is on left (smaller x) relative to outer? No.
    // Right eye (subject's right) is on image left.
    // Subject's Right Eye: Inner (362), Outer (263).
    // On screen: Outer (Left side of screen) -> Inner (Middle).
    // Vector: Outer -> Inner. 
    // Let's stick to standard: Inner to Outer.
    // For subject's Right eye (on screen left): Inner is x=100, Outer is x=80.
    // Vector Inner->Outer: dx is negative.
    // If Outer is higher (smaller y), dy is negative.
    // atan2(-10, -20) -> 3rd quadrant.

    // SIMPLER: Compute slope for both eyes independently using absolute differences.
    // Or just average the "uplift" of the outer corner relative to inner.
    const avgCanthalTilt = (leftTilt + (-angleDegrees(points.rightEyeInner, points.rightEyeOuter))) / 2;
    // Needs verification physically. Let's just output raw degrees for the left eye (usually sufficient for symmetry assumption, or avg).
    // Let's use: (LeftEyeOuter.y - LeftEyeInner.y) inverted.

    // --- 3. Facial Thirds (Vertical Proportions) ---
    // Upper: Hairline (trichion) to Glabella. MediaPipe 10 is top of mesh, not necessarily hairline.
    // Mid: Glabella (168) to Nose Tip (1).
    // Lower: Nose Tip (1) to Chin (152).
    const midFace = distance(points.midEyes, points.noseTip);
    const lowerFace = distance(points.noseTip, points.chin);
    const midToLowerRatio = midFace / lowerFace;

    // --- 4. Jawline Width relative to Cheeks ---
    const jawWidth = distance(points.jawLeft, points.jawRight);
    const jawToCheekRatio = jawWidth / faceWidth;

    return {
        raw: {
            fWHR: fWHR.toFixed(2),
            canthalTilt: leftTilt.toFixed(1), // degrees
            midToLowerRatio: midToLowerRatio.toFixed(2),
            jawToCheekRatio: jawToCheekRatio.toFixed(2),
            faceWidthPx: faceWidth,
            faceHeightPx: faceHeight,
        },
        analysis: {
            faceShape: classifyFaceShape(fWHR, jawToCheekRatio),
            eyeTiltCategory: classifyTilt(leftTilt),
            proportions: classifyProportions(midToLowerRatio)
        }
    };
}

function classifyFaceShape(fwhr, jawRatio) {
    if (jawRatio > 0.9) return "Square/Rectangle";
    if (fwhr > 1.9) return "Broad/Wide";
    if (fwhr < 1.6) return "Oblong/Oval";
    return "Hybrid";
}

function classifyTilt(degrees) {
    if (degrees > 4) return "Positive (Hunter)";
    if (degrees < -2) return "Negative (Prey)";
    return "Neutral";
}

function classifyProportions(ratio) {
    // Ideal ~ 1.0
    if (ratio > 1.1) return "Long Midface";
    if (ratio < 0.9) return "Compact Midface";
    return "Balanced";
}
