import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

let model = null;

/**
 * Loads the MediaPipe Face Mesh model.
 * Singleton pattern to avoid reloading.
 */
export async function loadModel() {
    if (model) return model;

    try {
        model = await faceLandmarksDetection.createDetector(
            faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
            {
                runtime: 'tfjs',
                refineLandmarks: true, // critical for precise eye/iris landmarks
                maxFaces: 1,
            }
        );
        console.log("Face Mesh Model Loaded");
        return model;
    } catch (err) {
        console.error("Failed to load Face Mesh model:", err);
        throw err;
    }
}

/**
 * Detects face landmarks from an image, video, or canvas element.
 * @param {HTMLImageElement | HTMLVideoElement | HTMLCanvasElement} input 
 * @returns {Promise<faceLandmarksDetection.Face[]>}
 */
export async function detectLandmarks(input) {
    if (!model) await loadModel();

    try {
        const predictions = await model.estimateFaces(input, {
            flipHorizontal: false,
            staticImageMode: true, // Set to true for images to improve accuracy
        });

        return predictions;
    } catch (err) {
        console.error("Error detecting landmarks:", err);
        throw err;
    }
}
