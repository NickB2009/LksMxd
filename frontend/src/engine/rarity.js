// Mock Reference Population Statistics (Based on general anthropometric data approximations)
const POPULATION_STATS = {
    fWHR: { mean: 1.75, std: 0.1 },
    canthalTilt: { mean: 1, std: 3 }, // degrees
    midToLowerRatio: { mean: 1.0, std: 0.15 },
    jawToCheekRatio: { mean: 0.75, std: 0.08 },
};

/**
 * Calculates the Cumulative Distribution Function (CDF) for a standard normal distribution
 */
function cdf(x) {
    return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function erf(x) {
    // Approximation of the error function
    const t = 1.0 / (1.0 + 0.5 * Math.abs(x));
    const tau = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - tau : tau - 1;
}

/**
 * Calculates rarity based on statistical deviation (Z-score).
 * Rarity increases as you move away from the mean (in either direction).
 * @param {Object} rawMetrics - Raw values from metrics.js
 */
export function calculateRarity(rawMetrics) {
    if (!rawMetrics) return null;

    let totalDeviation = 0;
    let count = 0;
    const traits = {};

    for (const [key, stats] of Object.entries(POPULATION_STATS)) {
        if (rawMetrics[key] === undefined) continue;

        const value = parseFloat(rawMetrics[key]);
        const zScore = (value - stats.mean) / stats.std;

        // Rarity is absolute deviation. A face is "rare" if it's very wide OR very narrow.
        // 0 = Average (Common), High = Rare.
        const rarityScore = Math.abs(zScore);

        // Percentile (how many people are you different from?)
        // If z=2, you are in the top 2.5% or bottom 2.5% (5% total). You are rarer than 95%.
        const p = cdf(zScore);
        const percentile = (p > 0.5 ? p : 1 - p) * 2; // Normalize to "distance from center" logic? 
        // Actually, simple percentile of "exceeding others in deviation":
        const probability = 2 * (1 - cdf(Math.abs(zScore))); // Prob of finding someone this extreme or more
        // Rarity Score 1-10:
        // 10 = Extemely Rare (p < 0.001)
        // 5 = Uncommon (p < 0.1)
        // 1 = Average (p ~ 1.0)

        // Map probability to 1-10 linear-ish
        // P=1.0 -> 1
        // P=0.01 -> 10
        const traitRarity = Math.min(10, Math.max(1, 1 + (-Math.log10(probability) * 3.5)));

        totalDeviation += traitRarity;
        count++;

        traits[key] = {
            value,
            zScore: zScore.toFixed(2),
            rarity: traitRarity.toFixed(1),
            label: getRarityLabel(traitRarity)
        };
    }

    const avgRarity = totalDeviation / count;

    return {
        score: avgRarity.toFixed(1),
        label: getGlobalRarityLabel(avgRarity),
        details: traits
    };
}

function getRarityLabel(score) {
    if (score > 8) return "Extremely Rare";
    if (score > 6) return "Distinct";
    if (score > 4) return "Uncommon";
    return "Typical";
}

function getGlobalRarityLabel(score) {
    if (score > 7) return "Statistically Unique";
    if (score > 5) return "High Distinctiveness";
    if (score > 3) return "Above Average";
    return "Common Morphology";
}
