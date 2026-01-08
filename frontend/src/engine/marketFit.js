// Market Profiles - Defined by Ideal Ranges (Mean) and Tolerance (Std Dev)
const MARKETS = {
    editorial: {
        description: "High fashion, sharp features, unconventional beauty.",
        bounty: {
            fWHR: { mean: 1.9, std: 0.15 },
            canthalTilt: { mean: 6, std: 4 },
            jawToCheekRatio: { mean: 0.9, std: 0.1 },
            midToLowerRatio: { mean: 1.0, std: 0.2 },
        }
    },
    commercial: {
        description: "Approachable, classic beauty, balanced proportions.",
        bounty: {
            fWHR: { mean: 1.75, std: 0.08 },
            canthalTilt: { mean: 2, std: 2 },
            jawToCheekRatio: { mean: 0.8, std: 0.05 },
            midToLowerRatio: { mean: 1.0, std: 0.1 },
        }
    },
    character: {
        description: "Unconventional features, distinct deviations.",
        bounty: {
            fWHR: { mean: 1.75, std: 0.3 },
            canthalTilt: { mean: 0, std: 10 },
            jawToCheekRatio: { mean: 0.75, std: 0.2 },
        }
    }
};

function gaussian(x, mean, std) {
    return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / std, 2));
}

function computeScore(metrics, profile) {
    let scoreSum = 0;
    let weightSum = 0;

    for (const [trait, stats] of Object.entries(profile.bounty)) {
        if (metrics[trait] === undefined) continue;

        const val = parseFloat(metrics[trait]);
        const p = gaussian(val, stats.mean, stats.std);
        const maxP = gaussian(stats.mean, stats.mean, stats.std);

        // Normalized score 0-1 for this trait
        const matchScore = p / maxP;

        scoreSum += matchScore;
        weightSum += 1;
    }

    return weightSum > 0 ? (scoreSum / weightSum) : 0;
}

export function calculateMarketFit(rawMetrics) {
    if (!rawMetrics) return null;
    const results = {};
    for (const [marketName, profile] of Object.entries(MARKETS)) {
        const avgFit = computeScore(rawMetrics, profile);
        results[marketName] = {
            score: Math.round(avgFit * 100),
            description: profile.description
        };
    }
    return results;
}

/**
 * Calculates a "Potential Score" simulating reduced facial puffiness.
 * Hypothesis: Lower body fat / puffiness increases Jaw Definition (JawToCheekRatio) 
 * and might slightly affect fWHR perception (more angular).
 */
export function calculatePotential(rawMetrics) {
    if (!rawMetrics) return null;

    // Simulation: Improve Jaw Definition by 10-15% towards the "Ideal" of 1.0 (squaredness)
    // or just strictly increase it if it's low.
    let currentJaw = parseFloat(rawMetrics.jawToCheekRatio);

    // Simulation: "Leaning out" increases defined jaw width relative to soft cheeks.
    const projectedJaw = currentJaw * 1.12;

    const projectedMetrics = {
        ...rawMetrics,
        jawToCheekRatio: projectedJaw
    };

    const current = calculateMarketFit(rawMetrics);
    const projected = calculateMarketFit(projectedMetrics);

    const potentialResults = {};

    for (const market in current) {
        potentialResults[market] = {
            current: current[market].score,
            potential: projected[market].score,
            gain: projected[market].score - current[market].score
        };
    }

    return potentialResults;
}
