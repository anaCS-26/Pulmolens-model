const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface PredictionResult {
    predictions: Record<string, number>;
    heatmap: string;
    report?: string;
    image_id: string;
}

export const warmup = async () => {
    try {
        await fetch(`${API_BASE_URL}/health`);
    } catch (e) {
        // ignore
    }
};

export const predict = async (file: File): Promise<PredictionResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
};

export interface SummarizeResponse {
    report: string;
    sources: string[];
}

export const summarizeAI = async (
    predictions: Record<string, number>,
    heatmapB64: string
): Promise<SummarizeResponse> => {
    const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            predictions,
            heatmap: heatmapB64,
        }),
    });

    if (!response.ok) {
        throw new Error(`Summarize failed: ${response.statusText}`);
    }

    return response.json();
};

export const submitFeedback = async (
    file: File,
    rating: 'good' | 'bad',
    details?: string,
    predictions?: Record<string, number>
) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('rating', rating);
    if (details) formData.append('details', details);
    if (predictions) formData.append('predictions', JSON.stringify(predictions));

    const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Feedback submission failed: ${response.statusText}`);
    }

    return response.json();
};
