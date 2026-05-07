import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface PredictionResult {
    predictions: Record<string, number>;
    heatmap: string;
    report: string;
    image_id: string;
}

export const predict = async (file: File): Promise<PredictionResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export interface SummarizeResponse {
    report: string;
    sources: string[];
}

export const summarizeAI = async (
    predictions: Record<string, number>,
    heatmapB64: string
): Promise<SummarizeResponse> => {
    const response = await axios.post(`${API_BASE_URL}/summarize`, {
        predictions,
        heatmap: heatmapB64,
    });

    return response.data;
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

    const response = await axios.post(`${API_BASE_URL}/feedback`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};
