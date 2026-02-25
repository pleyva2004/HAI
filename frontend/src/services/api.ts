import { GenerateRequest, GeneratedResponse } from '@/types';

const API_BASE_URL = 'http://localhost:8000/api';

export const api = {
    /**
     * Check if the backend is healthy
     */
    async checkHealth(): Promise<boolean> {
        try {
            const res = await fetch(`${API_BASE_URL}/health`);
            return res.ok;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    },

    /**
     * Upload a screenshot and get the base64 representation
     */
    async uploadScreenshot(file: File): Promise<{ image: string }> {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`${API_BASE_URL}/upload-screenshot`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            throw new Error('Failed to upload screenshot');
        }

        return res.json();
    },

    /**
     * Generate an SAT question based on input
     */
    async generateQuestion(data: GenerateRequest): Promise<GeneratedResponse> {
        const res = await fetch(`${API_BASE_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to generate question');
        }

        return res.json();
    },

    /**
     * Submit feedback to refine a generated question
     */
    async submitFeedback(workflowId: string, feedbackText: string): Promise<GeneratedResponse> {
        const res = await fetch(`${API_BASE_URL}/feedback/${workflowId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ feedback_text: feedbackText }),
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Failed to submit feedback');
        }

        return res.json();
    },
};
