export interface Question {
    text: string;
    equation?: string | null;
    table?: {
        headers: string[];
        rows: string[][];
    } | null;
    visual?: string | null;
    answer_choices: {
        [key: string]: string; // "A": "...", "B": "..."
    };
    correct_answer?: string; // "A", "B", "C", "D"
    explanation?: string;
}

export interface GeneratedResponse {
    question: Question;
    metadata: {
        workflow_id: string;
        generate_attempts: number;
        validation_passed: boolean;
    };
}

export interface GenerateRequest {
    image?: string; // base64
    description?: string;
    requested_section?: 'Math' | 'Reading and Writing';
    requested_difficulty?: 'Easy' | 'Medium' | 'Hard';
    requested_domain?: string;
    provide_answer?: boolean;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content?: string;
    image?: string; // base64 for user uploads
    question?: GeneratedResponse; // For assistant responses containing a question
    isLoading?: boolean;
    error?: boolean;
}
