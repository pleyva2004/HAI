# SAT Question Generator - Frontend

Modern Next.js frontend for the AI-powered SAT Question Generator application.

## Features

- **Chat-Like Interface**: Intuitive conversation-style question generation
- **File Upload Support**: Upload PDF/PNG examples for style matching
- **Metadata-Rich Display**:
  - Difficulty scores (0-100 calibrated scale)
  - Style match indicators (90%+ accuracy)
  - Real SAT vs AI-generated badges
  - Validation status indicators
- **Question Bank Search**: Search 10,000+ official SAT questions
- **Export Functionality**: Download questions as PDF or JSON
- **Responsive Design**: Works on desktop, tablet, and mobile

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix UI)
- **State Management**: Zustand
- **Server State**: React Query (TanStack Query)
- **HTTP Client**: Axios
- **PDF Generation**: jsPDF
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000/api/v1`

## Deployment

Deploy to Vercel:
```bash
npm install -g vercel
vercel
```

## License

This project is part of the SAT Question Generator application.
