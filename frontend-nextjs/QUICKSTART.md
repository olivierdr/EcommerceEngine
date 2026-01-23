# Quick Start Guide

## Prerequisites

1. **Node.js 18+** must be installed
   ```bash
   node --version  # Should show v18 or higher
   ```

2. **API must be running** on `http://localhost:8000`
   ```bash
   # In another terminal
   cd /Users/olivierdore/Documents/Github/ClassificationEcommerce
   source venv/bin/activate
   python3 src/api.py
   ```

## Start the Frontend

1. **Install dependencies** (first time only):
   ```bash
   cd frontend-nextjs
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Open browser**:
   Navigate to http://localhost:3000

## Features

### Home Page (`/`)
- **Test Products**: Load testset and test 10 random products
- **Manual Test**: Enter a product title/description to classify
- **Results Table**: View predictions with confidence scores
- **Statistics**: See accuracy, average confidence, and latency

### Categories Page (`/categories`)
- Browse all 100 product categories
- See example product titles for each category
- Search categories by name, ID, or example

## Testing the Connection

Before starting the frontend, test that the API is working:

```bash
cd frontend-nextjs
./test-api-connection.sh
```

All endpoints should return ✓ (check marks).

## Troubleshooting

- **"Cannot find module 'next'"**: Run `npm install` in the `frontend-nextjs` directory
- **API connection errors**: Make sure the API is running on port 8000
- **CORS errors**: The API should have CORS enabled (already configured)
- **Port 3000 in use**: Next.js will automatically use the next available port

