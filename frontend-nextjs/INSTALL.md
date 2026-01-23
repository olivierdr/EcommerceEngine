# Installation Instructions

## Prerequisites

You need to have Node.js 18+ and npm installed on your system.

### Install Node.js

**macOS (using Homebrew):**
```bash
brew install node
```

**Or download from:**
https://nodejs.org/

### Verify Installation

```bash
node --version  # Should be v18 or higher
npm --version   # Should be 9 or higher
```

## Setup

1. **Install dependencies:**
```bash
cd frontend-nextjs
npm install
```

2. **Start the development server:**
```bash
npm run dev
```

3. **Open your browser:**
Navigate to http://localhost:3000

## Make sure the API is running

The frontend expects the API to be running on `http://localhost:8000` by default.

Start the API in a separate terminal:
```bash
cd /Users/olivierdore/Documents/Github/ClassificationEcommerce
source venv/bin/activate
python3 src/api.py
# Or use: ./start_api.sh
```

## Troubleshooting

- **Port 3000 already in use?** Next.js will automatically use the next available port (3001, 3002, etc.)
- **API connection errors?** Make sure the API is running on the correct port and CORS is enabled
- **Build errors?** Try deleting `node_modules` and `.next` folders, then run `npm install` again

