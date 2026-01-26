# Ecommerce Classification Frontend

Next.js 14 frontend with Tailwind CSS for testing the e-commerce product classification API.

## Features

- **Interactive Product Testing**: Test products manually or load from testset
- **Real-time Results**: View predictions with confidence scores and latency
- **Statistics Dashboard**: See accuracy, average confidence, and latency metrics
- **Category Browser**: Browse all 100 categories with examples

## Getting Started

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## Configuration

The default API URL is `http://localhost:8000`. You can change it in the UI or modify the `DEFAULT_API_URL` constant in the components.

## Project Structure

```
frontend-nextjs/
├── app/
│   ├── page.tsx          # Home page with product tester
│   ├── categories/
│   │   └── page.tsx      # Categories browser
│   ├── layout.tsx        # Root layout
│   ├── globals.css       # Global styles with Tailwind
│   └── types.ts          # TypeScript types
├── components/
│   ├── ProductTester.tsx # Product testing component
│   ├── ResultsTable.tsx  # Results display table
│   └── StatsCards.tsx    # Statistics cards
└── package.json
```

## API Endpoints Used

- `POST /classify` - Classify a product
- `GET /category-names` - Get all category names and examples
- `GET /testset` - Get test dataset CSV


