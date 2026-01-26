/**
 * URL API par défaut (sans slash final).
 * Next charge .env.production lors de "next build". En dev, fallback http://localhost:8000.
 */
export const DEFAULT_API_URL = (
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
).replace(/\/$/, '');
