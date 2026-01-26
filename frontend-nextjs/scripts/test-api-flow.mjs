/**
 * Test flux API : /testset puis /classify.
 * Préréquis : API en écoute sur API_URL (ex. http://localhost:8000).
 * Exécution : API_URL=http://localhost:8000 node scripts/test-api-flow.mjs
 */
const API_URL = (process.env.API_URL || 'http://localhost:8000').replace(/\/$/, '');

async function main() {
  console.log('API_URL:', API_URL);

  let text;
  try {
    const r = await fetch(`${API_URL}/testset`);
    if (!r.ok) throw new Error(`testset ${r.status}: ${await r.text()}`);
    text = await r.text();
  } catch (e) {
    console.error('Fetch /testset failed:', e.message);
    console.log('(Démarrez l’API avec: uvicorn src.api:app --reload)');
    process.exit(1);
  }

  const lines = text.split('\n').filter((l) => l.trim());
  const firstLine = lines[0] || '';
  const hasProductId = /product_id/i.test(firstLine);
  const hasTitle = /title/i.test(firstLine);
  const hasCategoryId = /category_id/i.test(firstLine);
  if (!hasProductId || !hasTitle || !hasCategoryId) {
    console.error('FAIL testset format: missing product_id/title/category_id in header');
    process.exit(1);
  }
  console.log('GET /testset: OK (header has product_id, title, category_id)');

  const classifyRes = await fetch(`${API_URL}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'iPhone 14 Pro Max', description: 'Smartphone Apple' }),
  });
  if (!classifyRes.ok) {
    const body = await classifyRes.text();
    console.error('POST /classify failed:', classifyRes.status, body);
    process.exit(1);
  }
  const data = await classifyRes.json();
  const has = data && typeof data.category_id === 'string' && typeof data.confidence === 'number';
  if (!has) {
    console.error('FAIL /classify response shape:', JSON.stringify(data));
    process.exit(1);
  }
  console.log('POST /classify: OK', { category_id: data.category_id, confidence: data.confidence });
  console.log('All API flow checks passed.');
}

main();
