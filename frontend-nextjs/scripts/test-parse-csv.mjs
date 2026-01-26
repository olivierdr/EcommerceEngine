/**
 * Test du parser CSV et du mapping produits (même logique que ProductTester).
 * À lancer avec : node scripts/test-parse-csv.mjs
 */
import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (c === '"' && next === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        inQuotes = false;
      } else {
        field += c;
      }
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ',') {
        row.push(field.trim());
        field = '';
      } else if (c === '\n' || c === '\r') {
        if (c === '\r' && next === '\n') i++;
        row.push(field.trim());
        field = '';
        if (row.some((cell) => cell.length > 0)) rows.push(row);
        row = [];
      } else {
        field += c;
      }
    }
  }
  if (field || row.length > 0) {
    row.push(field.trim());
    if (row.some((cell) => cell.length > 0)) rows.push(row);
  }
  return rows;
}

// === Test 1: CSV avec guillemets et virgules (comme le front) ===
const sampleCSV = 'product_id,category_id,category_path,title,description\n' +
  'id1,cat1,path/a/b,"Title with, comma",desc';
const parsedSample = parseCSV(sampleCSV);
const headers = parsedSample[0].map((h) => h.trim().toLowerCase());
const row1 = parsedSample[1] || [];
const toProduct = (values) => {
  const row = {};
  headers.forEach((key, idx) => {
    const v = values[idx]?.trim() ?? '';
    if (key === 'product_id') row.product_id = v;
    else if (key === 'title') row.title = v;
    else if (key === 'description') row.description = v;
    else if (key === 'category_id') row.category_id = v;
    else if (key === 'category_name') row.category_name = v;
    else if (key === 'category_path') row.category_path = v;
  });
  return row;
};
const p1 = toProduct(row1);
const ok1 = p1.title === 'Title with, comma' && p1.category_id === 'cat1' && p1.product_id === 'id1';
if (!ok1) {
  console.error('FAIL mapping (quoted comma):', p1);
  process.exit(1);
}
console.log('Test 1 (quoted comma + mapping): OK');

// === Test 2: Fichier testset réel ===
const testsetPath = join(__dirname, '../../src/data/testset.csv');
const text = readFileSync(testsetPath, 'utf-8');
const parsed = parseCSV(text);
const realHeaders = parsed[0].map((h) => h.trim().toLowerCase());
const hasTitle = realHeaders.includes('title');
const hasCategoryId = realHeaders.includes('category_id');
const realRow1 = parsed[1] || [];
const get = (key) => realRow1[realHeaders.indexOf(key)] ?? '';
const title1 = get('title').trim();
const categoryId1 = get('category_id').trim();

const ok2 =
  parsed.length >= 2 &&
  hasTitle &&
  hasCategoryId &&
  title1.length > 0 &&
  categoryId1.length > 0 &&
  title1.includes(',');

if (!ok2) {
  console.error('FAIL testset réel:', { parsedLength: parsed.length, hasTitle, hasCategoryId, title1: title1?.slice(0, 50), categoryId1 });
  process.exit(1);
}
console.log('Test 2 (testset réel):', parsed.length, 'rows, title avec virgule OK, category_id:', categoryId1);

console.log('Tous les tests CSV/mapping OK.');
process.exit(0);
