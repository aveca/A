/*
# Create products table — product recognition memory

1. New Tables
- `products`
  - `id` (uuid, primary key)
  - `name` (text, not null) — human label for the product, e.g. "Lu Petit Beurre"
  - `barcode` (text, unique) — EAN-13 / UPC / QR identifier, nullable (some products have no readable barcode)
  - `price` (numeric, nullable) — last known price extracted from packaging
  - `currency` (text, nullable) — currency code or symbol, e.g. "EUR", "XOF", "USD"
  - `detected_text` (text, nullable) — raw OCR text captured from the packaging
  - `keywords` (text[], nullable) — tokenized lowercased words from name + detected_text, used for fuzzy recall by OCR
  - `color_fingerprint` (jsonb, nullable) — { avg: [r,g,b], hist: [..16 buckets] } visual signature for shape/texture recall
  - `thumbnail` (text, nullable) — small data URL of the captured frame for visual reference
  - `category` (text, nullable) — auto-guessed category (beverage, food, dairy, etc.)
  - `times_scanned` (int, default 0) — how many times the product has been re-recognized
  - `created_at` (timestamptz, default now())
  - `updated_at` (timestamptz, default now())

2. Indexes
- Unique index on `barcode` (partial, where not null) for fast barcode lookup
- GIN index on `keywords` for fast text-array overlap queries
- Index on `name` for library listing

3. Security
- Enable RLS on `products`.
- This is a single-tenant app with NO sign-in screen, so the anon-key frontend must read and write its own data.
- Four CRUD policies scoped TO anon, authenticated (intentionally shared public data).
*/

CREATE TABLE IF NOT EXISTS products (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  barcode text UNIQUE,
  price numeric,
  currency text,
  detected_text text,
  keywords text[],
  color_fingerprint jsonb,
  thumbnail text,
  category text,
  times_scanned integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE products ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_select_products" ON products;
CREATE POLICY "anon_select_products" ON products FOR SELECT
  TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_insert_products" ON products;
CREATE POLICY "anon_insert_products" ON products FOR INSERT
  TO anon, authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "anon_update_products" ON products;
CREATE POLICY "anon_update_products" ON products FOR UPDATE
  TO anon, authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "anon_delete_products" ON products;
CREATE POLICY "anon_delete_products" ON products FOR DELETE
  TO anon, authenticated USING (true);

CREATE UNIQUE INDEX IF NOT EXISTS products_barcode_idx
  ON products (barcode) WHERE barcode IS NOT NULL;

CREATE INDEX IF NOT EXISTS products_keywords_gin_idx
  ON products USING GIN (keywords);

CREATE INDEX IF NOT EXISTS products_name_idx
  ON products (name);

CREATE INDEX IF NOT EXISTS products_category_idx
  ON products (category);