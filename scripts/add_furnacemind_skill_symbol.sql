-- Add first-class FurnaceMind skill symbols.
-- Symbols are unique short labels shown on quick-skill buttons.

ALTER TABLE furnace_mind.skills
ADD COLUMN IF NOT EXISTS symbol varchar(16);

UPDATE furnace_mind.skills
SET symbol = NULLIF(BTRIM(metadata->>'symbol'), '')
WHERE (symbol IS NULL OR BTRIM(symbol) = '')
  AND metadata IS NOT NULL
  AND NULLIF(BTRIM(metadata->>'symbol'), '') IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_furnacemind_skills_symbol_lower
ON furnace_mind.skills (LOWER(symbol))
WHERE symbol IS NOT NULL AND BTRIM(symbol) <> '';