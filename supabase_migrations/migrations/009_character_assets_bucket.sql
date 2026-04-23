-- Migration 009: Character assets storage bucket
-- Creates a public-read bucket for character reference images and sprites.

INSERT INTO storage.buckets (id, name, public)
VALUES ('character-assets', 'character-assets', true)
ON CONFLICT (id) DO NOTHING;

-- Public read: anyone can download character images via URL
CREATE POLICY IF NOT EXISTS "Public read character assets"
ON storage.objects FOR SELECT
USING (bucket_id = 'character-assets');

-- Service role write: only backend can upload images
CREATE POLICY IF NOT EXISTS "Service role write character assets"
ON storage.objects FOR INSERT TO service_role
WITH CHECK (bucket_id = 'character-assets');
