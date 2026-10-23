-- Raw path, checksum and camera for every image with a head/tail label, for staging crops from the NAS.
-- PRIVATE: paths can contain people's names. Write the output under data/private/ (gitignored), never commit it.
--   docker exec -i -e PGOPTIONS='-c default_transaction_read_only=on' p2-fishsense-db \
--     psql -U postgres -d fishsense -A -F '|' -t < sql/extract_image_paths.sql > data/private/image_paths.psv
-- Columns: image_id|dive_id|camera_id|checksum|dive_path|image_path
SELECT i.id, i.dive_id, coalesce(i.camera_id, d.camera_id), i.checksum, d.path, i.path
FROM image i JOIN dive d ON d.id = i.dive_id
WHERE i.id IN (SELECT image_id FROM headtaillabel)
ORDER BY i.id;
