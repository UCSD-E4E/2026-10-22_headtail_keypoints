-- Label canvas size and coordinate source per head/tail annotation, to settle which images were
-- labelled on the 3987x3016 vs 4014x3016 canvas before training (laser_pairs.py notes both exist).
-- Only numeric ids and sizes: no names, paths or emails. Belongs in cscw-fishsense2027/sql/.
-- Run read-only against the restored production DB (container p2-fishsense-db):
--   docker exec -i -e PGOPTIONS='-c default_transaction_read_only=on' p2-fishsense-db \
--     psql -U postgres -d fishsense -A -F '|' -t < sql/extract_headtail_canvas.sql > data/db_extracts/headtail_canvas.psv
-- Columns: image_id|ann_id|original_width|original_height|n_keypoints|label_completed|label_superseded
SELECT l.image_id, (j.ann->>'id')::bigint,
       (SELECT max((r->>'original_width')::int)  FROM jsonb_array_elements(j.ann->'result') r),
       (SELECT max((r->>'original_height')::int) FROM jsonb_array_elements(j.ann->'result') r),
       (SELECT count(*) FROM jsonb_array_elements(j.ann->'result') r WHERE r->>'type' = 'keypointlabels'),
       l.completed, coalesce(l.superseded, false)
FROM headtaillabel l CROSS JOIN LATERAL jsonb_array_elements((l.label_studio_json::jsonb)->'annotations') j(ann)
WHERE NOT coalesce((j.ann->>'was_cancelled')::bool, false);
