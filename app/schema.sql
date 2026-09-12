PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS study_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotators (
    code TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks (
    task_uuid TEXT PRIMARY KEY,
    task_type TEXT NOT NULL CHECK (task_type IN ('frame', 'clip')),
    split TEXT NOT NULL CHECK (split IN ('test', 'train_alignment', 'calibration')),
    clip_id INTEGER NOT NULL,
    frame_index INTEGER CHECK (frame_index BETWEEN 1 AND 16 OR frame_index IS NULL),
    source_task_uuid TEXT,
    repeat_kind TEXT NOT NULL DEFAULT 'none'
        CHECK (repeat_kind IN ('none', 'second_rater', 'within_rater')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_task_uuid) REFERENCES tasks(task_uuid),
    CHECK ((task_type = 'frame' AND frame_index IS NOT NULL) OR
           (task_type = 'clip' AND frame_index IS NULL))
);

CREATE TABLE IF NOT EXISTS assignments (
    assignment_uuid TEXT PRIMARY KEY,
    task_uuid TEXT NOT NULL,
    annotator_code TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('primary', 'second', 'hidden_repeat', 'adjudication')),
    queue_position INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'started', 'complete', 'skipped')),
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    FOREIGN KEY (task_uuid) REFERENCES tasks(task_uuid),
    FOREIGN KEY (annotator_code) REFERENCES annotators(code),
    UNIQUE (task_uuid, annotator_code, role)
);

CREATE TABLE IF NOT EXISTS frame_ratings (
    assignment_uuid TEXT PRIMARY KEY,
    visible_category TEXT NOT NULL CHECK (visible_category IN
        ('Happiness','Sadness','Neutral','Anger','Surprise','Disgust','Fear','Mixed','Unclear','Face not visible')),
    intensity INTEGER NOT NULL CHECK (intensity BETWEEN 0 AND 6),
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assignment_uuid) REFERENCES assignments(assignment_uuid)
);

CREATE TABLE IF NOT EXISTS clip_ratings (
    assignment_uuid TEXT PRIMARY KEY,
    dominant_category TEXT NOT NULL CHECK (dominant_category IN
        ('Happiness','Sadness','Neutral','Anger','Surprise','Disgust','Fear','Mixed','Unclear','Face not visible')),
    intensities_json TEXT NOT NULL,
    occlusion INTEGER NOT NULL DEFAULT 0 CHECK (occlusion IN (0,1)),
    speaking INTEGER NOT NULL DEFAULT 0 CHECK (speaking IN (0,1)),
    abrupt_change INTEGER NOT NULL DEFAULT 0 CHECK (abrupt_change IN (0,1)),
    subject_switch INTEGER NOT NULL DEFAULT 0 CHECK (subject_switch IN (0,1)),
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assignment_uuid) REFERENCES assignments(assignment_uuid)
);

CREATE TABLE IF NOT EXISTS drafts (
    assignment_uuid TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assignment_uuid) REFERENCES assignments(assignment_uuid)
);

CREATE TABLE IF NOT EXISTS import_log (
    bundle_id TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_path TEXT NOT NULL,
    rows_imported INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS adjudication_cases (
    task_uuid TEXT PRIMARY KEY,
    reasons_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','complete')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_uuid) REFERENCES tasks(task_uuid)
);

CREATE INDEX IF NOT EXISTS idx_assignments_annotator_status_queue
ON assignments(annotator_code, status, queue_position);

CREATE INDEX IF NOT EXISTS idx_assignments_task
ON assignments(task_uuid);

CREATE INDEX IF NOT EXISTS idx_tasks_target
ON tasks(task_type, split, clip_id, frame_index);

CREATE INDEX IF NOT EXISTS idx_tasks_source
ON tasks(source_task_uuid);
