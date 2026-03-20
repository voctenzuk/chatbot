# Memory Cleanup (TTL/Decay Maintenance)

This document describes the memory cleanup system for preventing memory bloat
through TTL (Time-To-Live) expiration and importance-based decay heuristics.

## Overview

The memory cleanup service implements automated maintenance to prevent unbounded
growth of stored memories. It uses a combination of:

- **TTL-based expiration**: Memories expire after a category-specific time period
- **Importance decay**: Memory importance decreases over time (memories "fade")
- **Access boosting**: Frequently accessed memories are retained longer
- **Emotional weight**: High-emotion memories receive importance boosts

## Quick Start

### Preview Cleanup (Dry Run)

Always start with a dry run to preview what would be cleaned:

```bash
# Preview for specific users
python -m bot.scripts.memory_cleanup --users 12345 67890 --dry-run

# Show configuration
python -m bot.scripts.memory_cleanup --config

# Verbose output with JSON report
python -m bot.scripts.memory_cleanup --users 12345 --dry-run --verbose --json
```

### Run Actual Cleanup

```bash
# Clean specific users (requires --no-dry-run)
python -m bot.scripts.memory_cleanup --users 12345 --no-dry-run

# With custom TTL
MEMORY_DEFAULT_TTL_DAYS=30 python -m bot.scripts.memory_cleanup --users 12345 --no-dry-run
```

## Configuration

All parameters are tunable via environment variables:

### Base Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_DEFAULT_TTL_DAYS` | 90 | Base TTL for memories without explicit expiration |
| `MEMORY_DECAY_RATE` | 0.01 | Daily decay rate (1% = 0.01) |
| `MEMORY_MIN_IMPORTANCE` | 0.3 | Minimum importance to retain a memory |
| `MEMORY_MAX_PER_USER` | 10000 | Soft limit for memories per user |

### Safety Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_CLEANUP_DRY_RUN` | true | Default to dry-run mode |
| `MEMORY_MAX_DELETIONS_PER_RUN` | 1000 | Maximum deletions per cleanup run |
| `MEMORY_CLEANUP_BATCH_SIZE` | 100 | Processing batch size |

### Boost Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_EMOTIONAL_BOOST` | 0.2 | Importance boost for high-emotion memories |
| `MEMORY_ACCESS_THRESHOLD` | 5 | Access count threshold for boost |
| `MEMORY_ACCESS_BOOST` | 0.3 | Importance boost for frequently accessed |

## Category TTL Multipliers

Different memory categories have different retention periods:

| Category | Multiplier | Days (default) | Description |
|----------|-----------|----------------|-------------|
| Semantic (facts) | 2.0x | 180 | Long-lived factual knowledge |
| Relationship | 3.0x | 270 | Relationship milestones |
| Procedural | 2.5x | 225 | Habits and routines |
| Preference | 1.5x | 135 | User preferences |
| Emotional | 0.75x | 68 | Emotional memories |
| Episodic | 0.5x | 45 | Conversation episodes |

## Type Importance Floors

Certain memory types have minimum importance floors to prevent accidental deletion:

| Type | Floor | Rationale |
|------|-------|-----------|
| Boundary | 2.0 | User boundaries are critical |
| Identity | 1.5 | Core user identity facts |
| Milestone | 1.5 | Relationship milestones |
| Goal | 1.2 | User goals and aspirations |
| Habit | 1.0 | User habits and routines |

## Scheduling Cleanup Jobs

### Using Cron

```bash
# Daily cleanup at 3 AM (dry-run by default for safety)
0 3 * * * cd /path/to/chatbot && python -m bot.scripts.memory_cleanup --users $(psql -c "SELECT telegram_id FROM users" | tr '\n' ' ') >> /var/log/memory-cleanup.log 2>&1

# Weekly aggressive cleanup (actual deletions)
0 2 * * 0 cd /path/to/chatbot && MEMORY_DEFAULT_TTL_DAYS=60 python -m bot.scripts.memory_cleanup --users $(psql -c "SELECT telegram_id FROM users" | tr '\n' ' ') --no-dry-run >> /var/log/memory-cleanup.log 2>&1
```

### Using Systemd Timer

**`/etc/systemd/system/chatbot-memory-cleanup.service`**:
```ini
[Unit]
Description=Chatbot Memory Cleanup
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/opt/chatbot
Environment=MEMORY_CLEANUP_DRY_RUN=false
Environment=PYTHONPATH=/opt/chatbot/src
ExecStart=/opt/chatbot/.venv/bin/python -m bot.scripts.memory_cleanup --users 12345 67890
User=chatbot
```

**`/etc/systemd/system/chatbot-memory-cleanup.timer`**:
```ini
[Unit]
Description=Run chatbot memory cleanup daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable chatbot-memory-cleanup.timer
sudo systemctl start chatbot-memory-cleanup.timer
```

### Using Docker

```yaml
# docker-compose.yml
services:
  memory-cleanup:
    image: chatbot:latest
    command: python -m bot.scripts.memory_cleanup --users 12345 --no-dry-run
    environment:
      - MEMORY_DEFAULT_TTL_DAYS=90
      - MEMORY_CLEANUP_DRY_RUN=false
      - MEM0_HOST=http://mem0:8000
    depends_on:
      - mem0
    deploy:
      replicas: 0  # Don't run continuously
    labels:
      # For ofelia scheduler
      ofelia.enabled: "true"
      ofelia.job-run.schedule: "0 0 3 * * *"
```

With [Ofelia](https://github.com/mcuadros/ofelia) scheduler:
```yaml
  scheduler:
    image: mcuadros/ofelia:latest
    command: daemon --docker
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
```

### Using Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: chatbot-memory-cleanup
spec:
  schedule: "0 3 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: ghcr.io/voctenzuk/chatbot:latest
            command:
            - python
            - -m
            - bot.scripts.memory_cleanup
            - --users
            - "12345"
            - "67890"
            env:
            - name: MEMORY_CLEANUP_DRY_RUN
              value: "false"
            - name: MEMORY_DEFAULT_TTL_DAYS
              value: "90"
            - name: MEM0_HOST
              value: "http://mem0:8000"
          restartPolicy: OnFailure
```

## Safety Features

### Dry Run by Default

The cleanup script defaults to dry-run mode. You must explicitly use `--no-dry-run`
to perform actual deletions.

### Maximum Deletions Limit

The `MEMORY_MAX_DELETIONS_PER_RUN` setting prevents runaway cleanup jobs from
deleting too many memories at once.

### Importance Floors

Critical memory types (boundaries, identity) have minimum importance floors that
prevent their deletion regardless of age.

### High Importance Override

Memories with importance >= 1.5 are kept even if their TTL has expired.

## Monitoring and Alerting

### Logging

The cleanup script logs all operations using loguru:

```python
# In your application
from bot.services.memory_cleanup import get_cleanup_service

service = get_cleanup_service()
report = await service.cleanup_all(user_ids=[12345], dry_run=False)

logger.info(
    "Memory cleanup completed: {} scanned, {} deleted, {} errors",
    report.total_memories_scanned,
    report.memories_deleted,
    len(report.errors),
)
```

### Metrics

Track these key metrics:

- `memories_scanned_total`: Total memories evaluated
- `memories_deleted_total`: Total memories deleted
- `cleanup_duration_seconds`: Time taken for cleanup
- `cleanup_errors_total`: Number of errors during cleanup

### Alerts

Set up alerts for:

1. **High deletion rate**: If >50% of memories are being deleted, review config
2. **Cleanup failures**: Any errors during cleanup
3. **Long runtime**: Cleanup taking >10 minutes

Example Prometheus alerting rule:
```yaml
groups:
- name: chatbot-memory
  rules:
  - alert: HighMemoryDeletionRate
    expr: rate(memories_deleted_total[1h]) / rate(memories_scanned_total[1h]) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory deletion rate detected"

  - alert: MemoryCleanupErrors
    expr: cleanup_errors_total > 0
    for: 1m
    labels:
      severity: critical
```

## Tuning Guide

### Conservative Settings (Safe Default)

```bash
MEMORY_DEFAULT_TTL_DAYS=180
MEMORY_DECAY_RATE=0.005
MEMORY_MIN_IMPORTANCE=0.5
```

### Aggressive Settings (Storage-Constrained)

```bash
MEMORY_DEFAULT_TTL_DAYS=60
MEMORY_DECAY_RATE=0.02
MEMORY_MIN_IMPORTANCE=0.4
```

### Per-User Tuning

You may want different settings for different user tiers:

```python
# Free tier - more aggressive cleanup
free_config = CleanupConfig(
    default_ttl_days=60,
    max_memories_per_user=5000,
)

# Pro tier - retain more
pro_config = CleanupConfig(
    default_ttl_days=180,
    max_memories_per_user=50000,
)
```

## Troubleshooting

### Too Many Memories Being Deleted

1. Increase `MEMORY_DEFAULT_TTL_DAYS`
2. Decrease `MEMORY_DECAY_RATE`
3. Lower `MEMORY_MIN_IMPORTANCE`
4. Check category multipliers

### Memory Bloat Not Reducing

1. Decrease `MEMORY_DEFAULT_TTL_DAYS`
2. Increase `MEMORY_DECAY_RATE`
3. Verify `--no-dry-run` is being used
4. Check for errors in logs

### Cleanup Taking Too Long

1. Reduce `MEMORY_CLEANUP_BATCH_SIZE`
2. Process users in smaller batches
3. Run cleanup more frequently (less to clean each time)

## Migration from No TTL

If you're enabling cleanup on an existing system with no TTL:

1. **Week 1-2**: Run with conservative settings in dry-run mode
   ```bash
   MEMORY_DEFAULT_TTL_DAYS=365 python -m bot.scripts.memory_cleanup --users ALL --dry-run
   ```

2. **Week 3**: Run actual cleanup with very conservative settings
   ```bash
   MEMORY_DEFAULT_TTL_DAYS=365 python -m bot.scripts.memory_cleanup --users ALL --no-dry-run
   ```

3. **Week 4+**: Gradually reduce TTL to target
   ```bash
   MEMORY_DEFAULT_TTL_DAYS=270 python -m bot.scripts.memory_cleanup --users ALL --no-dry-run
   # Then 180, 150, 120, 90...
   ```

## Code Integration

### Manual Cleanup Trigger

```python
from bot.services.memory_cleanup import MemoryCleanupService, CleanupConfig
from bot.services.mem0_memory_service import get_memory_service

async def cleanup_old_memories():
    memory_service = get_memory_service()
    
    config = CleanupConfig(
        default_ttl_days=90,
        min_importance_threshold=0.3,
    )
    
    cleanup_service = MemoryCleanupService(
        memory_service=memory_service,
        config=config,
    )
    
    # Preview
    preview = await cleanup_service.cleanup_user(
        user_id=12345,
        dry_run=True,
    )
    print(f"Would delete {preview.memories_expired} memories")
    
    # Execute
    report = await cleanup_service.cleanup_user(
        user_id=12345,
        dry_run=False,
    )
    print(f"Deleted {report.memories_deleted} memories")
```

### Periodic Cleanup in Application

```python
import asyncio
from bot.services.memory_cleanup import get_cleanup_service

async def periodic_cleanup():
    """Run cleanup every 24 hours."""
    cleanup_service = get_cleanup_service()
    
    while True:
        try:
            # Get active users from your database
            user_ids = await get_active_user_ids()
            
            report = await cleanup_service.cleanup_all(
                user_ids=user_ids,
                dry_run=False,
            )
            
            logger.info(
                "Periodic cleanup: {} users, {} deleted",
                len(report.users_processed),
                report.memories_deleted,
            )
        except Exception as e:
            logger.error("Cleanup failed: {}", e)
        
        # Wait 24 hours
        await asyncio.sleep(86400)
```

## See Also

- `ARCHITECTURE/MEMORY_DESIGN.md` - Overall memory system design
- `src/bot/services/memory_cleanup.py` - Cleanup service implementation
- `src/bot/scripts/memory_cleanup.py` - CLI script
- `tests/test_memory_cleanup.py` - Unit tests
