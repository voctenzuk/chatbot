#!/usr/bin/env python3
"""CLI command for memory cleanup maintenance.

This script provides a command-line interface for running memory cleanup
operations. It supports dry-run mode for safety and configurable parameters.

Examples:
    # Preview cleanup for specific users (dry-run)
    python -m bot.scripts.memory_cleanup --users 12345 67890 --dry-run

    # Actually run cleanup for specific users
    python -m bot.scripts.memory_cleanup --users 12345 67890

    # Run with custom config
    MEMORY_DEFAULT_TTL_DAYS=30 MEMORY_MIN_IMPORTANCE=0.5 \
        python -m bot.scripts.memory_cleanup --users 12345

    # Output JSON report
    python -m bot.scripts.memory_cleanup --users 12345 --json

Environment Variables:
    MEMORY_DEFAULT_TTL_DAYS: Base TTL in days (default: 90)
    MEMORY_DECAY_RATE: Daily decay rate (default: 0.01)
    MEMORY_MIN_IMPORTANCE: Minimum importance threshold (default: 0.3)
    MEMORY_CLEANUP_DRY_RUN: Default dry-run mode (default: true)
    MEMORY_MAX_DELETIONS_PER_RUN: Safety limit on deletions (default: 1000)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from loguru import logger

from bot.services.mem0_memory_service import Mem0MemoryService, get_memory_service
from bot.services.memory_cleanup import CleanupConfig, MemoryCleanupService


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the cleanup script."""
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Memory cleanup maintenance for chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --users 12345 67890 --dry-run     # Preview cleanup
  %(prog)s --users 12345                     # Run actual cleanup
  %(prog)s --users 12345 --json              # Output JSON report
  %(prog)s --config                          # Show current config
        """,
    )

    parser.add_argument(
        "--users",
        nargs="+",
        type=int,
        metavar="USER_ID",
        help="User IDs to process (space-separated)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Preview what would be deleted without making changes",
    )

    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually perform deletions (overrides --dry-run)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--ttl-days",
        type=int,
        default=None,
        help=f"Override default TTL days (default: {os.getenv('MEMORY_DEFAULT_TTL_DAYS', '90')})",
    )

    parser.add_argument(
        "--min-importance",
        type=float,
        default=None,
        help=f"Override minimum importance (default: {os.getenv('MEMORY_MIN_IMPORTANCE', '0.3')})",
    )

    parser.add_argument(
        "--max-deletions",
        type=int,
        default=None,
        help="Override max deletions per run",
    )

    return parser.parse_args()


def show_config() -> None:
    """Display current configuration."""
    config = CleanupConfig()

    print("=" * 60)
    print("Memory Cleanup Configuration")
    print("=" * 60)
    print()

    print("Base Settings:")
    print(f"  Default TTL days:        {config.default_ttl_days}")
    print(f"  Decay rate:              {config.importance_decay_rate} ({config.importance_decay_rate*100:.1f}% per day)")
    print(f"  Min importance:          {config.min_importance_threshold}")
    print(f"  Max memories per user:   {config.max_memories_per_user}")
    print(f"  Max deletions per run:   {config.max_deletions_per_run}")
    print(f"  Batch size:              {config.batch_size}")
    print()

    print("Safety Settings:")
    print(f"  Dry run default:         {config.dry_run_default}")
    print()

    print("Category TTL Multipliers:")
    for category, multiplier in sorted(config.category_multipliers.items(), key=lambda x: x[0].value):
        days = int(config.default_ttl_days * multiplier)
        print(f"  {category.value:20} {multiplier:4.1f}x ({days} days)")
    print()

    print("Type Importance Floors:")
    for mem_type, floor in sorted(config.type_importance_floor.items(), key=lambda x: x[0].value):
        print(f"  {mem_type.value:20} {floor:.1f}")
    print()

    print("Boost Settings:")
    print(f"  Emotional valence boost: {config.emotional_valence_boost}")
    print(f"  Access threshold:        {config.access_count_boost_threshold}")
    print(f"  Access boost value:      {config.access_count_boost_value}")
    print()

    print("Environment Variable Overrides:")
    env_vars = [
        "MEMORY_DEFAULT_TTL_DAYS",
        "MEMORY_DECAY_RATE",
        "MEMORY_MIN_IMPORTANCE",
        "MEMORY_CLEANUP_DRY_RUN",
        "MEMORY_MAX_DELETIONS_PER_RUN",
        "MEMORY_EMOTIONAL_BOOST",
        "MEMORY_ACCESS_THRESHOLD",
        "MEMORY_ACCESS_BOOST",
    ]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"  {var}={value}")
    print()

    print("=" * 60)


async def run_cleanup(args: argparse.Namespace) -> int:
    """Run the cleanup operation.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Determine dry-run mode
    dry_run = True  # Default to safe mode
    if args.no_dry_run:
        dry_run = False
    elif args.dry_run is not None:
        dry_run = args.dry_run

    # Build custom config if overrides provided
    config_overrides = {}
    if args.ttl_days is not None:
        config_overrides["default_ttl_days"] = args.ttl_days
    if args.min_importance is not None:
        config_overrides["min_importance_threshold"] = args.min_importance
    if args.max_deletions is not None:
        config_overrides["max_deletions_per_run"] = args.max_deletions

    config = None
    if config_overrides:
        # Create config with overrides
        original_env = {}
        for key, value in config_overrides.items():
            env_key = f"MEMORY_{key.upper()}"
            original_env[env_key] = os.getenv(env_key)
            os.environ[env_key] = str(value)

        config = CleanupConfig()

        # Restore original env
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # Initialize services
    try:
        memory_service = get_memory_service()
        cleanup_service = MemoryCleanupService(
            memory_service=memory_service,
            config=config,
        )
    except Exception as e:
        logger.error("Failed to initialize services: {}", e)
        return 1

    # Run cleanup
    if args.users:
        logger.info(
            "Running cleanup for {} users (dry_run={})",
            len(args.users),
            dry_run,
        )

        report = await cleanup_service.cleanup_all(
            user_ids=args.users,
            dry_run=dry_run,
        )

        # Output results
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("\n" + "=" * 60)
            print("Cleanup Report")
            print("=" * 60)
            print(f"Mode:              {'DRY RUN (preview only)' if dry_run else 'LIVE (changes made)'}")
            print(f"Users processed:   {len(report.users_processed)}")
            print(f"Memories scanned:  {report.total_memories_scanned}")
            print(f"Memories kept:     {report.memories_kept}")
            print(f"Memories decayed:  {report.memories_decayed}")
            print(f"Memories expired:  {report.memories_expired}")
            print(f"Memories deleted:  {report.memories_deleted}")
            print(f"Duration:          {report.duration_seconds:.2f}s")

            if report.errors:
                print(f"\nErrors ({len(report.errors)}):")
                for error in report.errors:
                    print(f"  - {error}")

            if dry_run and report.memories_expired > 0:
                print(f"\nTo actually delete these memories, run with --no-dry-run")

            print("=" * 60)

        return 0 if not report.errors else 1
    else:
        logger.error("No users specified. Use --users to specify user IDs.")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(verbose=args.verbose)

    if args.config:
        show_config()
        return 0

    if not args.users:
        print("Error: No users specified. Use --users to specify user IDs.", file=sys.stderr)
        print("Use --help for usage information.", file=sys.stderr)
        return 1

    try:
        return asyncio.run(run_cleanup(args))
    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Unexpected error during cleanup")
        return 1


if __name__ == "__main__":
    sys.exit(main())
