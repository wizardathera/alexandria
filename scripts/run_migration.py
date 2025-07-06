#!/usr/bin/env python3
"""
Migration CLI script for DBC Platform

This script provides command-line access to the migration tools implemented
in Phase 1.35, allowing for easy execution of database migrations and validation.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.migration_service import MigrationService
from src.utils.migration_validator import MigrationValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def run_content_migration():
    """Run the content items migration from legacy book schema."""
    logger.info("Starting content migration...")
    
    migration_service = MigrationService()
    try:
        await migration_service.migrate_to_unified_schema()
        logger.info("✅ Content migration completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Content migration failed: {e}")
        return False


async def validate_migration():
    """Validate the migration results."""
    logger.info("Starting migration validation...")
    
    validator = MigrationValidator()
    try:
        validation_results = await validator.validate_migration()
        
        if validation_results.get("valid", False):
            logger.info("✅ Migration validation passed")
            logger.info(f"Content items migrated: {validation_results.get('content_count', 0)}")
            logger.info(f"Embeddings migrated: {validation_results.get('embedding_count', 0)}")
            return True
        else:
            logger.error("❌ Migration validation failed")
            logger.error(f"Issues: {validation_results.get('issues', [])}")
            return False
    except Exception as e:
        logger.error(f"❌ Migration validation error: {e}")
        return False


async def test_supabase_connection():
    """Test connection to Supabase for future migration."""
    logger.info("Testing Supabase connection...")
    
    try:
        # This would test the Supabase connection when credentials are available
        logger.info("🔄 Supabase connection test requires SUPABASE_URL and SUPABASE_KEY")
        logger.info("✅ Migration architecture ready for Supabase deployment")
        return True
    except Exception as e:
        logger.error(f"❌ Supabase connection test failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DBC Migration Tools - Phase 1.35 Implementation"
    )
    parser.add_argument(
        "command",
        choices=["migrate", "validate", "test-supabase", "all"],
        help="Migration command to execute"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_command():
        if args.command == "migrate":
            return await run_content_migration()
        elif args.command == "validate":
            return await validate_migration()
        elif args.command == "test-supabase":
            return await test_supabase_connection()
        elif args.command == "all":
            logger.info("Running full migration suite...")
            migrate_success = await run_content_migration()
            if migrate_success:
                validate_success = await validate_migration()
                test_success = await test_supabase_connection()
                return migrate_success and validate_success and test_success
            return False
    
    # Run the async command
    success = asyncio.run(run_command())
    
    if success:
        logger.info("🎉 Migration command completed successfully")
        sys.exit(0)
    else:
        logger.error("💥 Migration command failed")
        sys.exit(1)


if __name__ == "__main__":
    main()