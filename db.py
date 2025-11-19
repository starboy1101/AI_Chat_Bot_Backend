import logging
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger("swarai.db")

try:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase environment not configured")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Supabase client initialized successfully.")
except Exception as e:
    supabase = None
    logger.warning(f"⚠️ Supabase client not configured: {e}")


# ---------- BASIC OPERATIONS ---------- #

def insert_row(table: str, payload: dict):
    if not supabase:
        raise RuntimeError("Supabase not configured")
    return supabase.table(table).insert(payload).execute()


def select_rows(table: str, filters: dict | None = None, order: tuple | None = None, limit: int | None = None):
    if not supabase:
        raise RuntimeError("Supabase not configured")

    query = supabase.table(table).select("*")

    if filters:
        for k, v in filters.items():
            query = query.filter(k, "eq", v)

    if order:
        field, desc = order
        query = query.order(field, desc=desc)

    if limit:
        query = query.limit(limit)

    return query.execute()


def delete_rows(table: str, where: dict):
    if not supabase:
        raise RuntimeError("Supabase not configured")

    query = supabase.table(table).delete()
    for k, v in where.items():
        query = query.filter(k, "eq", v)
    return query.execute()


def update_row(table: str, where: dict, payload: dict):
    if not supabase:
        raise RuntimeError("Supabase not configured")

    query = supabase.table(table).update(payload)
    for k, v in where.items():
        query = query.filter(k, "eq", v)
    return query.execute()
