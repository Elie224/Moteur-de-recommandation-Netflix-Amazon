"""Authentication helpers."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import User
from ..security import hash_password, verify_password


def get_user_by_email(db: Session, email: str) -> User | None:
    stmt = select(User).where(User.email == email.lower())
    return db.execute(stmt).scalar_one_or_none()


def create_user(db: Session, *, email: str, password: str, display_name=None, preferred_language: str = "fr", country: str = "FR", is_admin: bool = False) -> User:
    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        display_name=display_name,
        preferred_language=preferred_language,
        country=country,
        is_admin=is_admin,
    )
    db.add(user)
    db.flush()
    return user


def authenticate(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
