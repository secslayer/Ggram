from typing import List, Optional

from sqlalchemy.orm import Session

from core.security import get_password_hash, verify_password
from models.user import User


class UserService:
    """Service for user management."""

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get a user by email."""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        """Get a user by username."""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination."""
        return db.query(User).offset(skip).limit(limit).all()

    @staticmethod
    def create(db: Session, username: str, email: str, password: str, full_name: str = None) -> User:
        """Create a new user."""
        hashed_password = get_password_hash(password)
        db_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def update(db: Session, user_id: int, **kwargs) -> Optional[User]:
        """Update a user."""
        db_user = UserService.get_by_id(db, user_id)
        if not db_user:
            return None

        # Update password if provided
        if "password" in kwargs:
            kwargs["hashed_password"] = get_password_hash(kwargs.pop("password"))

        # Update user attributes
        for key, value in kwargs.items():
            if hasattr(db_user, key):
                setattr(db_user, key, value)

        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def delete(db: Session, user_id: int) -> bool:
        """Delete a user."""
        db_user = UserService.get_by_id(db, user_id)
        if not db_user:
            return False

        db.delete(db_user)
        db.commit()
        return True

    @staticmethod
    def authenticate(db: Session, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        # Try to find user by username or email
        user = (
            UserService.get_by_username(db, username_or_email) or
            UserService.get_by_email(db, username_or_email)
        )

        if not user:
            return None

        # Verify password
        if not verify_password(password, user.hashed_password):
            return None

        return user