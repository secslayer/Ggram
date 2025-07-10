from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api import schemas
from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.user_service import UserService

router = APIRouter()


@router.post("/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(user_in: schemas.UserCreate, db: Session = Depends(get_db)) -> Any:
    """Create a new user."""
    # Check if user with this email already exists
    user = UserService.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )
    
    # Check if user with this username already exists
    user = UserService.get_by_username(db, username=user_in.username)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this username already exists.",
        )
    
    # Create the user
    user = UserService.create(
        db=db,
        username=user_in.username,
        email=user_in.email,
        password=user_in.password,
        full_name=user_in.full_name,
    )
    return user


@router.get("/", response_model=List[schemas.User])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Retrieve users."""
    users = UserService.get_all(db, skip=skip, limit=limit)
    return users


@router.get("/me", response_model=schemas.User)
def read_user_me(current_user: User = Depends(get_current_active_user)) -> Any:
    """Get current user."""
    return current_user


@router.get("/{user_id}", response_model=schemas.User)
def read_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get a specific user by id."""
    user = UserService.get_by_id(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.put("/me", response_model=schemas.User)
def update_user_me(
    user_in: schemas.UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update own user."""
    # Check if email is being updated and already exists
    if user_in.email and user_in.email != current_user.email:
        user = UserService.get_by_email(db, email=user_in.email)
        if user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists.",
            )
    
    # Check if username is being updated and already exists
    if user_in.username and user_in.username != current_user.username:
        user = UserService.get_by_username(db, username=user_in.username)
        if user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this username already exists.",
            )
    
    # Update the user
    user = UserService.update(
        db=db,
        user_id=current_user.id,
        **user_in.dict(exclude_unset=True),
    )
    return user


@router.delete("/me", response_model=schemas.User)
def delete_user_me(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete own user."""
    user = current_user
    UserService.delete(db, user_id=user.id)
    return user