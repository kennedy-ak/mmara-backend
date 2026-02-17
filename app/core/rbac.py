"""
Role-Based Access Control (RBAC) utilities.
"""

from enum import Enum
from typing import Set

from fastapi import HTTPException, status


class Role(str, Enum):
    """User roles."""

    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class Permission(str, Enum):
    """Permissions for different operations."""

    # Chat permissions
    CHAT_SEND = "chat:send"
    CHAT_HISTORY = "chat:history"
    CHAT_DELETE = "chat:delete"

    # Document permissions
    DOCUMENT_VIEW = "document:view"
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_REINDEX = "document:reindex"

    # Admin permissions
    USER_MANAGE = "user:manage"
    USER_VIEW = "user:view"
    ANALYTICS_VIEW = "analytics:view"
    SETTINGS_MANAGE = "settings:manage"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.GUEST: {
        Permission.CHAT_SEND,
    },
    Role.USER: {
        Permission.CHAT_SEND,
        Permission.CHAT_HISTORY,
        Permission.CHAT_DELETE,
        Permission.DOCUMENT_VIEW,
    },
    Role.ADMIN: {
        # Admins have all permissions
        *[p for p in Permission]
    },
}


def has_permission(role: Role, permission: Permission) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        role: User role
        permission: Permission to check

    Returns:
        bool: True if role has permission
    """
    return permission in ROLE_PERMISSIONS.get(role, set())


def require_role(required_role: Role):
    """
    Decorator to require a specific role.

    Args:
        required_role: Required role

    Returns:
        Function that raises exception if role doesn't match
    """

    def checker(user_role: Role) -> None:
        if user_role != required_role and user_role != Role.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Role '{required_role}' required"
            )

    return checker


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission.

    Args:
        permission: Required permission

    Returns:
        Function that raises exception if permission not granted
    """

    def checker(user_role: Role) -> None:
        if not has_permission(user_role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required"
            )

    return checker


class RBACChecker:
    """Helper class for checking RBAC in endpoints."""

    @staticmethod
    def check_permission(user_role: str, permission: Permission) -> bool:
        """Check if user has permission."""
        try:
            role = Role(user_role)
            return has_permission(role, permission)
        except ValueError:
            return False

    @staticmethod
    def require_permission(user_role: str, permission: Permission):
        """Raise exception if user lacks permission."""
        if not RBACChecker.check_permission(user_role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required"
            )

    @staticmethod
    def is_admin(user_role: str) -> bool:
        """Check if user is admin."""
        return user_role == Role.ADMIN
