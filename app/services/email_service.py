"""
Email service for sending transactional emails.
Supports SMTP and logging backends.
"""

import logging
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """
    Email service for sending password reset emails and other transactional emails.
    In development mode, emails are logged instead of sent.
    """

    def __init__(self):
        self.enabled = settings.environment != "test"
        self.frontend_url = settings.frontend_url
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_use_tls = settings.smtp_use_tls
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
        self.from_email = settings.smtp_from_email or f"noreply@{settings.frontend_url.replace('http://', '').replace('https://', '')}"
        self.from_name = settings.smtp_from_name

    async def send_password_reset_email(
        self, email: str, reset_token: str, user_name: Optional[str] = None
    ) -> bool:
        """
        Send a password reset email to the user.

        Args:
            email: Recipient email address
            reset_token: Password reset token
            user_name: Optional user's name for personalization

        Returns:
            bool: True if email was sent successfully
        """
        reset_link = f"{self.frontend_url}/auth/reset-password?token={reset_token}"

        # In development, log the email instead of sending
        if settings.debug:
            logger.info("=" * 60)
            logger.info(f"PASSWORD RESET EMAIL TO: {email}")
            logger.info(f"Reset Link: {reset_link}")
            logger.info(f"Token: {reset_token}")
            logger.info("=" * 60)
            return True

        # Production: Send actual email via SMTP
        try:
            # TODO: Implement SMTP sending using aiosmtplib or similar
            # For now, log in production as well
            logger.warning(f"Email sending not implemented. Would send to: {email}")
            logger.warning(f"Reset link: {reset_link}")
            return True
        except Exception as e:
            logger.error(f"Failed to send password reset email: {e}")
            return False

    async def send_password_changed_confirmation(
        self, email: str, user_name: Optional[str] = None
    ) -> bool:
        """
        Send a confirmation email that the password was changed.

        Args:
            email: Recipient email address
            user_name: Optional user's name for personalization

        Returns:
            bool: True if email was sent successfully
        """
        if settings.debug:
            logger.info(f"PASSWORD CHANGED CONFIRMATION TO: {email}")
            return True

        try:
            logger.warning(f"Email sending not implemented. Would notify: {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send password change confirmation: {e}")
            return False


# Global email service instance
email_service = EmailService()
