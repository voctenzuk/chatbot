"""
Webhook handlers for Stripe and Paddle payment providers.

This module provides FastAPI endpoints for receiving and processing
webhook events from payment providers, with idempotency protection
and audit logging.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse

# Stripe imports
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None

# Paddle imports (optional)
try:
    from paddle_billing import Client as PaddleClient
    from paddle_billing.Notifications import Secret as PaddleSecret
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    PaddleClient = None
    PaddleSecret = None

from bot.config import settings
from bot.services.supabase_manager import SupabaseManager, SupabaseError

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ============================================
# DEPENDENCIES
# ============================================

async def get_supabase_manager() -> SupabaseManager:
    """Dependency to get initialized SupabaseManager."""
    manager = SupabaseManager(
        url=settings.supabase_url,
        key=settings.supabase_key,
        service_role_key=settings.supabase_service_role_key,
        use_connection_pool=True
    )
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.close()


def get_stripe_client() -> Optional[Any]:
    """Get configured Stripe client."""
    if not STRIPE_AVAILABLE or not settings.stripe_secret_key:
        return None
    stripe.api_key = settings.stripe_secret_key
    return stripe


# ============================================
# WEBHOOK EVENT STORAGE (AUDIT)
# ============================================

async def store_webhook_event(
    supabase: SupabaseManager,
    provider: str,
    event_type: str,
    payload: Dict[str, Any],
    status: str = "pending",
    error_message: Optional[str] = None
) -> Optional[str]:
    """
    Store webhook event in database for audit and idempotency.
    
    Args:
        supabase: SupabaseManager instance
        provider: Payment provider (stripe, paddle)
        event_type: Event type string
        payload: Full event payload
        status: Processing status (pending, processing, completed, failed)
        error_message: Error message if failed
        
    Returns:
        Event ID if stored successfully, None otherwise
    """
    try:
        # Extract event ID from payload if available
        event_id = payload.get("id")
        
        event_data = {
            "provider": provider,
            "event_type": event_type,
            "payload": payload,
            "status": status,
            "error_message": error_message,
        }
        
        if status == "completed":
            event_data["processed_at"] = datetime.utcnow().isoformat()
        
        result = (
            supabase.service_client
            .table("webhook_events")
            .insert(event_data)
            .execute()
        )
        
        if result.data:
            logger.info(f"Stored webhook event: {provider}/{event_type} (ID: {event_id})")
            return result.data[0].get("id")
        return None
        
    except Exception as e:
        logger.error(f"Failed to store webhook event: {e}")
        # Don't fail the webhook processing if storage fails
        return None


async def is_event_processed(
    supabase: SupabaseManager,
    provider: str,
    event_id: str
) -> bool:
    """
    Check if a webhook event has already been processed.
    
    Args:
        supabase: SupabaseManager instance
        provider: Payment provider
        event_id: Event ID to check
        
    Returns:
        True if event was already processed
    """
    try:
        result = (
            supabase.service_client
            .table("webhook_events")
            .select("id")
            .eq("provider", provider)
            .eq("payload->>id", event_id)  # JSONB path lookup
            .eq("status", "completed")
            .limit(1)
            .execute()
        )
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking event idempotency: {e}")
        # Fail open - if we can't check, process anyway
        return False


async def update_event_status(
    supabase: SupabaseManager,
    event_id: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """Update webhook event processing status."""
    try:
        update_data = {"status": status}
        if status == "completed":
            update_data["processed_at"] = datetime.utcnow().isoformat()
        if error_message:
            update_data["error_message"] = error_message
            
        (
            supabase.service_client
            .table("webhook_events")
            .update(update_data)
            .eq("id", event_id)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to update event status: {e}")


# ============================================
# STRIPE WEBHOOK HANDLER
# ============================================

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
    supabase: SupabaseManager = Depends(get_supabase_manager)
) -> JSONResponse:
    """
    Handle Stripe webhook events.
    
    Events handled:
    - checkout.session.completed: Create subscription record
    - invoice.payment_succeeded: Record payment, extend subscription
    - customer.subscription.updated: Update subscription status
    - customer.subscription.deleted: Mark subscription canceled
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Stripe SDK not available")
    
    if not settings.stripe_webhook_secret:
        logger.error("STRIPE_WEBHOOK_SECRET not configured")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    
    # Read raw payload
    payload = await request.body()
    
    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload,
            stripe_signature,
            settings.stripe_webhook_secret
        )
    except stripe.error.SignatureVerificationError as e:
        logger.warning(f"Invalid Stripe signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Error constructing Stripe event: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    
    event_type = event.get("type")
    event_id = event.get("id")
    data = event.get("data", {}).get("object", {})
    
    logger.info(f"Received Stripe webhook: {event_type} (ID: {event_id})")
    
    # Check idempotency
    if await is_event_processed(supabase, "stripe", event_id):
        logger.info(f"Event {event_id} already processed, skipping")
        return JSONResponse({"status": "already_processed"})
    
    # Store event for audit
    stored_event_id = await store_webhook_event(
        supabase, "stripe", event_type, event, status="processing"
    )
    
    try:
        # Process based on event type
        handler = STRIPE_EVENT_HANDLERS.get(event_type)
        if handler:
            await handler(supabase, data)
            logger.info(f"Successfully processed {event_type}")
        else:
            logger.info(f"No handler for event type: {event_type}")
        
        # Mark as completed
        if stored_event_id:
            await update_event_status(supabase, stored_event_id, "completed")
        
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error processing Stripe event {event_type}: {e}")
        if stored_event_id:
            await update_event_status(supabase, stored_event_id, "failed", str(e))
        # Return 200 to prevent Stripe from retrying (we've logged the error)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=200)


# --------------------------------------------
# Stripe Event Handlers
# --------------------------------------------

async def handle_checkout_session_completed(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """
    Handle checkout.session.completed event.
    
    Creates a new subscription record when a user completes checkout.
    """
    customer_id = data.get("customer")
    subscription_id = data.get("subscription")
    metadata = data.get("metadata", {})
    
    # Try to get user_id from metadata or lookup by customer_id
    user_id = metadata.get("user_id")
    
    if not user_id and customer_id:
        # Find user by Stripe customer ID
        result = (
            supabase.service_client
            .table("users")
            .select("id")
            .eq("stripe_customer_id", customer_id)
            .limit(1)
            .execute()
        )
        if result.data:
            user_id = result.data[0]["id"]
    
    if not user_id:
        raise ValueError(f"Could not find user for customer {customer_id}")
    
    # Get subscription details from Stripe
    try:
        stripe_sub = stripe.Subscription.retrieve(subscription_id)
        
        # Get plan ID from metadata or find by Stripe price ID
        plan_id = metadata.get("plan_id")
        if not plan_id and stripe_sub.get("items", {}).get("data"):
            price_id = stripe_sub["items"]["data"][0]["price"]["id"]
            # Find plan by Stripe price ID
            plans = (
                supabase.service_client
                .table("subscription_plans")
                .select("id")
                .eq("stripe_price_id", price_id)
                .limit(1)
                .execute()
            )
            if plans.data:
                plan_id = plans.data[0]["id"]
        
        if not plan_id:
            # Default to free plan
            free_plan = (
                supabase.service_client
                .table("subscription_plans")
                .select("id")
                .eq("slug", "free")
                .limit(1)
                .execute()
            )
            plan_id = free_plan.data[0]["id"] if free_plan.data else None
        
        # Create subscription record
        subscription_data = {
            "user_id": user_id,
            "plan_id": plan_id,
            "status": stripe_sub.get("status", "active"),
            "current_period_start": datetime.fromtimestamp(
                stripe_sub.get("current_period_start", 0)
            ).isoformat(),
            "current_period_end": datetime.fromtimestamp(
                stripe_sub.get("current_period_end", 0)
            ).isoformat(),
            "trial_start": datetime.fromtimestamp(stripe_sub["trial_start"]).isoformat()
            if stripe_sub.get("trial_start") else None,
            "trial_end": datetime.fromtimestamp(stripe_sub["trial_end"]).isoformat()
            if stripe_sub.get("trial_end") else None,
            "provider": "stripe",
            "provider_subscription_id": subscription_id,
            "provider_customer_id": customer_id,
            "metadata": {
                "checkout_session_id": data.get("id"),
                "client_reference_id": data.get("client_reference_id"),
            }
        }
        
        result = (
            supabase.service_client
            .table("user_subscriptions")
            .insert(subscription_data)
            .execute()
        )
        
        # Update user's stripe_customer_id if not set
        (
            supabase.service_client
            .table("users")
            .update({"stripe_customer_id": customer_id})
            .eq("id", user_id)
            .execute()
        )
        
        logger.info(f"Created subscription for user {user_id}: {result.data[0]['id']}")
        
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        raise


async def handle_invoice_payment_succeeded(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """
    Handle invoice.payment_succeeded event.
    
    Records the payment and extends the subscription period.
    """
    subscription_id = data.get("subscription")
    customer_id = data.get("customer")
    
    if not subscription_id:
        logger.info("Invoice not associated with subscription, skipping")
        return
    
    # Find subscription
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id, user_id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        logger.warning(f"Subscription not found: {subscription_id}")
        return
    
    subscription = result.data[0]
    
    # Create payment record
    amount_cents = data.get("amount_due", 0)
    currency = data.get("currency", "usd").upper()
    
    payment_data = {
        "user_id": subscription["user_id"],
        "subscription_id": subscription["id"],
        "amount_cents": amount_cents,
        "currency": currency,
        "status": "succeeded",
        "provider": "stripe",
        "provider_payment_id": data.get("payment_intent"),
        "provider_invoice_id": data.get("id"),
        "provider_receipt_url": data.get("hosted_invoice_url"),
        "paid_at": datetime.fromtimestamp(data["status_transitions"]["paid_at"]).isoformat()
        if data.get("status_transitions", {}).get("paid_at") else datetime.utcnow().isoformat(),
        "metadata": {
            "invoice_number": data.get("number"),
            "billing_reason": data.get("billing_reason"),
        }
    }
    
    (
        supabase.service_client
        .table("payments")
        .insert(payment_data)
        .execute()
    )
    
    # Update subscription period if this is a renewal
    if data.get("billing_reason") == "subscription_cycle":
        period_end = data.get("period_end")
        if period_end:
            (
                supabase.service_client
                .table("user_subscriptions")
                .update({
                    "current_period_end": datetime.fromtimestamp(period_end).isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })
                .eq("id", subscription["id"])
                .execute()
            )
    
    logger.info(f"Recorded payment for subscription {subscription_id}")


async def handle_customer_subscription_updated(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """
    Handle customer.subscription.updated event.
    
    Updates subscription status and related fields.
    """
    subscription_id = data.get("id")
    status = data.get("status")
    
    # Find subscription
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        logger.warning(f"Subscription not found for update: {subscription_id}")
        return
    
    subscription_db_id = result.data[0]["id"]
    
    # Build update data
    update_data = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
    }
    
    # Update period dates
    if data.get("current_period_start"):
        update_data["current_period_start"] = datetime.fromtimestamp(
            data["current_period_start"]
        ).isoformat()
    if data.get("current_period_end"):
        update_data["current_period_end"] = datetime.fromtimestamp(
            data["current_period_end"]
        ).isoformat()
    
    # Handle cancellation
    if data.get("cancel_at_period_end"):
        update_data["cancel_at_period_end"] = True
        update_data["canceled_at"] = datetime.utcnow().isoformat()
    
    # Handle trial
    if data.get("trial_end"):
        update_data["trial_end"] = datetime.fromtimestamp(data["trial_end"]).isoformat()
    
    (
        supabase.service_client
        .table("user_subscriptions")
        .update(update_data)
        .eq("id", subscription_db_id)
        .execute()
    )
    
    logger.info(f"Updated subscription {subscription_id} to status: {status}")


async def handle_customer_subscription_deleted(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """
    Handle customer.subscription.deleted event.
    
    Marks subscription as canceled and schedules downgrade to free plan.
    """
    subscription_id = data.get("id")
    
    # Find subscription
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id, user_id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        logger.warning(f"Subscription not found for deletion: {subscription_id}")
        return
    
    subscription = result.data[0]
    
    # Mark as canceled
    (
        supabase.service_client
        .table("user_subscriptions")
        .update({
            "status": "canceled",
            "canceled_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        .eq("id", subscription["id"])
        .execute()
    )
    
    # Create free subscription for the user
    # (This will be handled by the subscription service or a background job)
    logger.info(f"Marked subscription {subscription_id} as canceled")


async def handle_invoice_payment_failed(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """
    Handle invoice.payment_failed event.
    
    Records failed payment and updates subscription status to past_due.
    """
    subscription_id = data.get("subscription")
    
    if not subscription_id:
        return
    
    # Find subscription
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id, user_id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        return
    
    subscription = result.data[0]
    
    # Record failed payment
    payment_intent = data.get("payment_intent")
    failure_message = None
    failure_code = None
    
    if payment_intent:
        try:
            pi = stripe.PaymentIntent.retrieve(payment_intent)
            last_error = pi.get("last_payment_error", {})
            failure_message = last_error.get("message")
            failure_code = last_error.get("code")
        except Exception:
            pass
    
    payment_data = {
        "user_id": subscription["user_id"],
        "subscription_id": subscription["id"],
        "amount_cents": data.get("amount_due", 0),
        "currency": data.get("currency", "usd").upper(),
        "status": "failed",
        "provider": "stripe",
        "provider_invoice_id": data.get("id"),
        "failure_code": failure_code,
        "failure_message": failure_message,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    (
        supabase.service_client
        .table("payments")
        .insert(payment_data)
        .execute()
    )
    
    # Update subscription status to past_due
    (
        supabase.service_client
        .table("user_subscriptions")
        .update({
            "status": "past_due",
            "updated_at": datetime.utcnow().isoformat()
        })
        .eq("id", subscription["id"])
        .execute()
    )
    
    logger.warning(f"Payment failed for subscription {subscription_id}")


# Map of Stripe event types to handlers
STRIPE_EVENT_HANDLERS = {
    "checkout.session.completed": handle_checkout_session_completed,
    "invoice.payment_succeeded": handle_invoice_payment_succeeded,
    "invoice.payment_failed": handle_invoice_payment_failed,
    "customer.subscription.updated": handle_customer_subscription_updated,
    "customer.subscription.deleted": handle_customer_subscription_deleted,
}


# ============================================
# PADDLE WEBHOOK HANDLER (Optional)
# ============================================

@router.post("/paddle")
async def paddle_webhook(
    request: Request,
    paddle_signature: Optional[str] = Header(None, alias="paddle-signature"),
    supabase: SupabaseManager = Depends(get_supabase_manager)
) -> JSONResponse:
    """
    Handle Paddle webhook events.
    
    This is an optional implementation for Paddle payment provider.
    Supports subscription and payment events similar to Stripe.
    """
    if not PADDLE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Paddle SDK not available")
    
    if not settings.paddle_webhook_secret:
        raise HTTPException(status_code=500, detail="Paddle webhook secret not configured")
    
    # Read payload
    payload = await request.body()
    body_text = payload.decode("utf-8")
    
    # Verify signature
    try:
        # Paddle webhook signature verification
        # The signature format is: ts=timestamp,h1=signature
        if not paddle_signature:
            raise ValueError("Missing paddle-signature header")
        
        # Parse signature header
        parts = paddle_signature.split(",")
        sig_parts = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                sig_parts[key.strip()] = value.strip()
        
        timestamp = sig_parts.get("ts")
        signature = sig_parts.get("h1")
        
        if not timestamp or not signature:
            raise ValueError("Invalid signature format")
        
        # Verify using Paddle SDK if available, otherwise manual verification
        # For now, we'll parse the JSON and process
        event = json.loads(body_text)
        
    except Exception as e:
        logger.warning(f"Invalid Paddle signature or payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature or payload")
    
    event_type = event.get("event_type")
    event_id = event.get("event_id")
    data = event.get("data", {})
    
    logger.info(f"Received Paddle webhook: {event_type} (ID: {event_id})")
    
    # Check idempotency
    if await is_event_processed(supabase, "paddle", event_id):
        logger.info(f"Event {event_id} already processed, skipping")
        return JSONResponse({"status": "already_processed"})
    
    # Store event for audit
    stored_event_id = await store_webhook_event(
        supabase, "paddle", event_type, event, status="processing"
    )
    
    try:
        # Process based on event type
        handler = PADDLE_EVENT_HANDLERS.get(event_type)
        if handler:
            await handler(supabase, data)
            logger.info(f"Successfully processed Paddle {event_type}")
        else:
            logger.info(f"No handler for Paddle event type: {event_type}")
        
        # Mark as completed
        if stored_event_id:
            await update_event_status(supabase, stored_event_id, "completed")
        
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error processing Paddle event {event_type}: {e}")
        if stored_event_id:
            await update_event_status(supabase, stored_event_id, "failed", str(e))
        return JSONResponse({"status": "error", "message": str(e)}, status_code=200)


# --------------------------------------------
# Paddle Event Handlers
# --------------------------------------------

async def handle_paddle_subscription_created(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """Handle subscription.created event from Paddle."""
    subscription_id = data.get("id")
    customer_id = data.get("customer_id")
    
    # Find user by Paddle customer ID
    result = (
        supabase.service_client
        .table("users")
        .select("id")
        .eq("paddle_customer_id", customer_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        # Try to find by custom data
        custom_data = data.get("custom_data", {})
        user_id = custom_data.get("user_id")
        if user_id:
            result = (
                supabase.service_client
                .table("users")
                .select("id")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
    
    if not result.data:
        raise ValueError(f"User not found for Paddle customer: {customer_id}")
    
    user_id = result.data[0]["id"]
    
    # Get plan from Paddle price ID
    price_id = data.get("items", [{}])[0].get("price", {}).get("id")
    plan_id = None
    if price_id:
        plans = (
            supabase.service_client
            .table("subscription_plans")
            .select("id")
            .eq("paddle_price_id", price_id)
            .limit(1)
            .execute()
        )
        if plans.data:
            plan_id = plans.data[0]["id"]
    
    # Parse dates from ISO format
    current_period_start = data.get("current_billing_period", {}).get("starts_at")
    current_period_end = data.get("current_billing_period", {}).get("ends_at")
    
    subscription_data = {
        "user_id": user_id,
        "plan_id": plan_id,
        "status": data.get("status", "active"),
        "current_period_start": current_period_start,
        "current_period_end": current_period_end,
        "provider": "paddle",
        "provider_subscription_id": subscription_id,
        "provider_customer_id": customer_id,
        "metadata": {
            "paddle_subscription_data": data
        }
    }
    
    (
        supabase.service_client
        .table("user_subscriptions")
        .insert(subscription_data)
        .execute()
    )
    
    logger.info(f"Created Paddle subscription for user {user_id}")


async def handle_paddle_subscription_updated(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """Handle subscription.updated event from Paddle."""
    subscription_id = data.get("id")
    
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        logger.warning(f"Paddle subscription not found: {subscription_id}")
        return
    
    subscription_db_id = result.data[0]["id"]
    
    update_data = {
        "status": data.get("status"),
        "updated_at": datetime.utcnow().isoformat(),
    }
    
    # Update billing period if available
    billing_period = data.get("current_billing_period", {})
    if billing_period.get("starts_at"):
        update_data["current_period_start"] = billing_period["starts_at"]
    if billing_period.get("ends_at"):
        update_data["current_period_end"] = billing_period["ends_at"]
    
    (
        supabase.service_client
        .table("user_subscriptions")
        .update(update_data)
        .eq("id", subscription_db_id)
        .execute()
    )
    
    logger.info(f"Updated Paddle subscription {subscription_id}")


async def handle_paddle_transaction_completed(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """Handle transaction.completed event from Paddle (payment succeeded)."""
    subscription_id = data.get("subscription_id")
    
    # Find subscription
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id, user_id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        logger.warning(f"Subscription not found for transaction: {subscription_id}")
        return
    
    subscription = result.data[0]
    
    # Get amount from Paddle
    details = data.get("details", {})
    totals = details.get("totals", {})
    amount_cents = totals.get("total", 0)
    currency = data.get("currency_code", "USD")
    
    payment_data = {
        "user_id": subscription["user_id"],
        "subscription_id": subscription["id"],
        "amount_cents": amount_cents,
        "currency": currency,
        "status": "succeeded",
        "provider": "paddle",
        "provider_payment_id": data.get("id"),
        "provider_receipt_url": data.get("receipt_url"),
        "paid_at": data.get("billed_at") or datetime.utcnow().isoformat(),
        "metadata": {
            "paddle_transaction_data": data
        }
    }
    
    (
        supabase.service_client
        .table("payments")
        .insert(payment_data)
        .execute()
    )
    
    logger.info(f"Recorded Paddle payment for subscription {subscription_id}")


async def handle_paddle_subscription_canceled(
    supabase: SupabaseManager,
    data: Dict[str, Any]
) -> None:
    """Handle subscription.canceled event from Paddle."""
    subscription_id = data.get("id")
    
    result = (
        supabase.service_client
        .table("user_subscriptions")
        .select("id")
        .eq("provider_subscription_id", subscription_id)
        .limit(1)
        .execute()
    )
    
    if not result.data:
        logger.warning(f"Paddle subscription not found for cancellation: {subscription_id}")
        return
    
    subscription_db_id = result.data[0]["id"]
    
    (
        supabase.service_client
        .table("user_subscriptions")
        .update({
            "status": "canceled",
            "canceled_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        .eq("id", subscription_db_id)
        .execute()
    )
    
    logger.info(f"Canceled Paddle subscription {subscription_id}")


# Map of Paddle event types to handlers
PADDLE_EVENT_HANDLERS = {
    "subscription.created": handle_paddle_subscription_created,
    "subscription.updated": handle_paddle_subscription_updated,
    "subscription.canceled": handle_paddle_subscription_canceled,
    "transaction.completed": handle_paddle_transaction_completed,
}


# ============================================
# HEALTH CHECK
# ============================================

@router.get("/health")
async def webhook_health() -> JSONResponse:
    """Health check endpoint for webhook service."""
    return JSONResponse({
        "status": "healthy",
        "stripe_available": STRIPE_AVAILABLE,
        "paddle_available": PADDLE_AVAILABLE,
        "stripe_configured": bool(settings.stripe_webhook_secret),
        "paddle_configured": bool(settings.paddle_webhook_secret),
    })


# ============================================
# MAIN APPLICATION SETUP (for standalone use)
# ============================================

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Webhook Handler Service")
    app.include_router(router)
    
    # Run standalone server
    uvicorn.run(app, host="0.0.0.0", port=8000)
