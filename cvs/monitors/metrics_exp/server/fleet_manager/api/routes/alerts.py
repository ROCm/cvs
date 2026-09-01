"""API routes for alert configuration management."""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...models import (
    get_db,
    AlertContactPoint,
    AlertRuleTemplate,
    AlertRule,
    MonitoringServer,
    NodeGroup,
)
from ...services import GrafanaProvisioner
from ..schemas import (
    AlertContactPointCreate,
    AlertContactPointUpdate,
    AlertContactPointResponse,
    AlertRuleTemplateResponse,
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertRuleResponse,
    BulkAlertRuleCreate,
    AlertConfigurationSummary,
    AlertSyncResponse,
    AlertTestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# ============================================
# Helper Functions
# ============================================


def get_contact_point_or_404(db: Session, contact_point_id: int) -> AlertContactPoint:
    """Get a contact point by ID or raise 404."""
    cp = db.query(AlertContactPoint).filter(AlertContactPoint.id == contact_point_id).first()
    if not cp:
        raise HTTPException(status_code=404, detail="Contact point not found")
    return cp


def get_alert_rule_or_404(db: Session, rule_id: int) -> AlertRule:
    """Get an alert rule by ID or raise 404."""
    rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return rule


def get_template_or_404(db: Session, template_id: int) -> AlertRuleTemplate:
    """Get an alert template by ID or raise 404."""
    template = db.query(AlertRuleTemplate).filter(AlertRuleTemplate.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Alert template not found")
    return template


def get_monitoring_server_or_404(db: Session, server_id: int) -> MonitoringServer:
    """Get a monitoring server by ID or raise 404."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")
    return server


def _redact_sensitive_settings(settings: dict, contact_type: str) -> tuple[dict, bool]:
    """Redact sensitive fields from contact point settings.

    Returns (redacted_settings, has_credentials).
    """
    sensitive_keys = {"password", "token", "api_key", "integration_key", "authorization_credentials", "url"}
    redacted = {}
    has_credentials = False

    for key, value in settings.items():
        if key.lower() in sensitive_keys and value:
            redacted[key] = "********"
            has_credentials = True
        else:
            redacted[key] = value

    return redacted, has_credentials


def build_contact_point_response(cp: AlertContactPoint) -> AlertContactPointResponse:
    """Build contact point response with redacted settings."""
    redacted_settings, has_credentials = _redact_sensitive_settings(cp.settings or {}, cp.contact_type)

    return AlertContactPointResponse(
        id=cp.id,
        name=cp.name,
        description=cp.description,
        contact_type=cp.contact_type,
        settings=redacted_settings,
        has_credentials=has_credentials,
        grafana_uid=cp.grafana_uid,
        last_synced_at=cp.last_synced_at,
        sync_error=cp.sync_error,
        created_at=cp.created_at,
        updated_at=cp.updated_at,
        alert_rule_count=len(cp.alert_rules),
    )


def build_alert_rule_response(rule: AlertRule) -> AlertRuleResponse:
    """Build alert rule response with relationship names."""
    return AlertRuleResponse(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        enabled=rule.enabled,
        severity=rule.severity,
        monitoring_server_id=rule.monitoring_server_id,
        template_id=rule.template_id,
        contact_point_id=rule.contact_point_id,
        node_group_id=rule.node_group_id,
        datasource_type=rule.datasource_type,
        query_expression=rule.query_expression,
        threshold_config=rule.threshold_config,
        labels=rule.labels or {},
        summary=rule.summary,
        runbook_url=rule.runbook_url,
        grafana_uid=rule.grafana_uid,
        grafana_folder_uid=rule.grafana_folder_uid or "fleet-alerts",
        last_synced_at=rule.last_synced_at,
        sync_error=rule.sync_error,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
        template_name=rule.template.display_name if rule.template else None,
        contact_point_name=rule.contact_point.name if rule.contact_point else None,
        node_group_name=rule.node_group.name if rule.node_group else None,
        monitoring_server_name=rule.monitoring_server.name if rule.monitoring_server else None,
    )


# ============================================
# Contact Point Endpoints
# ============================================


@router.get("/contact-points", response_model=List[AlertContactPointResponse])
def list_contact_points(db: Session = Depends(get_db)):
    """List all alert contact points."""
    contact_points = db.query(AlertContactPoint).order_by(AlertContactPoint.name).all()
    return [build_contact_point_response(cp) for cp in contact_points]


@router.post("/contact-points", response_model=AlertContactPointResponse, status_code=201)
async def create_contact_point(
    contact_point: AlertContactPointCreate,
    sync_to_grafana: bool = True,
    db: Session = Depends(get_db),
):
    """Create a new alert contact point."""
    # Check for duplicate name
    existing = db.query(AlertContactPoint).filter(AlertContactPoint.name == contact_point.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Contact point with this name already exists")

    # Create in database
    cp = AlertContactPoint(
        name=contact_point.name,
        description=contact_point.description,
        contact_type=contact_point.contact_type.value,
        settings=contact_point.settings,
    )
    db.add(cp)
    db.commit()
    db.refresh(cp)

    # Sync to Grafana if requested
    if sync_to_grafana:
        try:
            grafana = GrafanaProvisioner()
            result = await grafana.create_contact_point(
                name=cp.name,
                contact_type=cp.contact_type,
                settings=cp.settings,
            )
            if "error" not in result:
                cp.grafana_uid = result.get("uid")
                cp.last_synced_at = datetime.utcnow()
                cp.sync_error = None
            else:
                cp.sync_error = str(result.get("error", "Unknown error"))
            db.commit()
        except Exception as e:
            logger.error(f"Failed to sync contact point to Grafana: {e}")
            cp.sync_error = str(e)
            db.commit()

    return build_contact_point_response(cp)


@router.get("/contact-points/{contact_point_id}", response_model=AlertContactPointResponse)
def get_contact_point(contact_point_id: int, db: Session = Depends(get_db)):
    """Get a contact point by ID."""
    cp = get_contact_point_or_404(db, contact_point_id)
    return build_contact_point_response(cp)


@router.put("/contact-points/{contact_point_id}", response_model=AlertContactPointResponse)
async def update_contact_point(
    contact_point_id: int,
    update: AlertContactPointUpdate,
    sync_to_grafana: bool = True,
    db: Session = Depends(get_db),
):
    """Update a contact point."""
    cp = get_contact_point_or_404(db, contact_point_id)

    # Check for duplicate name if changing
    if update.name and update.name != cp.name:
        existing = db.query(AlertContactPoint).filter(AlertContactPoint.name == update.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Contact point with this name already exists")

    # Apply updates
    if update.name is not None:
        cp.name = update.name
    if update.description is not None:
        cp.description = update.description
    if update.contact_type is not None:
        cp.contact_type = update.contact_type.value
    if update.settings is not None:
        cp.settings = update.settings

    db.commit()

    # Sync to Grafana if requested and has UID
    if sync_to_grafana and cp.grafana_uid:
        try:
            grafana = GrafanaProvisioner()
            result = await grafana.update_contact_point(
                uid=cp.grafana_uid,
                name=cp.name,
                contact_type=cp.contact_type,
                settings=cp.settings,
            )
            if "error" not in result:
                cp.last_synced_at = datetime.utcnow()
                cp.sync_error = None
            else:
                cp.sync_error = str(result.get("error", "Unknown error"))
            db.commit()
        except Exception as e:
            logger.error(f"Failed to sync contact point to Grafana: {e}")
            cp.sync_error = str(e)
            db.commit()

    return build_contact_point_response(cp)


@router.delete("/contact-points/{contact_point_id}", status_code=204)
async def delete_contact_point(
    contact_point_id: int,
    force: bool = False,
    db: Session = Depends(get_db),
):
    """Delete a contact point.

    Args:
        contact_point_id: Contact point ID
        force: Delete even if used by alert rules (rules will have no contact point)
    """
    cp = get_contact_point_or_404(db, contact_point_id)

    # Check for associated rules
    if cp.alert_rules and not force:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete - used by {len(cp.alert_rules)} alert rules. Use force=true to delete anyway.",
        )

    # Delete from Grafana if synced
    if cp.grafana_uid:
        try:
            grafana = GrafanaProvisioner()
            await grafana.delete_contact_point(cp.grafana_uid)
        except Exception as e:
            logger.warning(f"Failed to delete contact point from Grafana: {e}")

    db.delete(cp)
    db.commit()


@router.post("/contact-points/{contact_point_id}/test", response_model=AlertTestResponse)
async def test_contact_point(contact_point_id: int, db: Session = Depends(get_db)):
    """Send a test notification to the contact point."""
    cp = get_contact_point_or_404(db, contact_point_id)

    if not cp.grafana_uid:
        raise HTTPException(
            status_code=400,
            detail="Contact point not synced to Grafana. Please sync first.",
        )

    try:
        grafana = GrafanaProvisioner()
        result = await grafana.test_contact_point(cp.grafana_uid)

        if "error" in result:
            return AlertTestResponse(success=False, message=str(result.get("error")))

        return AlertTestResponse(success=True, message="Test notification sent successfully")
    except Exception as e:
        logger.error(f"Failed to test contact point: {e}")
        return AlertTestResponse(success=False, message=str(e))


@router.post("/contact-points/{contact_point_id}/sync", response_model=AlertContactPointResponse)
async def sync_contact_point(contact_point_id: int, db: Session = Depends(get_db)):
    """Force sync contact point to Grafana."""
    cp = get_contact_point_or_404(db, contact_point_id)

    grafana = GrafanaProvisioner()

    try:
        if cp.grafana_uid:
            # Update existing
            result = await grafana.update_contact_point(
                uid=cp.grafana_uid,
                name=cp.name,
                contact_type=cp.contact_type,
                settings=cp.settings,
            )
        else:
            # Create new
            result = await grafana.create_contact_point(
                name=cp.name,
                contact_type=cp.contact_type,
                settings=cp.settings,
            )
            if "error" not in result:
                cp.grafana_uid = result.get("uid")

        if "error" not in result:
            cp.last_synced_at = datetime.utcnow()
            cp.sync_error = None
        else:
            cp.sync_error = str(result.get("error", "Unknown error"))

        db.commit()
    except Exception as e:
        logger.error(f"Failed to sync contact point: {e}")
        cp.sync_error = str(e)
        db.commit()

    return build_contact_point_response(cp)


# ============================================
# Alert Template Endpoints
# ============================================


@router.get("/templates", response_model=List[AlertRuleTemplateResponse])
def list_alert_templates(
    category: Optional[str] = None,
    defaults_only: bool = False,
    db: Session = Depends(get_db),
):
    """List all alert rule templates.

    Args:
        category: Filter by category (node_health, gpu_hardware, thermal, memory, network, logs)
        defaults_only: Only return default (pre-checked) templates
    """
    query = db.query(AlertRuleTemplate)

    if category:
        query = query.filter(AlertRuleTemplate.category == category)
    if defaults_only:
        query = query.filter(AlertRuleTemplate.is_default)

    templates = query.order_by(AlertRuleTemplate.is_default.desc(), AlertRuleTemplate.name).all()
    return templates


@router.get("/templates/{template_id}", response_model=AlertRuleTemplateResponse)
def get_alert_template(template_id: int, db: Session = Depends(get_db)):
    """Get an alert template by ID."""
    template = get_template_or_404(db, template_id)
    return template


# ============================================
# Alert Rule Endpoints
# ============================================


@router.get("/rules", response_model=List[AlertRuleResponse])
def list_alert_rules(
    monitoring_server_id: Optional[int] = None,
    node_group_id: Optional[int] = None,
    enabled: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    """List all alert rules with optional filtering."""
    query = db.query(AlertRule)

    if monitoring_server_id is not None:
        query = query.filter(AlertRule.monitoring_server_id == monitoring_server_id)
    if node_group_id is not None:
        query = query.filter(AlertRule.node_group_id == node_group_id)
    if enabled is not None:
        query = query.filter(AlertRule.enabled == enabled)

    rules = query.order_by(AlertRule.monitoring_server_id, AlertRule.name).all()
    return [build_alert_rule_response(r) for r in rules]


@router.post("/rules", response_model=AlertRuleResponse, status_code=201)
async def create_alert_rule(
    rule: AlertRuleCreate,
    sync_to_grafana: bool = True,
    db: Session = Depends(get_db),
):
    """Create a new alert rule."""
    # Validate monitoring server
    # server = get_monitoring_server_or_404(db, rule.monitoring_server_id)

    # Validate template if provided
    template = None
    if rule.template_id:
        template = get_template_or_404(db, rule.template_id)

    # Validate contact point if provided
    if rule.contact_point_id:
        get_contact_point_or_404(db, rule.contact_point_id)

    # Validate node group if provided
    if rule.node_group_id:
        ng = db.query(NodeGroup).filter(NodeGroup.id == rule.node_group_id).first()
        if not ng:
            raise HTTPException(status_code=404, detail="Node group not found")

    # Check for duplicate (same template + server + node_group)
    if rule.template_id:
        existing = (
            db.query(AlertRule)
            .filter(
                AlertRule.monitoring_server_id == rule.monitoring_server_id,
                AlertRule.template_id == rule.template_id,
                AlertRule.node_group_id == rule.node_group_id,
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Alert rule for this template/server/node_group combination already exists",
            )

    # Determine query expression and threshold
    query_expression = rule.query_expression
    threshold_config = rule.threshold_config.model_dump() if rule.threshold_config else None
    datasource_type = rule.datasource_type

    if template:
        query_expression = query_expression or template.query_expression
        threshold_config = threshold_config or template.default_threshold
        datasource_type = template.datasource_type

    if not query_expression:
        raise HTTPException(status_code=400, detail="Query expression required (from template or custom)")

    # Create rule
    alert_rule = AlertRule(
        name=rule.name,
        description=rule.description,
        monitoring_server_id=rule.monitoring_server_id,
        template_id=rule.template_id,
        contact_point_id=rule.contact_point_id,
        node_group_id=rule.node_group_id,
        enabled=rule.enabled,
        severity=rule.severity.value,
        datasource_type=datasource_type,
        query_expression=query_expression,
        threshold_config=threshold_config,
        labels=rule.labels,
        summary=rule.summary,
        runbook_url=rule.runbook_url,
    )
    db.add(alert_rule)
    db.commit()
    db.refresh(alert_rule)

    # Sync to Grafana if enabled and requested
    if sync_to_grafana and alert_rule.enabled:
        await _sync_rule_to_grafana(alert_rule, db)

    return build_alert_rule_response(alert_rule)


@router.post("/rules/bulk", response_model=List[AlertRuleResponse], status_code=201)
async def create_alert_rules_bulk(
    bulk: BulkAlertRuleCreate,
    sync_to_grafana: bool = True,
    db: Session = Depends(get_db),
):
    """Create multiple alert rules from templates."""
    # Validate monitoring server
    # server = get_monitoring_server_or_404(db, bulk.monitoring_server_id)

    # Validate contact point if provided
    if bulk.contact_point_id:
        get_contact_point_or_404(db, bulk.contact_point_id)

    # Validate node group if provided
    if bulk.node_group_id:
        ng = db.query(NodeGroup).filter(NodeGroup.id == bulk.node_group_id).first()
        if not ng:
            raise HTTPException(status_code=404, detail="Node group not found")

    created_rules = []

    for template_id in bulk.template_ids:
        template = db.query(AlertRuleTemplate).filter(AlertRuleTemplate.id == template_id).first()
        if not template:
            logger.warning(f"Template {template_id} not found, skipping")
            continue

        # Check if already exists
        existing = (
            db.query(AlertRule)
            .filter(
                AlertRule.monitoring_server_id == bulk.monitoring_server_id,
                AlertRule.template_id == template_id,
                AlertRule.node_group_id == bulk.node_group_id,
            )
            .first()
        )
        if existing:
            logger.info(f"Rule for template {template_id} already exists, skipping")
            continue

        # Create rule from template
        alert_rule = AlertRule(
            name=template.display_name,
            description=template.description,
            monitoring_server_id=bulk.monitoring_server_id,
            template_id=template_id,
            contact_point_id=bulk.contact_point_id,
            node_group_id=bulk.node_group_id,
            enabled=True,
            severity=template.default_severity,
            datasource_type=template.datasource_type,
            query_expression=template.query_expression,
            threshold_config=template.default_threshold,
            summary=template.summary_template,
            runbook_url=template.runbook_url,
        )
        db.add(alert_rule)
        created_rules.append(alert_rule)

    db.commit()

    # Sync all created rules to Grafana
    if sync_to_grafana:
        for rule in created_rules:
            db.refresh(rule)
            if rule.enabled:
                await _sync_rule_to_grafana(rule, db)

    return [build_alert_rule_response(r) for r in created_rules]


@router.get("/rules/{rule_id}", response_model=AlertRuleResponse)
def get_alert_rule(rule_id: int, db: Session = Depends(get_db)):
    """Get an alert rule by ID."""
    rule = get_alert_rule_or_404(db, rule_id)
    return build_alert_rule_response(rule)


@router.put("/rules/{rule_id}", response_model=AlertRuleResponse)
async def update_alert_rule(
    rule_id: int,
    update: AlertRuleUpdate,
    sync_to_grafana: bool = True,
    db: Session = Depends(get_db),
):
    """Update an alert rule."""
    rule = get_alert_rule_or_404(db, rule_id)

    # Apply updates
    if update.name is not None:
        rule.name = update.name
    if update.description is not None:
        rule.description = update.description
    if update.enabled is not None:
        rule.enabled = update.enabled
    if update.severity is not None:
        rule.severity = update.severity.value
    if update.contact_point_id is not None:
        if update.contact_point_id > 0:
            get_contact_point_or_404(db, update.contact_point_id)
            rule.contact_point_id = update.contact_point_id
        else:
            rule.contact_point_id = None
    if update.node_group_id is not None:
        if update.node_group_id > 0:
            ng = db.query(NodeGroup).filter(NodeGroup.id == update.node_group_id).first()
            if not ng:
                raise HTTPException(status_code=404, detail="Node group not found")
            rule.node_group_id = update.node_group_id
        else:
            rule.node_group_id = None
    if update.threshold_config is not None:
        rule.threshold_config = update.threshold_config.model_dump()
    if update.labels is not None:
        rule.labels = update.labels
    if update.summary is not None:
        rule.summary = update.summary
    if update.runbook_url is not None:
        rule.runbook_url = update.runbook_url

    db.commit()

    # Sync to Grafana
    if sync_to_grafana:
        await _sync_rule_to_grafana(rule, db)

    return build_alert_rule_response(rule)


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_alert_rule(rule_id: int, db: Session = Depends(get_db)):
    """Delete an alert rule."""
    rule = get_alert_rule_or_404(db, rule_id)

    # Delete from Grafana if synced
    if rule.grafana_uid:
        try:
            grafana = GrafanaProvisioner()
            await grafana.delete_alert_rule(rule.grafana_uid)
        except Exception as e:
            logger.warning(f"Failed to delete alert rule from Grafana: {e}")

    db.delete(rule)
    db.commit()


@router.post("/rules/{rule_id}/enable", response_model=AlertRuleResponse)
async def enable_alert_rule(rule_id: int, db: Session = Depends(get_db)):
    """Enable an alert rule."""
    rule = get_alert_rule_or_404(db, rule_id)
    rule.enabled = True
    db.commit()

    await _sync_rule_to_grafana(rule, db)
    return build_alert_rule_response(rule)


@router.post("/rules/{rule_id}/disable", response_model=AlertRuleResponse)
async def disable_alert_rule(rule_id: int, db: Session = Depends(get_db)):
    """Disable an alert rule."""
    rule = get_alert_rule_or_404(db, rule_id)
    rule.enabled = False

    # Delete from Grafana if synced
    if rule.grafana_uid:
        try:
            grafana = GrafanaProvisioner()
            await grafana.delete_alert_rule(rule.grafana_uid)
            rule.grafana_uid = None
            rule.last_synced_at = None
        except Exception as e:
            logger.warning(f"Failed to delete alert rule from Grafana: {e}")

    db.commit()
    return build_alert_rule_response(rule)


@router.post("/rules/{rule_id}/sync", response_model=AlertRuleResponse)
async def sync_alert_rule(rule_id: int, db: Session = Depends(get_db)):
    """Force sync an alert rule to Grafana."""
    rule = get_alert_rule_or_404(db, rule_id)
    await _sync_rule_to_grafana(rule, db)
    return build_alert_rule_response(rule)


# ============================================
# Bulk Operations
# ============================================


@router.post("/monitoring-servers/{server_id}/sync-all", response_model=AlertSyncResponse)
async def sync_all_alerts(server_id: int, db: Session = Depends(get_db)):
    """Sync all alert rules for a monitoring server to Grafana."""
    # server = get_monitoring_server_or_404(db, server_id)

    rules = db.query(AlertRule).filter(AlertRule.monitoring_server_id == server_id).all()

    synced = 0
    errors = []

    for rule in rules:
        if rule.enabled:
            try:
                await _sync_rule_to_grafana(rule, db)
                if not rule.sync_error:
                    synced += 1
                else:
                    errors.append(f"{rule.name}: {rule.sync_error}")
            except Exception as e:
                errors.append(f"{rule.name}: {str(e)}")
        else:
            # Delete disabled rules from Grafana
            if rule.grafana_uid:
                try:
                    grafana = GrafanaProvisioner()
                    await grafana.delete_alert_rule(rule.grafana_uid)
                    rule.grafana_uid = None
                    db.commit()
                except Exception as e:
                    errors.append(f"{rule.name}: Failed to remove disabled rule: {e}")

    return AlertSyncResponse(
        success=len(errors) == 0,
        message=f"Synced {synced} rules" if not errors else f"Synced {synced} rules with {len(errors)} errors",
        synced_count=synced,
        error_count=len(errors),
        errors=errors,
    )


@router.get("/monitoring-servers/{server_id}/summary", response_model=AlertConfigurationSummary)
def get_alert_summary(server_id: int, db: Session = Depends(get_db)):
    """Get alert configuration summary for a monitoring server."""
    server = get_monitoring_server_or_404(db, server_id)

    rules = db.query(AlertRule).filter(AlertRule.monitoring_server_id == server_id).all()

    enabled = sum(1 for r in rules if r.enabled)
    disabled = sum(1 for r in rules if not r.enabled)
    with_errors = sum(1 for r in rules if r.sync_error)

    contact_points = set()
    last_sync = None

    for rule in rules:
        if rule.contact_point:
            contact_points.add(rule.contact_point.name)
        if rule.last_synced_at:
            if last_sync is None or rule.last_synced_at > last_sync:
                last_sync = rule.last_synced_at

    return AlertConfigurationSummary(
        monitoring_server_id=server_id,
        monitoring_server_name=server.name,
        total_rules=len(rules),
        enabled_rules=enabled,
        disabled_rules=disabled,
        rules_with_sync_errors=with_errors,
        contact_points=list(contact_points),
        last_sync_at=last_sync,
    )


@router.post("/monitoring-servers/{server_id}/setup-defaults", response_model=List[AlertRuleResponse])
async def setup_default_alerts(
    server_id: int,
    contact_point_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Set up default alert rules for a monitoring server.

    Creates alert rules for all templates marked as is_default=True.
    """
    # server = get_monitoring_server_or_404(db, server_id)

    if contact_point_id:
        get_contact_point_or_404(db, contact_point_id)

    # Get default templates
    default_templates = db.query(AlertRuleTemplate).filter(AlertRuleTemplate.is_default).all()

    if not default_templates:
        raise HTTPException(status_code=404, detail="No default alert templates found")

    bulk = BulkAlertRuleCreate(
        monitoring_server_id=server_id,
        contact_point_id=contact_point_id,
        template_ids=[t.id for t in default_templates],
    )

    return await create_alert_rules_bulk(bulk, sync_to_grafana=True, db=db)


# ============================================
# Internal Helper Functions
# ============================================


async def _sync_rule_to_grafana(rule: AlertRule, db: Session):
    """Sync an alert rule to Grafana."""
    grafana = GrafanaProvisioner()

    try:
        # Ensure alert folder exists
        folder_uid = await grafana.get_or_create_alert_folder()

        # Build annotations
        annotations = {}
        if rule.summary:
            annotations["summary"] = rule.summary
        if rule.template and rule.template.description_template:
            annotations["description"] = rule.template.description_template
        if rule.runbook_url:
            annotations["runbook_url"] = rule.runbook_url

        # Get threshold config
        threshold = rule.threshold_config or {}
        threshold_value = threshold.get("value")
        threshold_operator = threshold.get("operator", ">")
        for_duration = threshold.get("for_duration", "5m")

        # Get datasource UID (prometheus or loki)
        datasource_uid = "prometheus" if rule.datasource_type == "prometheus" else "loki"

        # Build labels
        labels = dict(rule.labels or {})
        labels["severity"] = rule.severity
        if rule.node_group_id and rule.node_group:
            labels["node_group"] = rule.node_group.name

        if rule.grafana_uid:
            # Update existing rule
            # Note: Grafana update requires full rule data, not just changed fields
            result = await grafana.create_alert_rule(
                title=rule.name,
                folder_uid=folder_uid,
                rule_group=f"fleet-alerts-{rule.monitoring_server_id}",
                datasource_uid=datasource_uid,
                query_expression=rule.query_expression,
                threshold_value=threshold_value,
                threshold_operator=threshold_operator,
                for_duration=for_duration,
                labels=labels,
                annotations=annotations,
                datasource_type=rule.datasource_type,
            )
        else:
            # Create new rule
            result = await grafana.create_alert_rule(
                title=rule.name,
                folder_uid=folder_uid,
                rule_group=f"fleet-alerts-{rule.monitoring_server_id}",
                datasource_uid=datasource_uid,
                query_expression=rule.query_expression,
                threshold_value=threshold_value,
                threshold_operator=threshold_operator,
                for_duration=for_duration,
                labels=labels,
                annotations=annotations,
                datasource_type=rule.datasource_type,
            )

        if "error" not in result:
            rule.grafana_uid = result.get("uid")
            rule.grafana_folder_uid = folder_uid
            rule.last_synced_at = datetime.utcnow()
            rule.sync_error = None
        else:
            rule.sync_error = str(result.get("error", "Unknown error"))

        db.commit()

    except Exception as e:
        logger.error(f"Failed to sync alert rule {rule.name} to Grafana: {e}")
        rule.sync_error = str(e)
        db.commit()
