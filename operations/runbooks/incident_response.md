# Incident Response System - Runbook

## Roles
- **Incident Commander (IC)**: Leads response, coordinates team.
- **Ops Lead**: Investigates technical root cause (logs, metrics).
- **Comms Lead**: Updates stakeholders (status page, execs).

## Severity Levels
- **SEV-1 (Critical)**: User-facing outage > 1%. Action: Page secondary immediately.
- **SEV-2 (Major)**: Degradation (latency > SLO). Action: Page primary.
- **SEV-3 (Minor)**: Internal tool failure. Action: Ticket.

## Procedure
1. **Detection**: Alert fires (PagerDuty) or User Report.
2. **Acknowledge**: IC acknowledges alert within 5m.
3. **Triaging**:
   - Check `SystemOverview` dashboard.
   - Check `AlertList`.
   - Is it Infrastructure (AWS) or Application (Code)?
4. **Mitigation**:
   - Rollback recent deployment?
   - Scale up resources?
   - Enable "Degraded Mode" (Circuit Breaker).
5. **Resolution**: Restore service health.
6. **Post-Mortem**: Document root cause, timeline, and action items (Jira).
