"""
AINIC PFC / QoS / DCQCN Control-Plane Validation Module

Ports ``ansible/ainicvalidation/scripts/validate_pfc.sh``,
``validate_qos.sh``, ``validate_dcqcn.sh``, and the orchestrating
``playbooks/validate_all.yml``. Rather than deploying the shell scripts to
each node and executing them out-of-band (as the original playbook did),
CVS runs the equivalent ``nicctl`` discovery/query commands directly via
``phdl.exec`` and performs the same field-by-field comparison against a
golden-value table.

The golden values below are the generic AINIC/Pollara control-plane
defaults distilled from the original validation scripts. They are shipped
as CVS's out-of-the-box defaults for AINIC deployments and are fully
overridable via ``preflight.connectivity_check.ifoe.pfc_qos_dcqcn`` -- they
are not specific to any single deployment.
"""

from cvs.lib.preflight.base import PreflightCheck

# Generic AINIC PFC defaults.
DEFAULT_PFC_EXPECTED_CARD_COUNT = 8
DEFAULT_PFC_EXPECTED_PAUSE_TYPE = "PFC"

# Generic AINIC QoS defaults (DSCP-to-priority mapping, PFC no-drop
# configuration, and per-priority DWRR/strict scheduling parameters).
DEFAULT_QOS_EXPECTED_CARD_COUNT = 8
DEFAULT_QOS_DSCP24_PRIORITY = "3"
DEFAULT_QOS_DSCP46_PRIORITY = "6"
DEFAULT_QOS_DSCP46_PURPOSE = "xccl-cts"
DEFAULT_QOS_PFC_PRIORITY_BITMAP = "0x8"
DEFAULT_QOS_PFC_NO_DROP_PRIORITIES = "3"
DEFAULT_QOS_PRIORITY0_SCHEDULING = "DWRR|1|N/A"
DEFAULT_QOS_PRIORITY3_SCHEDULING = "DWRR|99|N/A"
DEFAULT_QOS_PRIORITY6_SCHEDULING = "strict|N/A|10"

# Generic AINIC DCQCN defaults (profile 1).
DEFAULT_DCQCN_EXPECTED_DEVICE_COUNT = 8
DEFAULT_DCQCN_PROFILE_ID = 1
DEFAULT_DCQCN_STATUS = "Enabled"
DEFAULT_DCQCN_AI_RATE = "160"
DEFAULT_DCQCN_BYTE_COUNT = "431068"
DEFAULT_DCQCN_ALPHA_G = "512"
DEFAULT_DCQCN_ALPHA_INTERVAL = "1"
DEFAULT_DCQCN_HAI_RATE = "300"
DEFAULT_DCQCN_INITIAL_ALPHA = "64"
DEFAULT_DCQCN_MONITOR_PERIOD = "1"
DEFAULT_DCQCN_RATE_THRESHOLD = "1"
DEFAULT_DCQCN_RATE_INTERVAL = "1"
DEFAULT_DCQCN_TOKEN_BUCKET = "800000"
DEFAULT_DCQCN_CNP_DSCP = "46"


def _parse_result_line(output):
    """Parse a terse ``RESULT=...|KEY=VALUE|...`` line into a dict.

    Returns ``{}`` if no ``RESULT=`` line is found (malformed/empty output).
    """
    for line in (output or '').strip().split('\n'):
        line = line.strip()
        if line.startswith('RESULT='):
            fields = {}
            for token in line.split('|'):
                if '=' in token:
                    key, _, value = token.partition('=')
                    fields[key] = value
            return fields
    return {}


class PfcValidationCheck(PreflightCheck):
    """Validate PFC pause-type configuration on every AINIC card."""

    def __init__(
        self,
        phdl,
        expected_card_count=DEFAULT_PFC_EXPECTED_CARD_COUNT,
        expected_pause_type=DEFAULT_PFC_EXPECTED_PAUSE_TYPE,
        use_sudo=True,
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_card_count = expected_card_count
        self.expected_pause_type = expected_pause_type
        self.use_sudo = use_sudo

    def _build_command(self):
        sudo = "sudo " if self.use_sudo else ""
        return f"""
        CARD_IDS=$({sudo}nicctl show port 2>/dev/null | awk '/^NIC[[:space:]]*:/ {{print $3}}')
        if [ -z "$CARD_IDS" ]; then
            echo "RESULT=ERROR|CHECK=PFC|REASON=no_card_ids_found"
            exit 0
        fi
        PASS=0
        FAIL=0
        TOTAL=0
        FAILED_CARDS=""
        for CARD_ID in $CARD_IDS; do
            TOTAL=$((TOTAL+1))
            PORT_OUTPUT=$({sudo}nicctl show port -c "$CARD_ID" --brief 2>/dev/null)
            if [ $? -ne 0 ] || [ -z "$PORT_OUTPUT" ]; then
                FAIL=$((FAIL+1))
                FAILED_CARDS="${{FAILED_CARDS}} ${{CARD_ID}}:NO_OUTPUT"
                continue
            fi
            DATA_LINE=$(echo "$PORT_OUTPUT" | grep -E '\\seth[0-9/]+\\s')
            if [ -z "$DATA_LINE" ]; then
                FAIL=$((FAIL+1))
                FAILED_CARDS="${{FAILED_CARDS}} ${{CARD_ID}}:NO_DATA_LINE"
                continue
            fi
            PAUSE_TYPE=$(echo "$DATA_LINE" | awk '{{print $9}}')
            if [ "$PAUSE_TYPE" = "{self.expected_pause_type}" ]; then
                PASS=$((PASS+1))
            else
                FAIL=$((FAIL+1))
                FAILED_CARDS="${{FAILED_CARDS}} ${{CARD_ID}}:PAUSE_TYPE=${{PAUSE_TYPE:-EMPTY}}"
            fi
        done
        FAILED_CARDS="${{FAILED_CARDS# }}"
        if [ "$FAIL" -eq 0 ] && [ "$TOTAL" -eq {self.expected_card_count} ]; then
            echo "RESULT=PASS|CHECK=PFC|CARDS=${{TOTAL}}|PASSED=${{PASS}}|FAILED=${{FAIL}}"
        else
            echo "RESULT=FAIL|CHECK=PFC|CARDS=${{TOTAL}}|PASSED=${{PASS}}|FAILED=${{FAIL}}|FAILED_CARDS=${{FAILED_CARDS}}"
        fi
        """

    def run(self):
        """
        Execute the PFC pause-type validation on every reachable node.

        Returns:
            dict: Per-node results ``{node: {status, cards, passed, failed, errors}}``.
        """
        self.results = {}
        out_dict = self.phdl.exec(self._build_command())

        for node, output in out_dict.items():
            fields = _parse_result_line(output)
            if not fields or fields.get('RESULT') == 'ERROR':
                reason = fields.get('REASON', 'malformed or empty nicctl output')
                self.results[node] = {
                    'status': 'FAIL',
                    'cards': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': [f"PFC validation could not discover AINIC cards: {reason}"],
                }
                continue

            cards = int(fields.get('CARDS', 0) or 0)
            passed = int(fields.get('PASSED', 0) or 0)
            failed = int(fields.get('FAILED', 0) or 0)
            result = fields.get('RESULT')
            errors = []
            if result != 'PASS':
                if cards != self.expected_card_count:
                    errors.append(f"Expected {self.expected_card_count} AINIC card(s), found {cards}")
                if fields.get('FAILED_CARDS'):
                    errors.append(f"PFC mismatch on card(s): {fields['FAILED_CARDS']}")
                if not errors:
                    errors.append("PFC validation failed")

            self.results[node] = {
                'status': 'PASS' if result == 'PASS' else 'FAIL',
                'cards': cards,
                'passed': passed,
                'failed': failed,
                'errors': errors,
            }
        return self.results


class QosValidationCheck(PreflightCheck):
    """Validate DSCP/QoS/scheduling configuration on every AINIC card."""

    def __init__(
        self,
        phdl,
        expected_card_count=DEFAULT_QOS_EXPECTED_CARD_COUNT,
        dscp24_priority=DEFAULT_QOS_DSCP24_PRIORITY,
        dscp46_priority=DEFAULT_QOS_DSCP46_PRIORITY,
        dscp46_purpose=DEFAULT_QOS_DSCP46_PURPOSE,
        pfc_priority_bitmap=DEFAULT_QOS_PFC_PRIORITY_BITMAP,
        pfc_no_drop_priorities=DEFAULT_QOS_PFC_NO_DROP_PRIORITIES,
        priority0_scheduling=DEFAULT_QOS_PRIORITY0_SCHEDULING,
        priority3_scheduling=DEFAULT_QOS_PRIORITY3_SCHEDULING,
        priority6_scheduling=DEFAULT_QOS_PRIORITY6_SCHEDULING,
        use_sudo=True,
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_card_count = expected_card_count
        self.dscp24_priority = dscp24_priority
        self.dscp46_priority = dscp46_priority
        self.dscp46_purpose = dscp46_purpose
        self.pfc_priority_bitmap = pfc_priority_bitmap
        self.pfc_no_drop_priorities = pfc_no_drop_priorities
        self.priority0_scheduling = priority0_scheduling
        self.priority3_scheduling = priority3_scheduling
        self.priority6_scheduling = priority6_scheduling
        self.use_sudo = use_sudo

    def _build_command(self):
        sudo = "sudo " if self.use_sudo else ""
        return f"""
        CARD_IDS=$({sudo}nicctl show port 2>/dev/null | awk '/^NIC[[:space:]]*:/ {{print $3}}')
        if [ -z "$CARD_IDS" ]; then
            echo "RESULT=ERROR|CHECK=QOS|REASON=no_card_ids_found"
            exit 0
        fi
        PASS=0
        FAIL=0
        TOTAL=0
        DETAILS=""
        for CARD_ID in $CARD_IDS; do
            TOTAL=$((TOTAL+1))
            QOS_OUTPUT=$({sudo}nicctl show qos -c "$CARD_ID" 2>/dev/null)
            if [ $? -ne 0 ] || [ -z "$QOS_OUTPUT" ]; then
                FAIL=$((FAIL+1))
                DETAILS="${{DETAILS}} ${{CARD_ID}}:NO_OUTPUT"
                continue
            fi

            DSCP24=$(echo "$QOS_OUTPUT" | awk '/DSCP[[:space:]]*:[[:space:]]*24[[:space:]]*==>/ {{print $NF}}' | head -1)
            DSCP46_PRIO=$(echo "$QOS_OUTPUT" | awk '/DSCP[[:space:]]*:[[:space:]]*46[[:space:]]*==>/ {{print $NF}}' | head -1)
            DSCP46_PURPOSE=$(echo "$QOS_OUTPUT" | awk '/DSCP-to-purpose/ && /46/ {{print $NF}}' | head -1)
            PFC_BITMAP=$(echo "$QOS_OUTPUT" | awk -F: '/PFC priority bitmap/ {{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $NF); print $NF}}' | head -1)
            PFC_NODROP=$(echo "$QOS_OUTPUT" | awk -F: '/PFC no-drop priorities/ {{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $NF); print $NF}}' | head -1)
            P0=$(echo "$QOS_OUTPUT" | awk '$1=="0" {{print $2"|"$3"|"$4}}' | head -1)
            P3=$(echo "$QOS_OUTPUT" | awk '$1=="3" {{print $2"|"$3"|"$4}}' | head -1)
            P6=$(echo "$QOS_OUTPUT" | awk '$1=="6" {{print $2"|"$3"|"$4}}' | head -1)

            CARD_FAIL=0
            REASONS=""
            [ "$DSCP24" = "{self.dscp24_priority}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};dscp24_priority=${{DSCP24:-EMPTY}}"; }}
            [ "$DSCP46_PRIO" = "{self.dscp46_priority}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};dscp46_priority=${{DSCP46_PRIO:-EMPTY}}"; }}
            [ "$DSCP46_PURPOSE" = "{self.dscp46_purpose}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};dscp46_purpose=${{DSCP46_PURPOSE:-EMPTY}}"; }}
            [ "$PFC_BITMAP" = "{self.pfc_priority_bitmap}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};pfc_priority_bitmap=${{PFC_BITMAP:-EMPTY}}"; }}
            [ "$PFC_NODROP" = "{self.pfc_no_drop_priorities}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};pfc_no_drop_priorities=${{PFC_NODROP:-EMPTY}}"; }}
            [ "$P0" = "{self.priority0_scheduling}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};priority0_scheduling=${{P0:-EMPTY}}"; }}
            [ "$P3" = "{self.priority3_scheduling}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};priority3_scheduling=${{P3:-EMPTY}}"; }}
            [ "$P6" = "{self.priority6_scheduling}" ] || {{ CARD_FAIL=1; REASONS="${{REASONS}};priority6_scheduling=${{P6:-EMPTY}}"; }}

            if [ "$CARD_FAIL" -eq 0 ]; then
                PASS=$((PASS+1))
            else
                FAIL=$((FAIL+1))
                DETAILS="${{DETAILS}} ${{CARD_ID}}:[${{REASONS#;}}]"
            fi
        done
        DETAILS="${{DETAILS# }}"
        if [ "$FAIL" -eq 0 ] && [ "$TOTAL" -eq {self.expected_card_count} ]; then
            echo "RESULT=PASS|CHECK=QOS|CARDS=${{TOTAL}}|PASSED=${{PASS}}|FAILED=${{FAIL}}"
        else
            echo "RESULT=FAIL|CHECK=QOS|CARDS=${{TOTAL}}|PASSED=${{PASS}}|FAILED=${{FAIL}}|DETAILS=${{DETAILS}}"
        fi
        """

    def run(self):
        """
        Execute the QoS/DSCP/scheduling validation on every reachable node.

        Returns:
            dict: Per-node results ``{node: {status, cards, passed, failed, errors}}``.
        """
        self.results = {}
        out_dict = self.phdl.exec(self._build_command())

        for node, output in out_dict.items():
            fields = _parse_result_line(output)
            if not fields or fields.get('RESULT') == 'ERROR':
                reason = fields.get('REASON', 'malformed or empty nicctl output')
                self.results[node] = {
                    'status': 'FAIL',
                    'cards': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': [f"QoS validation could not discover AINIC cards: {reason}"],
                }
                continue

            cards = int(fields.get('CARDS', 0) or 0)
            passed = int(fields.get('PASSED', 0) or 0)
            failed = int(fields.get('FAILED', 0) or 0)
            result = fields.get('RESULT')
            errors = []
            if result != 'PASS':
                if cards != self.expected_card_count:
                    errors.append(f"Expected {self.expected_card_count} AINIC card(s), found {cards}")
                if fields.get('DETAILS'):
                    errors.append(f"QoS mismatch on card(s): {fields['DETAILS']}")
                if not errors:
                    errors.append("QoS validation failed")

            self.results[node] = {
                'status': 'PASS' if result == 'PASS' else 'FAIL',
                'cards': cards,
                'passed': passed,
                'failed': failed,
                'errors': errors,
            }
        return self.results


class DcqcnValidationCheck(PreflightCheck):
    """Validate DCQCN congestion-control profile parameters on every AINIC device."""

    def __init__(
        self,
        phdl,
        expected_device_count=DEFAULT_DCQCN_EXPECTED_DEVICE_COUNT,
        profile_id=DEFAULT_DCQCN_PROFILE_ID,
        status=DEFAULT_DCQCN_STATUS,
        ai_rate=DEFAULT_DCQCN_AI_RATE,
        byte_count=DEFAULT_DCQCN_BYTE_COUNT,
        alpha_g=DEFAULT_DCQCN_ALPHA_G,
        alpha_interval=DEFAULT_DCQCN_ALPHA_INTERVAL,
        hai_rate=DEFAULT_DCQCN_HAI_RATE,
        initial_alpha=DEFAULT_DCQCN_INITIAL_ALPHA,
        monitor_period=DEFAULT_DCQCN_MONITOR_PERIOD,
        rate_threshold=DEFAULT_DCQCN_RATE_THRESHOLD,
        rate_interval=DEFAULT_DCQCN_RATE_INTERVAL,
        token_bucket=DEFAULT_DCQCN_TOKEN_BUCKET,
        cnp_dscp=DEFAULT_DCQCN_CNP_DSCP,
        use_sudo=True,
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_device_count = expected_device_count
        self.profile_id = profile_id
        self.golden = {
            'Status': status,
            'Rate increase in AI phase': ai_rate,
            'Rate increase byte count': byte_count,
            'Alpha update G value': alpha_g,
            'Alpha update interval': alpha_interval,
            'Rate increase in HAI phase': hai_rate,
            'Initial alpha value': initial_alpha,
            'Rate reduce monitor period': monitor_period,
            'Rate increase threshold': rate_threshold,
            'Rate increase interval': rate_interval,
            'Token bucket size': token_bucket,
            'DSCP value used for CNP': cnp_dscp,
        }
        self.use_sudo = use_sudo

    def _build_command(self):
        sudo = "sudo " if self.use_sudo else ""
        labels = list(self.golden.keys())
        # Emit one "LABEL=value" line per golden field for each device, using a stable
        # slug (position index) as the key so we don't have to worry about shell-quoting
        # labels that contain spaces.
        extract_lines = "\n".join(
            f'            V{i}=$(echo "$DCQCN_OUTPUT" | grep -i "{label}" | awk -F: \'{{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $NF); print $NF}}\' | head -1)'
            for i, label in enumerate(labels)
        )
        emit_lines = "|".join(f'F{i}=${{V{i}}}' for i in range(len(labels)))
        return f"""
        DEVICES=$(ibv_devices 2>/dev/null | grep "ionic_" | awk '{{print $1}}')
        if [ -z "$DEVICES" ]; then
            echo "RESULT=ERROR|CHECK=DCQCN|REASON=no_devices_found"
            exit 0
        fi
        PASS=0
        FAIL=0
        TOTAL=0
        DETAILS=""
        for ROCE_DEV in $DEVICES; do
            TOTAL=$((TOTAL+1))
            DCQCN_OUTPUT=$({sudo}nicctl show dcqcn -r "$ROCE_DEV" -i {self.profile_id} 2>/dev/null)
            if [ $? -ne 0 ] || [ -z "$DCQCN_OUTPUT" ]; then
                FAIL=$((FAIL+1))
                DETAILS="${{DETAILS}} ${{ROCE_DEV}}:NO_OUTPUT"
                continue
            fi
{extract_lines}
            echo "DEV_FIELDS:${{ROCE_DEV}}:{emit_lines}"
        done
        echo "RESULT=RAW|CHECK=DCQCN|DEVICES=${{TOTAL}}"
        """

    def run(self):
        """
        Execute the DCQCN profile validation on every reachable node.

        The remote command emits one ``DEV_FIELDS:<dev>:F0=...|F1=...`` line per
        discovered device (positionally aligned with the golden-value labels)
        plus a trailing ``RESULT=`` summary line; parsing and golden-value
        comparison happens here in Python.

        Returns:
            dict: Per-node results ``{node: {status, devices, passed, failed, errors}}``.
        """
        self.results = {}
        out_dict = self.phdl.exec(self._build_command())
        labels = list(self.golden.keys())

        for node, output in out_dict.items():
            lines = (output or '').strip().split('\n')
            summary = _parse_result_line(output)

            if not summary:
                self.results[node] = {
                    'status': 'FAIL',
                    'devices': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': ["DCQCN validation produced malformed or empty output"],
                }
                continue

            if summary.get('RESULT') == 'ERROR':
                reason = summary.get('REASON', 'no AINIC devices found')
                self.results[node] = {
                    'status': 'FAIL',
                    'devices': 0,
                    'passed': 0,
                    'failed': 0,
                    'errors': [f"DCQCN validation could not discover AINIC devices: {reason}"],
                }
                continue

            device_count = 0
            passed = 0
            failed = 0
            details = []
            for line in lines:
                if not line.startswith('DEV_FIELDS:'):
                    continue
                device_count += 1
                _, dev, field_blob = line.split(':', 2)
                values = {}
                for token in field_blob.split('|'):
                    if '=' in token:
                        key, _, value = token.partition('=')
                        values[key] = value

                mismatches = []
                for i, label in enumerate(labels):
                    actual = values.get(f'F{i}', '')
                    expected = self.golden[label]
                    if actual != expected:
                        mismatches.append(f"{label}={actual or 'EMPTY'} (expected {expected})")

                if mismatches:
                    failed += 1
                    details.append(f"{dev}:[{'; '.join(mismatches)}]")
                else:
                    passed += 1

            errors = []
            if device_count != self.expected_device_count:
                errors.append(f"Expected {self.expected_device_count} AINIC device(s), found {device_count}")
            if details:
                errors.append(f"DCQCN mismatch on device(s): {'; '.join(details)}")

            status = 'PASS' if not errors else 'FAIL'
            self.results[node] = {
                'status': status,
                'devices': device_count,
                'passed': passed,
                'failed': failed,
                'errors': errors,
            }
        return self.results
