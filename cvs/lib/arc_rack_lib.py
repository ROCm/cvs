'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Build ARC rack files from the cluster file's racks block.

A rack entry lists its devices in tray order; the position in each list
becomes the ARC tray number. CVS fills the template ARC itself generates for
the rack's platform, so slot counts and the tray-to-slot map stay ARC's.
'''

import copy
import re

CREDENTIAL_KEYS = ("switch_ssh_user", "switch_ssh_password", "switch_ssh_key_file", "bmc_user", "bmc_password")
PLATFORM_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SERIAL_MAX_LEN = 64
PLACEHOLDER = "<changeme>"
NO_BMC_VALUES = ("", "NA", "N/A")
# ARC component key -> (cluster rack field with host addresses, rack field with BMC addresses)
DEVICE_FIELDS = {
    "switch": ("switch_trays", "switch_bmc"),
    "front_end_switch": ("front_end_switches", None),
    "back_end_switch": ("back_end_switches", None),
}
BMC_ONLY_FIELDS = {
    "manager": "rmc",
    "power": "power_shelves",
}


def redacted(value):
    '''Return a copy of a cluster dict with every *password* value masked, for logging.'''
    if isinstance(value, dict):
        return {
            key: "***" if "password" in str(key).lower() and item else redacted(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [redacted(item) for item in value]
    return value


def racks_block(cluster_dict):
    '''Return the cluster file's racks block, honoring the deprecated rack_groups alias.'''
    block = cluster_dict.get("racks")
    if block is None:
        block = cluster_dict.get("rack_groups")
    return block if isinstance(block, dict) else {}


def rack_entries(cluster_dict):
    '''Return {rack_id: rack_dict} in file order, without credentials or comments.'''
    return {
        key: value
        for key, value in racks_block(cluster_dict).items()
        if key not in CREDENTIAL_KEYS and not key.startswith("_") and isinstance(value, dict)
    }


def rack_nodes(cluster_dict, rack_id):
    '''Return node_dict keys tagged with rack_id, in file order (tray order).'''
    return [
        host for host, info in (cluster_dict.get("node_dict") or {}).items() if (info or {}).get("rack_id") == rack_id
    ]


def rack_controller(cluster_dict, rack_id):
    '''Return the rack's arc_controller, else its first node, else None.'''
    rack = rack_entries(cluster_dict).get(rack_id) or {}
    explicit = rack.get("arc_controller")
    if explicit:
        return explicit
    nodes = rack_nodes(cluster_dict, rack_id)
    return nodes[0] if nodes else None


def rack_controllers(cluster_dict):
    '''Return {controller_host: rack_id} for every rack that has a controller.'''
    result = {}
    for rack_id in rack_entries(cluster_dict):
        host = rack_controller(cluster_dict, rack_id)
        if host and host not in result:
            result[host] = rack_id
    return result


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _placeholder_paths(value, prefix=""):
    '''Yield dotted paths of <changeme> values, skipping _comment keys.'''
    if isinstance(value, dict):
        for key, item in value.items():
            if not str(key).startswith("_"):
                yield from _placeholder_paths(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _placeholder_paths(item, f"{prefix}[{index}]")
    elif isinstance(value, str) and PLACEHOLDER in value.lower():
        yield prefix


def validate_rack_controllers(cluster_dict):
    '''Return problems with choosing one ARC controller per rack.'''
    problems = []
    racks = rack_entries(cluster_dict)
    nodes = cluster_dict.get("node_dict") or {}
    seen = {}
    for rack_id in racks:
        if not rack_nodes(cluster_dict, rack_id):
            problems.append(f"racks.{rack_id} has no node_dict entries with rack_id {rack_id!r}")
        controller = rack_controller(cluster_dict, rack_id)
        if not controller:
            continue
        if controller not in nodes:
            problems.append(f"racks.{rack_id}.arc_controller {controller!r} is not in node_dict")
        if controller in seen:
            problems.append(f"racks {seen[controller]!r} and {rack_id!r} share controller {controller!r}")
        else:
            seen[controller] = rack_id
    for host, info in nodes.items():
        rack_id = (info or {}).get("rack_id")
        if rack_id and rack_id not in racks:
            problems.append(f"node_dict {host!r} has rack_id {rack_id!r}, which is not in the racks block")
    return problems


def validate_racks(cluster_dict):
    '''Return problems that would stop ARC rack files from being generated.'''
    racks = rack_entries(cluster_dict)
    if not racks:
        return ["cluster file has no racks block; set arc.rack_config or describe racks in the cluster file"]
    problems = validate_rack_controllers(cluster_dict)
    problems.extend(
        f"racks.{path} is still <changeme>; set it or remove it"
        for path in _placeholder_paths(racks_block(cluster_dict))
    )
    for rack_id, rack in racks.items():
        platform = rack.get("platform")
        if not platform:
            problems.append(f"racks.{rack_id}.platform must be set to generate its ARC rack file")
        elif not PLATFORM_RE.fullmatch(str(platform)):
            problems.append(f"racks.{rack_id}.platform {platform!r} may contain only letters, digits, '_' or '-'")
        trays, bmcs = _as_list(rack.get("switch_trays")), _as_list(rack.get("switch_bmc"))
        if bmcs and len(bmcs) != len(trays):
            problems.append(f"racks.{rack_id}.switch_bmc has {len(bmcs)} entries; switch_trays has {len(trays)}")
    return problems


def _serial(*parts):
    raw = re.sub(r"[^A-Za-z0-9_-]", "-", "-".join(str(part) for part in parts))
    if not raw[:1].isalnum():
        raw = f"r{raw}"
    return raw[:SERIAL_MAX_LEN]


def _set_interface(slot, name, hostname, username, password=None, key_file=None):
    interfaces = slot.get("interfaces")
    if not isinstance(interfaces, dict) or name not in interfaces:
        return False
    entry = {"hostname": hostname, "username": username or ""}
    if key_file:
        entry["ssh_key_filename"] = key_file
    else:
        entry["password"] = password or ""
    interfaces[name] = entry
    return True


def _bmc_login(cluster_dict):
    block = racks_block(cluster_dict)
    user = block.get("bmc_user")
    return (user, block.get("bmc_password")) if user else (None, None)


def _switch_login(cluster_dict):
    block = racks_block(cluster_dict)
    return block.get("switch_ssh_user"), block.get("switch_ssh_password"), block.get("switch_ssh_key_file")


class _SlotFiller:
    '''Fill one component type's template slots in order.'''

    def __init__(self, rack_id, template_rack, row_zero):
        self.rack_id = rack_id
        self.template_rack = template_rack
        self.row_zero = row_zero

    def slots(self, component, count):
        slots = self.template_rack.get(component) or []
        if count > len(slots):
            noun = "none" if not slots else f"only {len(slots)}"
            raise ValueError(
                f"rack {self.rack_id}: {count} {component} entries listed, but the platform declares {noun}"
            )
        return slots[:count]

    def identify(self, slot, component, index):
        slot["serial_number"] = _serial(self.rack_id, component, index)
        location = slot.get("location")
        # ARC rejects two components at one location, and the platform gives
        # these no slot of their own, so spread them along the unused row 0.
        if slot.get("tray_id") is None and isinstance(location, dict):
            slot["location"] = {"row": 0, "position": next(self.row_zero)}


def _row_zero_positions():
    position = 0
    while True:
        yield position
        position += 1


def fill_rack_template(template, cluster_dict, rack_id):
    '''
    Return a filled copy of ARC's generated template for rack_id.

    Raises:
        ValueError: when the rack lists more devices than the platform declares.
    '''
    rack = rack_entries(cluster_dict).get(rack_id) or {}
    data = copy.deepcopy(template)
    data["serial_number"] = _serial(rack_id)
    data.setdefault("rack", {})
    if rack.get("switch_model"):
        data.setdefault("models", {})["switch"] = rack["switch_model"]
    filler = _SlotFiller(rack_id, data["rack"], _row_zero_positions())
    bmc_user, bmc_password = _bmc_login(cluster_dict)
    switch_user, switch_password, switch_key = _switch_login(cluster_dict)

    nodes = cluster_dict.get("node_dict") or {}
    hosts = rack_nodes(cluster_dict, rack_id)
    for index, (slot, host) in enumerate(zip(filler.slots("compute", len(hosts)), hosts)):
        info = nodes.get(host) or {}
        filler.identify(slot, "compute", index)
        _set_interface(
            slot,
            "eth1",
            info.get("vpc_ip") or host,
            cluster_dict.get("username"),
            password=cluster_dict.get("password"),
            key_file=cluster_dict.get("priv_key_file"),
        )
        bmc_ip = str(info.get("bmc_ip") or "").strip()
        if bmc_user and bmc_ip.upper() not in NO_BMC_VALUES:
            _set_interface(slot, "bmc", bmc_ip, bmc_user, password=bmc_password)

    for component, (host_field, bmc_field) in DEVICE_FIELDS.items():
        addresses = _as_list(rack.get(host_field))
        bmcs = _as_list(rack.get(bmc_field)) if bmc_field else []
        for index, (slot, address) in enumerate(zip(filler.slots(component, len(addresses)), addresses)):
            filler.identify(slot, component, index)
            _set_interface(slot, "eth1", address, switch_user, password=switch_password, key_file=switch_key)
            if bmc_user and index < len(bmcs):
                _set_interface(slot, "bmc", bmcs[index], bmc_user, password=bmc_password)

    for component, field in BMC_ONLY_FIELDS.items():
        addresses = _as_list(rack.get(field))
        if addresses and not bmc_user:
            raise ValueError(f"rack {rack_id}: {field} is listed but racks.bmc_user is not set")
        for index, (slot, address) in enumerate(zip(filler.slots(component, len(addresses)), addresses)):
            filler.identify(slot, component, index)
            _set_interface(slot, "bmc", address, bmc_user, password=bmc_password)

    return data
