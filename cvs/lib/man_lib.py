"""Introspection and rendering of test config parameters for ``cvs man``.

Config parameters are documented on the pydantic models in ``cvs/parsers/``;
this module turns those models into flat, printable parameter references so
the sample config files can stay free of ``_comment_*`` documentation keys.
"""

import json
import typing

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

TYPE_ALIASES = {
    "str": "string",
    "int": "integer",
    "float": "float",
    "bool": "boolean",
    "NoneType": "null",
}

# Rendered when a field is a free-form mapping whose keys the user chooses.
OPEN_MAPPING = "mapping"


class ParamDoc:
    """A single documented config parameter, flattened to a dotted path.

    ``is_section`` marks a field that only exists to group other parameters
    (a nested model). Its own default is the whole sub-config, which is noise
    in a man page, so renderers show it as a heading rather than a value.
    """

    def __init__(self, path, type_name, default, required, description, examples, constraints, is_section=False):
        self.path = path
        self.type_name = type_name
        self.default = default
        self.required = required
        self.description = description
        self.examples = examples or []
        self.constraints = constraints or []
        self.is_section = is_section

    @property
    def section(self):
        """Dotted path of the enclosing section, or "" for a top-level parameter."""
        return self.path.rsplit(".", 1)[0] if "." in self.path else ""

    @property
    def name(self):
        return self.path.rsplit(".", 1)[-1]

    def to_dict(self):
        entry = {
            "path": self.path,
            "type": self.type_name,
            "required": self.required,
            "description": self.description,
        }
        if self.is_section:
            entry["section"] = True
        elif not self.required:
            entry["default"] = self.default
        if self.examples:
            entry["examples"] = self.examples
        if self.constraints:
            entry["constraints"] = self.constraints
        return entry


def _unwrap_optional(annotation):
    """Return (inner_annotation, is_optional) for Optional[X] / Union[X, None]."""
    if typing.get_origin(annotation) is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _model_of(annotation):
    """Return (model, path_suffix, is_section) for the BaseModel behind an annotation.

    A directly nested model is a section: it groups parameters and has no value
    of its own worth printing. A model reached through a container is a real
    parameter, and ``path_suffix`` records the levels crossed to get there so a
    reader can see that ``model_params.single_node.<key>.<key>.precision`` sits
    two mappings deep rather than being a direct child.
    """
    inner, _ = _unwrap_optional(annotation)
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner, "", True

    origin = typing.get_origin(inner)
    args = typing.get_args(inner)

    if origin in (dict, typing.Dict) and len(args) == 2:
        model, suffix, _ = _model_of(args[1])
        if model is not None:
            return model, f".<key>{suffix}", False
    if origin in (list, typing.List) and args:
        model, suffix, _ = _model_of(args[0])
        if model is not None:
            return model, f"[]{suffix}", False

    return None, "", False


def _type_name(annotation):
    inner, optional = _unwrap_optional(annotation)
    origin = typing.get_origin(inner)

    if origin in (list, typing.List):
        args = typing.get_args(inner)
        rendered = f"list[{_type_name(args[0])}]" if args else "list"
    elif origin in (dict, typing.Dict):
        args = typing.get_args(inner)
        rendered = f"mapping[{_type_name(args[0])} -> {_type_name(args[1])}]" if args else OPEN_MAPPING
    elif typing.get_origin(inner) is typing.Literal:
        rendered = " | ".join(repr(a) for a in typing.get_args(inner))
    else:
        raw = getattr(inner, "__name__", str(inner))
        rendered = TYPE_ALIASES.get(raw, raw)

    return f"{rendered} (optional)" if optional else rendered


def _constraints(field):
    """Render pydantic constraint metadata (Ge, Le, MinLen, ...) as readable strings."""
    labels = {
        "ge": ">=",
        "gt": ">",
        "le": "<=",
        "lt": "<",
        "min_length": "min length",
        "max_length": "max length",
    }
    rendered = []
    for meta in field.metadata:
        for attr, label in labels.items():
            value = getattr(meta, attr, None)
            if value is not None:
                rendered.append(f"{label} {value}")
    return rendered


def _default_of(field):
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        return field.default_factory()
    return None


def iter_parameters(model, prefix=""):
    """Flatten a pydantic config model into ParamDoc entries, depth first.

    Nested models are recursed into so the caller gets one entry per leaf
    parameter, keyed by the dotted path a user would edit in their config file.
    """
    params = []
    for name, field in model.model_fields.items():
        # Config keys such as "32_cu_local_read" are not valid Python
        # identifiers, so the schema aliases them. Document the on-disk key.
        key = field.alias or name
        path = f"{prefix}.{key}" if prefix else key
        nested, suffix, is_section = _model_of(field.annotation)

        params.append(
            ParamDoc(
                path=path,
                type_name=_type_name(field.annotation),
                default=None if is_section else _default_of(field),
                required=field.is_required(),
                description=field.description or "",
                examples=list(field.examples) if field.examples else [],
                constraints=_constraints(field),
                is_section=is_section,
            )
        )

        if nested is not None:
            params.extend(iter_parameters(nested, prefix=f"{path}{suffix}"))

    return params


def find_parameters(params, query):
    """Match parameters by exact name or path first, falling back to substring.

    An exact match that has documented descendants -- a section, or a field
    whose value is itself a nested model, list, or dict of models -- includes
    those descendants too, since the parameter's own default is not the whole
    answer to "what does <query> configure".
    """
    exact = [p for p in params if query in (p.name, p.path)]
    if exact:
        exact_paths = {p.path for p in exact}
        descendants = [
            p
            for p in params
            if p.path not in exact_paths
            and any(p.path.startswith(f"{ep}.") or p.path.startswith(f"{ep}[") for ep in exact_paths)
        ]
        return exact + descendants
    return [p for p in params if query.lower() in p.path.lower()]


def _format_value(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return repr(value) if isinstance(value, str) else str(value)


def _wrap(text, width, indent):
    """Wrap description prose without importing textwrap's paragraph handling."""
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(f"{indent}{current}")
            current = word
        else:
            current = candidate
    if current:
        lines.append(f"{indent}{current}")
    return lines


def render_text(params, title=None, summary=None):
    """Render a parameter reference in the plain-print house style used by every plugin."""
    lines = []
    if title:
        lines.append("")
        lines.append(title)
        lines.append("=" * 80)
    if summary:
        lines.append("")
        lines.extend(_wrap(summary, 78, ""))

    section_notes = {p.path: p.description for p in params if p.is_section}
    leaves = [p for p in params if not p.is_section]

    # Group rather than emit on change: a section's own fields and its nested
    # sections interleave in declaration order, which would repeat headings.
    grouped = {}
    for param in leaves:
        grouped.setdefault(param.section, []).append(param)

    for section, members in grouped.items():
        lines.append("")
        lines.append(f"  {section or '(top level)'}")
        lines.append("  " + "-" * 78)
        note = section_notes.get(section)
        if note:
            lines.extend(_wrap(note, 76, "  "))

        for param in members:
            marker = "required" if param.required else f"default {_format_value(param.default)}"
            lines.append("")
            lines.append(f"  • {param.name}  [{param.type_name}, {marker}]")

            if param.description:
                lines.extend(_wrap(param.description, 74, "      "))
            else:
                lines.append("      (undocumented)")

            if param.constraints:
                lines.append(f"      constraints: {', '.join(param.constraints)}")
            if param.examples:
                lines.append(f"      example: {', '.join(_format_value(e) for e in param.examples)}")

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"Total: {len(leaves)} parameter{'' if len(leaves) == 1 else 's'}")
    lines.append("")
    return "\n".join(lines)


def render_json(params, test=None, config_files=None):
    payload = {"parameters": [p.to_dict() for p in params]}
    if test:
        payload["test"] = test
    if config_files:
        payload["config_files"] = list(config_files)
    return json.dumps(payload, indent=2, default=str)
