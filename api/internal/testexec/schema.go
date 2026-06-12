package testexec

import (
	"sort"
	"strings"
)

// Field describes one node of a config form, inferred from an example config.
// It is a lightweight, self-describing descriptor the UI renders directly
// (not a full JSON Schema) — types come from the example values, and inline
// `_comment_*` keys become field descriptions.
type Field struct {
	Name        string  `json:"name"`
	Type        string  `json:"type"` // object | array | string | number | boolean | null
	Description string  `json:"description,omitempty"`
	Fields      []Field `json:"fields,omitempty"` // for type=object
	Item        *Field  `json:"item,omitempty"`   // element descriptor for type=array
}

// InferSchema builds a field tree from a parsed config value. The top-level
// value is typically an object; its fields are returned sorted by name.
func InferSchema(value any) []Field {
	root := inferField("", value)
	if root.Type == "object" {
		return root.Fields
	}
	// Non-object root: wrap as a single anonymous field.
	return []Field{root}
}

func inferField(name string, value any) Field {
	switch v := value.(type) {
	case map[string]any:
		return Field{Name: name, Type: "object", Fields: inferObjectFields(v)}
	case []any:
		f := Field{Name: name, Type: "array"}
		if len(v) > 0 {
			item := inferField("", v[0])
			f.Item = &item
		}
		return f
	case bool:
		return Field{Name: name, Type: "boolean"}
	case float64, int, int64:
		return Field{Name: name, Type: "number"}
	case string:
		return Field{Name: name, Type: "string"}
	case nil:
		return Field{Name: name, Type: "null"}
	default:
		return Field{Name: name, Type: "string"}
	}
}

// inferObjectFields turns a map into sorted fields, folding `_comment_<field>`
// keys into the description of `<field>` and dropping all `_comment*` keys.
func inferObjectFields(obj map[string]any) []Field {
	comments := map[string]string{}
	for k, val := range obj {
		if strings.HasPrefix(k, "_comment_") {
			field := strings.TrimPrefix(k, "_comment_")
			if s, ok := val.(string); ok {
				comments[field] = s
			}
		}
	}

	fields := []Field{}
	for k, val := range obj {
		if strings.HasPrefix(k, "_comment") {
			continue
		}
		f := inferField(k, val)
		if desc, ok := comments[k]; ok {
			f.Description = desc
		}
		fields = append(fields, f)
	}
	sort.Slice(fields, func(i, j int) bool { return fields[i].Name < fields[j].Name })
	return fields
}
