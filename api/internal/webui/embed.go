// Package webui serves the embedded single-page React application.
//
// The production UI bundle is built by Vite (see web/) and copied into the
// dist/ directory at container build time. A placeholder dist/index.html is
// committed so that local `go build` works without first building the UI.
package webui

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"
)

//go:embed all:dist
var embedded embed.FS

// FS returns the embedded dist/ directory as a filesystem rooted at the bundle.
func FS() fs.FS {
	sub, err := fs.Sub(embedded, "dist")
	if err != nil {
		// embed guarantees dist exists at build time; this is unreachable.
		panic(err)
	}
	return sub
}

// Handler serves the SPA: static assets are served directly, and any unknown
// non-asset path falls back to index.html so client-side routing works.
//
// Cache policy:
//   - assets/* (Vite content-hashed bundles) → immutable, 1-year max-age
//   - everything else (index.html, favicon, …) → no-cache so browsers
//     always revalidate and pick up new bundle references after a deploy
func Handler() http.Handler {
	content := FS()
	fileServer := http.FileServer(http.FS(content))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upath := strings.TrimPrefix(r.URL.Path, "/")
		if upath == "" {
			upath = "index.html"
		}

		if strings.HasPrefix(upath, "assets/") {
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		} else {
			w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		}

		if f, err := content.Open(upath); err == nil {
			_ = f.Close()
			fileServer.ServeHTTP(w, r)
			return
		}

		// SPA fallback: serve index.html for client-side routes (/cvs, /cluster, /fleet, ...).
		r2 := r.Clone(r.Context())
		r2.URL.Path = "/"
		fileServer.ServeHTTP(w, r2)
	})
}
