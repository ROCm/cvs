package sshkeys

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var filenameRE = regexp.MustCompile(`^[a-zA-Z0-9_\-.]{1,64}$`)

func pemBegin(kind string) []byte {
	return []byte("-----BEGIN " + kind + " PRIVATE KEY-----")
}

var privateKeyHeaders = [][]byte{
	pemBegin("OPENSSH"),
	pemBegin("RSA"),
	pemBegin("EC"),
	pemBegin("DSA"),
}

var allowedNonKeys = map[string]bool{"known_hosts": true, "config": true}

// sshDirOverride redirects key storage in tests so they do not touch a real
// home directory.
var sshDirOverride string

func Dir() string {
	if sshDirOverride != "" {
		return sshDirOverride
	}
	if st, err := os.Stat("/root/.ssh"); err == nil && st.IsDir() {
		return "/root/.ssh"
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "/root/.ssh"
	}
	return filepath.Join(home, ".ssh")
}

func EnsureDir() error {
	dir := Dir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	return os.Chmod(dir, 0o700)
}

func ValidateFilename(name string) error {
	if !filenameRE.MatchString(name) {
		return fmt.Errorf("invalid filename. Use only letters, digits, underscores, hyphens, and dots (max 64 chars)")
	}
	if strings.Contains(name, "..") || strings.Contains(name, "/") {
		return fmt.Errorf("invalid filename")
	}
	return nil
}

func ValidatePrivateKey(content []byte, filename string) error {
	if allowedNonKeys[filename] {
		return nil
	}
	stripped := bytesTrimLeft(content)
	for _, h := range privateKeyHeaders {
		if hasPrefix(stripped, h) {
			return nil
		}
	}
	return fmt.Errorf("%q does not appear to be an SSH private key", filename)
}

func List() ([]map[string]any, error) {
	dir := Dir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return []map[string]any{}, nil
		}
		return nil, err
	}
	var keys []map[string]any
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		keys = append(keys, map[string]any{
			"filename":    e.Name(),
			"size":        info.Size(),
			"permissions": fmt.Sprintf("%o", info.Mode().Perm()),
		})
	}
	if keys == nil {
		keys = []map[string]any{}
	}
	return keys, nil
}

func Save(filename string, content []byte) error {
	if err := ValidateFilename(filename); err != nil {
		return err
	}
	if err := ValidatePrivateKey(content, filename); err != nil {
		return err
	}
	dir := Dir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	path := filepath.Join(dir, filename)
	if info, err := os.Lstat(path); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("refusing to write through symlink %s", filename)
	}
	mode := os.FileMode(0o600)
	if allowedNonKeys[filename] {
		mode = 0o644
	}
	// WriteFile applies mode only when the file is created. Chmod enforces it
	// when an existing key was restored or copied with a looser mode.
	if err := os.WriteFile(path, content, mode); err != nil {
		return err
	}
	return os.Chmod(path, mode)
}

func Delete(filename string) error {
	if err := ValidateFilename(filename); err != nil {
		return err
	}
	path := filepath.Join(Dir(), filename)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return os.ErrNotExist
	}
	return os.Remove(path)
}

func ReadAll(r io.Reader, limit int64) ([]byte, error) {
	return io.ReadAll(io.LimitReader(r, limit))
}

func bytesTrimLeft(b []byte) []byte {
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\n' || b[i] == '\r' || b[i] == '\t') {
		i++
	}
	return b[i:]
}

func hasPrefix(b, prefix []byte) bool {
	if len(b) < len(prefix) {
		return false
	}
	for i := range prefix {
		if b[i] != prefix[i] {
			return false
		}
	}
	return true
}
