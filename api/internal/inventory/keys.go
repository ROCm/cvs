package inventory

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ErrBadKeyName is returned for upload filenames that could escape the key dir.
var ErrBadKeyName = errors.New("invalid key name")

// KeyStore manages uploaded SSH private keys on a writable volume. The runtime
// runs as a non-root user, so the directory must be writable by that user
// (e.g. a mounted data volume), not /root/.ssh.
type KeyStore struct {
	dir string
}

// NewKeyStore returns a key store rooted at dir.
func NewKeyStore(dir string) *KeyStore { return &KeyStore{dir: dir} }

// Dir returns the directory keys are stored in.
func (k *KeyStore) Dir() string { return k.dir }

// sanitizeKeyName validates an upload filename, rejecting anything that is not
// already a bare basename (no separators, no traversal) so it cannot escape the
// key directory.
func sanitizeKeyName(name string) (string, error) {
	clean := strings.TrimSpace(name)
	if clean == "" || clean == "." || clean == ".." ||
		strings.ContainsAny(clean, "/\\") || clean != filepath.Base(clean) {
		return "", ErrBadKeyName
	}
	return clean, nil
}

// Save writes an uploaded private key with 0600 perms, returning its filename.
func (k *KeyStore) Save(name string, r io.Reader) (string, error) {
	clean, err := sanitizeKeyName(name)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(k.dir, 0o700); err != nil {
		return "", err
	}
	dst := filepath.Join(k.dir, clean)
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := io.Copy(f, r); err != nil {
		return "", err
	}
	return clean, nil
}

// Path returns the absolute path of a stored key, for the cvs CLI --key_file.
func (k *KeyStore) Path(name string) (string, error) {
	clean, err := sanitizeKeyName(name)
	if err != nil {
		return "", err
	}
	return filepath.Join(k.dir, clean), nil
}

// List returns the stored key filenames, sorted.
func (k *KeyStore) List() ([]string, error) {
	entries, err := os.ReadDir(k.dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []string{}, nil
		}
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names, nil
}
