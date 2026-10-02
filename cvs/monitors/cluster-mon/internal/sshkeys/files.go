package sshkeys

import (
	"os"
	"path/filepath"
)

// sshDirOverride redirects key storage in tests so they do not touch a real
// home directory.
var sshDirOverride string

func Dir() string {
	if sshDirOverride != "" {
		return sshDirOverride
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".ssh"
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
