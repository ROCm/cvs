package store

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// NewID returns a sortable, collision-resistant identifier: a big-endian
// timestamp prefix (so IDs sort chronologically) plus random bytes.
func NewID() string {
	var b [10]byte
	now := uint64(time.Now().UnixNano())
	for i := 7; i >= 0; i-- {
		b[i] = byte(now)
		now >>= 8
	}
	_, _ = rand.Read(b[8:])
	return hex.EncodeToString(b[:])
}
