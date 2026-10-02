package rccl

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"time"
)

const (
	defaultRASMaxBytes = 4 * 1024 * 1024
	hardRASMaxBytes    = 64 * 1024 * 1024
)

// ErrResponseTooLarge is returned when VERBOSE STATUS exceeds the configured
// cap. The partial body is discarded so a truncated dump is not parsed.
var ErrResponseTooLarge = errors.New("rcclras response exceeds limit")

func NormalizeRASMax(n int) int {
	if n <= 0 {
		return defaultRASMaxBytes
	}
	if n > hardRASMaxBytes {
		return hardRASMaxBytes
	}
	return n
}

const (
	protoTextOnly   = 2
	protoJSONFormat = 3
	protoMonitor    = 4
)

// Client speaks the rcclras newline protocol over an already-forwarded TCP conn.
type Client struct {
	conn     net.Conn
	br       *bufio.Reader
	Protocol int
}

func NewClient(c net.Conn) *Client {
	return &Client{conn: c, br: bufio.NewReader(c)}
}

func (c *Client) Handshake(timeout time.Duration) error {
	_ = c.conn.SetDeadline(time.Now().Add(timeout))
	if _, err := c.conn.Write([]byte("CLIENT PROTOCOL 2\n")); err != nil {
		return err
	}
	line, err := c.br.ReadString('\n')
	if err != nil {
		return err
	}
	s := strings.TrimSpace(strings.TrimPrefix(line, "SERVER PROTOCOL "))
	n, err := strconv.Atoi(s)
	if err != nil {
		return fmt.Errorf("unexpected handshake: %q", line)
	}
	c.Protocol = n
	return nil
}

func (c *Client) SetTimeout(secs int, wait time.Duration) error {
	_ = c.conn.SetDeadline(time.Now().Add(wait))
	if _, err := fmt.Fprintf(c.conn, "TIMEOUT %d\n", secs); err != nil {
		return err
	}
	line, err := c.br.ReadString('\n')
	if err != nil {
		return err
	}
	if strings.TrimSpace(line) != "OK" {
		return fmt.Errorf("TIMEOUT: %q", line)
	}
	return nil
}

func (c *Client) SetFormatJSON(wait time.Duration) error {
	if c.Protocol < protoJSONFormat {
		return fmt.Errorf("SET FORMAT requires protocol %d+, server is %d", protoJSONFormat, c.Protocol)
	}
	_ = c.conn.SetDeadline(time.Now().Add(wait))
	if _, err := c.conn.Write([]byte("SET FORMAT json\n")); err != nil {
		return err
	}
	line, err := c.br.ReadString('\n')
	if err != nil {
		return err
	}
	if strings.TrimSpace(line) != "OK" {
		return fmt.Errorf("SET FORMAT: %q", line)
	}
	return nil
}

func (c *Client) Monitor(wait time.Duration) (string, error) {
	if c.Protocol < protoMonitor {
		return "", fmt.Errorf("MONITOR requires protocol %d+, server is %d", protoMonitor, c.Protocol)
	}
	_ = c.conn.SetDeadline(time.Now().Add(wait))
	if _, err := c.conn.Write([]byte("MONITOR\n")); err != nil {
		return "", err
	}
	line, err := c.br.ReadString('\n')
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(line), nil
}

func (c *Client) VerboseStatus(wait time.Duration, maxBytes int) (string, error) {
	maxBytes = NormalizeRASMax(maxBytes)
	_ = c.conn.SetDeadline(time.Now().Add(wait))
	if _, err := c.conn.Write([]byte("VERBOSE STATUS\n")); err != nil {
		return "", err
	}
	var buf bytes.Buffer
	tmp := make([]byte, 32*1024)
	for {
		n, err := c.br.Read(tmp)
		if n > 0 {
			if buf.Len()+n > maxBytes {
				return "", fmt.Errorf("%w (%d bytes)", ErrResponseTooLarge, maxBytes)
			}
			buf.Write(tmp[:n])
		}
		if err == io.EOF {
			return buf.String(), nil
		}
		if err != nil {
			return "", err
		}
	}
}
