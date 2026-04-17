# CVS REST API Server Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Design Decisions](#design-decisions)
4. [Authentication & Security](#authentication--security)
5. [API Specifications](#api-specifications)
6. [WebSocket Implementation](#websocket-implementation)
7. [Implementation Plan](#implementation-plan)
8. [Deployment Architecture](#deployment-architecture)
9. [Development Phases](#development-phases)

---

## Executive Summary

This document outlines the design and implementation plan for a Go-based REST API server for CVS (Cluster Validation Suite) that enables external clients to dynamically execute test suites through API calls and present dynamic forms to users.

### Key Design Choices
- **API Contracts**: JSON Schema (over Protocol Buffers) for dynamic discovery and web-first ecosystem
- **Authentication**: JWT tokens for service-to-service communication with multi-tier fallbacks
- **Real-time Communication**: WebSocket support for live log streaming and status updates
- **Security**: TLS/SSL encryption with comprehensive authentication layers
- **Deployment**: Docker containerization for scalable production deployment

---

## Architecture Overview

The CVS REST API follows a microservices architecture with clear separation of concerns:

![Architecture Overview](images/architecture-overview.png)
*Figure 1: Complete system architecture showing data flow from user to test execution*

> **Editable Source**: [architecture-overview.drawio](images/architecture-overview.drawio)

### System Components

1. **User Interface Layer**
   - User interacts with Client Frontend
   - Dynamic form generation based on JSON schemas

2. **Service Layer**
   - Client Backend handles user requests
   - CVS Go API Server processes test execution

3. **Execution Layer**
   - Python CVS CLI performs actual test execution
   - Results stored and returned through the API

4. **Supporting Infrastructure**
   - Schema Registry for dynamic test suite discovery
   - Docker containers for scalable deployment
   - Authentication layer with JWT and TLS/SSL

---

## Design Decisions

### 1. API Contract Format: JSON Schema vs Protocol Buffers

#### Decision: JSON Schema ✅

**Rationale:**
- **Dynamic Discovery**: Client teams can fetch schemas at runtime without CVS repository access
- **Zero Repository Coupling**: No build-time dependency between client and CVS repositories
- **Web-First Ecosystem**: Rich form generation libraries available
- **Configuration-Centric**: Natural fit for complex nested config structures

**Alternative Considered: Protocol Buffers**

**Pros of Protobuf:**
- Strong compile-time type safety
- Efficient binary serialization
- Built-in schema evolution
- Multi-language code generation

**Cons of Protobuf (Why Rejected):**
- Requires CVS repository access for client teams
- Complex form generation (limited tooling)
- Overkill for configuration validation use case
- Additional build complexity

### 2. Authentication Strategy: JWT vs API Keys vs mTLS

#### Decision: JWT Authentication ✅

**Rationale:**
- **Service-to-Service Fit**: Perfect for Client ↔ CVS communication
- **Stateless**: No session storage required
- **Flexible**: Can embed permissions and metadata
- **Industry Standard**: Excellent library support

**Alternatives Considered:**

**API Keys:**
- **Pros**: Simple implementation, fast validation
- **Cons**: No expiration, no metadata, static tokens
- **Verdict**: Too simple for production microservices

**Mutual TLS (mTLS):**
- **Pros**: Strongest security, certificate-based auth
- **Cons**: Complex PKI management, difficult debugging
- **Verdict**: Overkill for internal services

### 3. Client Access Patterns

#### Decision: Multi-Tier Authentication ✅

**Supported Access Methods:**
1. **Client Service**: JWT tokens (primary)
2. **Local Node Access**: IP-based allowlist (admin/debug)
3. **Third-Party Clients**: API keys (external integrations)
4. **Enterprise**: OAuth/OIDC support (future)

## Implementation Architecture

### Core Components

#### 1. Schema Registry
**Purpose**: Dynamic discovery and validation of test suite configurations

**Key Features:**
- Scans [`cvs/input/config_file/`](cvs/input/config_file/) directories
- Generates JSON Schema from existing configs like [`rccl_config.json`](cvs/input/config_file/rccl/rccl_config.json)
- Provides UI hints for form generation
- Supports schema versioning

#### 2. REST API Endpoints

**HTTP REST Endpoints:**
```
GET  /api/v1/suites                    # List available test suites
GET  /api/v1/suites/{suite}/schema     # Get JSON schema for suite
GET  /api/v1/suites/{suite}/examples   # Get example configurations
POST /api/v1/tests/execute             # Execute test suite
GET  /api/v1/executions/{id}           # Get execution status
GET  /api/v1/executions/{id}/logs      # Get historical logs (HTTP)
```

**WebSocket Endpoints:**
```
WS   /ws/executions/{id}/logs          # Real-time log streaming
WS   /ws/executions/{id}/status        # Real-time status updates
WS   /ws/notifications                 # Global notifications (errors, completions)
```

**WebSocket Message Format:**
```json
{
  "type": "log|status|error|completion",
  "execution_id": "exec-123",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "message": "Test completed successfully",
    "status": "passed|failed|running",
    "progress": 85
  }
}
```

#### 3. Configuration Converter
**Purpose**: Transform JSON API requests to CVS CLI format

**Process:**
1. Validate request against JSON Schema
2. Generate temporary config files
3. Execute: `cvs run {suite} --cluster_file {cluster.json} --config_file {generated.json}`
4. Return results in standardized format

### WebSocket Implementation

#### Real-Time Communication Requirements

**Use Cases:**
1. **Log Streaming**: When client user clicks "Show Logs" button
2. **Status Updates**: Immediate notification when tests pass/fail
3. **Error Notifications**: Real-time error alerts during execution

**WebSocket Flow:**
```mermaid
sequenceDiagram
    participant Client as Client Frontend
    participant CVS as CVS API Server
    participant CLI as Python CVS CLI
    
    Client->>CVS: POST /api/v1/tests/execute
    CVS->>Client: HTTP 202 {execution_id: "exec-123"}
    
    Client->>CVS: WS connect /ws/executions/exec-123/status
    CVS->>Client: WS: {"type": "status", "status": "running"}
    
    Note over CVS,CLI: Test execution begins
    CLI->>CVS: Log output (stdout/stderr)
    CVS->>Client: WS: {"type": "log", "message": "Starting RCCL test..."}
    
    CLI->>CVS: Test fails with error
    CVS->>Client: WS: {"type": "error", "message": "RCCL initialization failed"}
    CVS->>Client: WS: {"type": "status", "status": "failed"}
    
    Client->>CVS: WS disconnect
```

**Authentication for WebSocket:**
- WebSocket connections use JWT tokens via `Sec-WebSocket-Protocol` header
- Same multi-tier authentication as REST endpoints
- Connection upgrading validates JWT before establishing WebSocket

**WebSocket Connection Management:**
```go
// WebSocket connection with authentication
func (s *CVSServer) handleWebSocketUpgrade(w http.ResponseWriter, r *http.Request) {
    // Validate JWT token before upgrade
    token := r.Header.Get("Sec-WebSocket-Protocol")
    if !s.validateJWT(token) {
        http.Error(w, "Unauthorized", 401)
        return
    }
    
    // Upgrade to WebSocket
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        return
    }
    
    // Handle real-time communication
    s.handleWebSocketConnection(conn, executionID)
}
```

### Security Implementation

#### JWT Token Flow
```mermaid
sequenceDiagram
    participant Client as Client Service
    participant CVS as CVS API
    participant CLI as Python CVS CLI
    
    Client->>Client: Generate JWT token
    Client->>CVS: POST /api/v1/tests/execute<br/>Authorization: Bearer {jwt}
    CVS->>CVS: Validate JWT signature
    CVS->>CVS: Generate config files
    CVS->>CLI: Execute: cvs run rccl --cluster_file --config_file
    CLI->>CVS: Return results
    CVS->>Client: JSON response
```

#### Multi-Tier Authentication
```mermaid
flowchart TD
    Request[Incoming Request] --> CheckIP{Local IP?}
    CheckIP -->|Yes| AllowLocal[Allow Full Access]
    CheckIP -->|No| CheckJWT{JWT Token?}
    CheckJWT -->|Valid| AllowService[Allow Service Access]
    CheckJWT -->|Invalid/Missing| CheckAPI{API Key?}
    CheckAPI -->|Valid| AllowThirdParty[Allow Limited Access]
    CheckAPI -->|Invalid| Deny[Deny Access]
```

### File Structure

```
ROCm/cvs/
├── api/                           # New Go API server
│   ├── cmd/
│   │   └── server/
│   │       └── main.go           # API server entry point
│   ├── internal/
│   │   ├── auth/                 # JWT and authentication
│   │   ├── handlers/             # HTTP request handlers
│   │   ├── websocket/            # WebSocket handlers and connection management
│   │   ├── schema/               # JSON Schema registry
│   │   ├── executor/             # CVS CLI execution with real-time output
│   │   └── config/               # Configuration management
│   ├── pkg/
│   │   └── types/                # Shared types and models
│   ├── configs/
│   │   ├── api-config.yaml       # API server configuration
│   │   └── api-keys.yaml         # Third-party API keys
│   ├── scripts/
│   │   └── generate-schemas.go   # Schema generation utility
│   ├── go.mod
│   └── Dockerfile
└── cvs/                          # Existing Python CVS CLI (unchanged)
```

## Authentication & Security

### Multi-Tier Authentication Flow

![Multi-Tier Authentication](images/multi-tier-auth.png)
*Figure 2: Authentication decision flow with multiple access methods*

> **Editable Source**: [multi-tier-auth.drawio](images/multi-tier-auth.drawio)

### JWT Token Flow

![JWT Token Flow](images/jwt-token-flow.png)
*Figure 3: Sequence diagram showing JWT authentication process*

> **Editable Source**: [jwt-token-flow.drawio](images/jwt-token-flow.drawio)

### Security Features

1. **Transport Layer Security**
   - TLS/SSL for all API calls
   - Certificate-based secure connections
   - Encrypted WebSocket communication

2. **Authentication Methods**
   ```
   Priority Order:
   1. Local IP Check (127.0.0.0/8, 192.168.0.0/16) → Full Access
   2. JWT Token Validation → Service Access  
   3. API Key Validation → Limited Access
   4. Deny Access
   ```

3. **Authorization Levels**
   - **Full Access**: All endpoints, admin operations
   - **Service Access**: Test execution, schema access
   - **Limited Access**: Read-only schema and status endpoints

## API Specifications

### REST Endpoints

#### Core API Endpoints
```http
# Test Suite Discovery
GET  /api/v1/suites                    # List available test suites
GET  /api/v1/suites/{suite}/schema     # Get JSON schema for suite  
GET  /api/v1/suites/{suite}/examples   # Get example configurations

# Test Execution
POST /api/v1/tests/execute             # Execute test suite
GET  /api/v1/executions/{id}           # Get execution status
GET  /api/v1/executions/{id}/logs      # Get historical logs (HTTP)

# Health & Status
GET  /api/v1/health                    # Service health check
GET  /api/v1/version                   # API version information
```

#### WebSocket Endpoints
```websocket
# Real-time Communication
WS   /ws/executions/{id}/logs          # Real-time log streaming
WS   /ws/executions/{id}/status        # Real-time status updates
WS   /ws/notifications                 # Global notifications
```

### Request/Response Examples

#### Execute Test Suite
```http
POST /api/v1/tests/execute
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "suite": "rccl",
  "cluster_config": {
    "nodes": ["node1", "node2"],
    "gpus_per_node": 8
  },
  "test_config": {
    "test_type": "allreduce",
    "data_size": "1GB",
    "iterations": 100
  }
}
```

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "execution_id": "exec-abc123",
  "status": "queued",
  "created_at": "2026-04-16T15:30:00Z",
  "websocket_url": "/ws/executions/exec-abc123/status"
}
```

## WebSocket Implementation

### Real-Time Communication Flow

![WebSocket Flow](images/websocket-flow.png)
*Figure 4: WebSocket communication sequence for real-time updates*

> **Editable Source**: [websocket-flow.drawio](images/websocket-flow.drawio)

### WebSocket Message Format

```json
{
  "type": "log|status|error|completion",
  "execution_id": "exec-abc123", 
  "timestamp": "2026-04-16T15:30:00Z",
  "data": {
    "message": "Test completed successfully",
    "status": "passed|failed|running",
    "progress": 85,
    "metadata": {}
  }
}
```

### Use Cases

1. **Log Streaming**: Real-time display of test execution logs
2. **Status Updates**: Immediate notification of test state changes
3. **Error Notifications**: Instant error alerts during execution
4. **Progress Tracking**: Live progress indicators for long-running tests

### WebSocket Authentication

WebSocket connections authenticate using JWT tokens via the `Sec-WebSocket-Protocol` header:

```javascript
const ws = new WebSocket('wss://cvs-api/ws/executions/exec-123/status', [jwtToken]);
```

## Implementation Plan

### Project Structure

```
ROCm/cvs/
├── api/                           # New Go API server
│   ├── cmd/
│   │   └── server/
│   │       └── main.go           # API server entry point
│   ├── internal/
│   │   ├── auth/                 # JWT and authentication
│   │   ├── handlers/             # HTTP request handlers
│   │   ├── websocket/            # WebSocket handlers and connection management
│   │   ├── schema/               # JSON Schema registry
│   │   ├── executor/             # CVS CLI execution with real-time output
│   │   └── config/               # Configuration management
│   ├── pkg/
│   │   └── types/                # Shared types and models
│   ├── configs/
│   │   ├── api-config.yaml       # API server configuration
│   │   └── api-keys.yaml         # Third-party API keys
│   ├── scripts/
│   │   └── generate-schemas.go   # Schema generation utility
│   ├── go.mod
│   └── Dockerfile
└── cvs/                          # Existing Python CVS CLI (unchanged)
```

### Core Components

#### 1. Schema Registry
**Purpose**: Dynamic discovery and validation of test suite configurations

**Features**:
- Scans `cvs/input/config_file/` directories automatically
- Generates JSON Schema from existing configs (e.g., `rccl_config.json`)
- Provides UI hints for form generation
- Supports schema versioning and validation

#### 2. Configuration Converter  
**Purpose**: Transform JSON API requests to CVS CLI format

**Process**:
1. Validate request against JSON Schema
2. Generate temporary config files in CVS format
3. Execute: `cvs run {suite} --cluster_file {cluster.json} --config_file {generated.json}`
4. Return results in standardized JSON format

#### 3. WebSocket Manager
**Purpose**: Handle real-time communication with clients

**Features**:
- Connection management with authentication
- Real-time log streaming from CVS CLI
- Status broadcasting to multiple clients
- Error handling and reconnection support

## Key Benefits of Final Design

### 1. Team Independence
- **Client Team**: No CVS repository access needed
- **CVS Team**: Owns API contracts and test configurations  
- **Dynamic Discovery**: New test suites automatically available

### 2. Scalability
- **Horizontal Scaling**: Stateless JWT authentication
- **Schema Evolution**: JSON Schema versioning support
- **Multiple Clients**: Support for various authentication methods

### 3. Security
- **Encrypted Communication**: TLS/SSL for all API calls
- **Service Authentication**: JWT tokens for service-to-service
- **Flexible Access**: IP-based, API key, and OAuth support

### 4. Operational Excellence
- **Containerized Deployment**: Docker containers with proper networking
- **Configuration Management**: Secure secret handling
- **Monitoring Ready**: Structured logging and metrics endpoints

## Deployment Architecture

### Docker Configuration

#### API Server Configuration
```yaml
# configs/api-config.yaml
server:
  port: 8443
  tls:
    cert_file: /certs/server.crt
    key_file: /certs/server.key

authentication:
  jwt:
    secret_file: /secrets/jwt_secret
    token_expiry: 1h
  
  local_access:
    enabled: true
    allowed_networks:
      - "127.0.0.0/8"
      - "192.168.0.0/16"

cvs:
  cli_path: "/usr/local/bin/cvs"
  config_dir: "/tmp/cvs-configs"
  timeout: 3600s

logging:
  level: "info"
  format: "json"
  file: "/var/log/cvs-api.log"
```

#### Docker Compose Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  cvs-api:
    image: cvs-api:latest
    ports:
      - "8443:8443"
    volumes:
      - ./certs:/certs:ro
      - ./secrets:/secrets:ro
      - ./logs:/var/log
    environment:
      - CONFIG_FILE=/config/api-config.yaml
    networks:
      - cvs-network
    
  client-service:
    image: client-service:latest  
    environment:
      - CVS_API_URL=https://cvs-api:8443
      - JWT_SECRET_FILE=/secrets/jwt_secret
    networks:
      - cvs-network

networks:
  cvs-network:
    driver: bridge
```

### Security Considerations

1. **Certificate Management**
   - TLS certificates for HTTPS/WSS
   - Certificate rotation procedures
   - CA trust chain configuration

2. **Secret Management**  
   - JWT signing keys in secure storage
   - API keys for third-party access
   - Database credentials (if applicable)

3. **Network Security**
   - Firewall rules for API access
   - VPN requirements for external access
   - Internal network isolation

## Development Phases

### Phase 1: Core API Foundation (Weeks 1-2)
- ✅ Set up Go project structure with proper module initialization
- ✅ Implement basic REST endpoints for test suite discovery
- ✅ Create schema registry with dynamic discovery
- ✅ Add JWT authentication middleware

**Deliverables**:
- Working API server with basic endpoints
- JWT authentication system
- Schema discovery from existing config files

### Phase 2: CVS Integration & WebSocket (Weeks 3-4)
- ✅ Build configuration converter for JSON to CVS format
- ✅ Implement CVS CLI executor with real-time output capture
- ✅ Add WebSocket support for log streaming and status updates
- ✅ Create result processing with immediate notifications

**Deliverables**:
- End-to-end test execution capability
- Real-time WebSocket communication
- Comprehensive error handling

### Phase 3: Security & Production (Weeks 5-6)
- ✅ Implement TLS/SSL certificates and secure communication
- ✅ Add comprehensive multi-tier authentication
- ✅ Create Docker containers and deployment configurations
- ✅ Add monitoring, logging, and health checks

**Deliverables**:
- Production-ready deployment
- Complete security implementation
- Monitoring and operational tools

### Phase 4: Advanced Features (Weeks 7-8)
- 🔄 Advanced WebSocket features (file upload progress, multi-execution monitoring)
- 🔄 Webhook notifications for external systems integration
- 🔄 Advanced schema features (conditional validation, dynamic forms)
- 🔄 Performance optimizations and caching

**Deliverables**:
- Enhanced user experience features
- External system integrations
- Performance optimization

## Configuration Examples

### API Server Configuration
```yaml
server:
  port: 8443
  tls:
    cert_file: /certs/server.crt
    key_file: /certs/server.key

authentication:
  jwt:
    secret_file: /secrets/jwt_secret
    token_expiry: 1h
  
  local_access:
    enabled: true
    allowed_networks:
      - "127.0.0.0/8"
      - "192.168.0.0/16"

cvs:
  cli_path: "/usr/local/bin/cvs"
  config_dir: "/tmp/cvs-configs"
  timeout: 3600s
```

### Docker Deployment
```yaml
version: '3.8'
services:
  cvs-api:
    image: cvs-api:latest
    ports:
      - "8443:8443"
    volumes:
      - ./certs:/certs:ro
      - ./secrets:/secrets:ro
    environment:
      - CONFIG_FILE=/config/api-config.yaml
    
  client-service:
    image: client-service:latest  
    environment:
      - CVS_API_URL=https://cvs-api:8443
      - JWT_SECRET_FILE=/secrets/jwt_secret
```

This design provides a robust, secure, and scalable foundation for CVS REST API integration while maintaining clear separation of concerns between teams and services.

---

*Document Version: 1.0*  
*Last Updated: April 16, 2026*  
*Authors: CVS API Development Team*

> **Related Documentation**: For client integration patterns, see [Client-CVS API Integration Architecture](client-cvs-integration-architecture.md)