# Client-CVS API Integration Architecture

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Responsibilities](#component-responsibilities)
4. [Data Flow Patterns](#data-flow-patterns)
5. [Proxy Implementation](#proxy-implementation)
6. [WebSocket Integration](#websocket-integration)
7. [Data Persistence Strategy](#data-persistence-strategy)
8. [Security Architecture](#security-architecture)
9. [Implementation Plan](#implementation-plan)
10. [Deployment Topology](#deployment-topology)

---

## Overview

This document describes how client systems integrate with the CVS REST API Server through a proxy/gateway architecture pattern. This integration enables client users to discover CVS test suites, generate dynamic forms, execute tests, and monitor results in real-time.

### Integration Goals
- **Seamless User Experience**: Client users can access CVS functionality without direct CVS knowledge
- **Security Isolation**: CVS API Server remains internal while client system provides internet-facing interface
- **Dynamic Form Generation**: Automatic UI generation from CVS JSON schemas  
- **Real-time Monitoring**: Live test execution updates through WebSocket proxying
- **Scalable Architecture**: Support multiple concurrent users and test executions

### Architecture Pattern
**Backend for Frontend (BFF) + API Gateway Pattern**
```
Internet User → UI Frontend → UI Backend → CVS API Server → Python CVS CLI
```

---

## System Architecture

### High-Level Architecture

![Client-CVS Integration Architecture](images/client-cvs-integration-overview.png)
*Complete integration architecture showing all components and data flows*

> **Editable Source**: [client-cvs-integration-overview.drawio](images/client-cvs-integration-overview.drawio)

### Network Topology

```
┌─────────────────┐    ┌────────────────────────────────────┐
│   INTERNET      │    │          DATACENTER                │
│                 │    │                                    │
│  ┌───────────┐  │    │  ┌─────────────┐  ┌─────────────┐ │
│  │    UI     │  │    │  │     UI      │  │     CVS     │ │
│  │  Frontend │◄─┼────┼─►│   Backend   │◄─┤  API Server │ │
│  └───────────┘  │    │  └─────────────┘  └─────────────┘ │
│                 │    │         │              │          │
└─────────────────┘    │    ┌─────────┐   ┌──────────┐     │
                       │    │Client DB│   │ CVS CLI  │     │
                       │    └─────────┘   └──────────┘     │
                       └────────────────────────────────────┘
```

### Component Distribution

| **Component** | **Location** | **Purpose** | **Technology** |
|---------------|--------------|-------------|----------------|
| **UI Frontend** | Internet/CDN | User interface, form rendering | React/Vue.js |
| **UI Backend** | Datacenter | API gateway, authentication, proxy | Go/Node.js |
| **Client Database** | Datacenter | User data, execution history | PostgreSQL/MySQL |
| **CVS API Server** | Datacenter | Test execution, schema provision | Go |
| **Python CVS CLI** | Datacenter | Actual test execution | Python |

---

## Component Responsibilities

### UI Frontend Responsibilities
- **User Authentication**: Login, session management, user profiles
- **Dynamic Form Generation**: Render forms based on CVS JSON schemas
- **Form Validation**: Client-side validation using JSON schema rules
- **Test Management**: Display execution history, results, and status
- **Real-time Updates**: WebSocket client for live test monitoring
- **Error Handling**: User-friendly error messages and recovery

### UI Backend Responsibilities  
- **API Gateway**: Proxy all CVS API calls with authentication injection (80% pure proxy)
- **Authentication Proxy**: Convert user sessions to CVS JWT tokens
- **Selective Transformation**: Transform only specific endpoints for optimal UI experience (20% of cases)
- **WebSocket Proxying**: Multiplex CVS WebSocket to multiple frontend connections
- **Caching Layer**: Cache schemas, configurations, and metadata
- **Persistence**: Store execution history, user preferences, and audit logs
- **Error Translation**: Convert technical errors to user-friendly messages

### CVS API Server Responsibilities
- **Test Suite Discovery**: Provide JSON schemas for available test suites
- **Test Execution**: Orchestrate CVS CLI execution with real-time monitoring
- **Schema Generation**: Generate JSON schemas from CVS configuration files
- **Service Authentication**: JWT-based authentication for service clients
- **Real-time Communication**: WebSocket streaming of logs and status updates

---

## Data Flow Patterns

### 1. Schema Discovery Flow

![Schema Discovery Flow](images/schema-discovery-flow.png)
*Sequence diagram showing how UI Frontend discovers and renders test suite forms*

> **Editable Source**: [schema-discovery-flow.drawio](images/schema-discovery-flow.drawio)

### 2. Test Execution Flow

![Test Execution Flow](images/test-execution-flow.png)
*Complete test execution sequence from form submission to results*

> **Editable Source**: [test-execution-flow.drawio](images/test-execution-flow.drawio)

### 3. Real-time Monitoring Flow

![Real-time Monitoring Flow](images/real-time-monitoring-flow.png)
*WebSocket communication for live test monitoring and status updates*

> **Editable Source**: [real-time-monitoring-flow.drawio](images/real-time-monitoring-flow.drawio)

---

## Proxy Implementation

### HTTP API Proxy Layer

#### Basic Proxy Pattern
```javascript
// UI Backend - API Gateway Implementation
class CVSAPIProxy {
  constructor(cvsApiUrl, jwtSecret) {
    this.cvsApiUrl = cvsApiUrl;
    this.jwtManager = new JWTManager(jwtSecret);
  }

  // Pure proxy for 80% of endpoints - no payload transformation
  async proxyRequest(req, res, path) {
    const jwt = await this.jwtManager.getValidToken();
    
    // Pure proxy: forward request AS-IS with only JWT authentication
    const response = await fetch(`${this.cvsApiUrl}${path}`, {
      method: req.method,
      headers: {
        ...req.headers,
        'Authorization': `Bearer ${jwt}`,
        'Host': new URL(this.cvsApiUrl).host
      },
      body: req.method !== 'GET' ? req.body : undefined  // No transformation
    });
    
    res.status(response.status).json(await response.json());
  }
}
```

#### Selective Transformation Layer (20% of Cases)
```javascript
// UI Backend - ONLY for UI-specific endpoints, not core CVS operations
class UITransformationLayer {
  // Only used for /api/ui/* endpoints, NOT /api/cvs/* proxy endpoints
  async getFormConfig(suiteName) {
    // Get raw schema from CVS API via pure proxy
    const schema = await this.cvsProxy.getSchema(suiteName);
    
    // Transform ONLY for UI form generation - not for core CVS operations
    return {
      formSchema: this.convertToReactJsonSchema(schema),
      validationRules: this.extractValidationRules(schema),
      uiHints: this.generateUIHints(schema),
      sections: this.groupFieldsIntoSections(schema)
    };
  }

  convertToReactJsonSchema(cvsSchema) {
    const formFields = {};
    
    Object.entries(cvsSchema.properties).forEach(([key, prop]) => {
      formFields[key] = {
        type: this.mapFieldType(prop.type),
        label: this.generateLabel(key, prop.description),
        required: cvsSchema.required?.includes(key),
        validation: this.buildValidationRules(prop),
        helpText: prop.description
      };
      
      // Add enum as select options
      if (prop.enum) {
        formFields[key].options = prop.enum.map(value => ({
          value,
          label: this.humanizeEnumValue(value)
        }));
      }
    });
    
    return formFields;
  }
}
```

### API Strategy: 80% Pure Proxy + 20% Transformation

#### Pure Proxy Pattern (80% of Cases)
For core CVS operations, the UI Backend acts as a **pure proxy** with:
- **No payload transformation**: Request/response bodies forwarded AS-IS
- **Authentication only**: Adds JWT tokens, validates user sessions
- **URL translation**: `/api/cvs/v1/*` → `/api/v1/*`

**Used for**: Test execution, status checking, log retrieval, WebSocket connections

#### Selective Transformation (20% of Cases)  
Only for UI-specific optimizations:
- **Form generation**: Converting CVS schemas to UI-friendly form configs
- **Dashboard data**: Aggregating user-specific execution history
- **UI preferences**: User-customized views and settings

**Used for**: Dynamic form rendering, user dashboards, UI preferences

### API Endpoint Structure

#### UI Backend API Routes
```
/api/
├── auth/                    # Client native - user authentication
│   ├── login                # User login
│   ├── logout               # User logout
│   └── profile              # User profile management
├── users/                   # Client native - user management
│   ├── preferences          # User preferences
│   └── history              # User test history
├── cvs/                     # PURE PROXY to CVS API (80% - no transformation)
│   ├── suites/              # → /api/v1/suites (pure proxy)
│   ├── execute/             # → /api/v1/tests/execute (pure proxy)
│   └── executions/{id}/     # → /api/v1/executions/{id} (pure proxy)
└── ui/                      # TRANSFORMATION LAYER (20% - UI-specific)
    ├── test-forms/{suite}/  # UI-optimized form configurations (transformed)
    ├── execution-status/    # Enhanced execution status with UI context (transformed)
    └── user-dashboard/      # User-specific dashboard data (transformed)
```

---

## WebSocket Integration

### WebSocket Proxy Architecture

```javascript
class WebSocketProxy {
  constructor() {
    this.clientConnections = new Map();     // userId -> WebSocket
    this.cvsConnection = null;              // Single CVS connection
    this.executionSessions = new Map();     // executionId -> metadata
    this.messageQueue = new Map();          // userId -> message buffer
  }

  // Handle new client connection
  handleClientConnection(ws, userId, sessionToken) {
    // Authenticate user session
    if (!this.validateUserSession(sessionToken)) {
      ws.close(1008, 'Authentication failed');
      return;
    }

    // Store client connection
    this.clientConnections.set(userId, {
      socket: ws,
      userId: userId,
      connectedAt: new Date(),
      subscriptions: new Set()
    });

    // Establish CVS connection if needed
    this.ensureCVSConnection();

    // Handle client messages
    ws.on('message', (message) => this.handleClientMessage(userId, message));
    ws.on('close', () => this.handleClientDisconnection(userId));
  }

  // Route messages from CVS to appropriate clients
  async handleCVSMessage(message) {
    const { type, execution_id, data } = JSON.parse(message);
    
    // Store important events in database
    if (['status', 'error', 'completion'].includes(type)) {
      await this.persistExecutionEvent(execution_id, type, data);
    }

    // Find clients subscribed to this execution
    const subscribedClients = this.getSubscribedClients(execution_id);
    
    // Light transformation only for UI presentation (20% pattern)
    const uiMessage = this.transformMessageForUI(type, data);
    
    // Forward to subscribed clients
    subscribedClients.forEach(client => {
      if (client.socket.readyState === WebSocket.OPEN) {
        client.socket.send(JSON.stringify(uiMessage));
      }
    });
  }

  // Transform CVS messages for better UI experience (selective 20% transformation)
  transformMessageForUI(type, data) {
    switch (type) {
      case 'status':
        return {
          type: 'status',
          status: this.humanizeStatus(data.status),
          progress: data.progress,
          statusIcon: this.getStatusIcon(data.status),
          canCancel: ['running', 'queued'].includes(data.status)
        };
        
      case 'completion':
        return {
          type: 'completion',
          success: data.exit_code === 0,
          results: this.formatResults(data.results),
          duration: this.formatDuration(data.execution_time),
          downloadUrl: this.generateDownloadUrl(data.execution_id)
        };
        
      default:
        return { type, ...data };
    }
  }
}
```

### Connection Management

#### Client Connection Lifecycle
```javascript
// Frontend WebSocket client
class CVSWebSocketClient {
  constructor(userId, sessionToken) {
    this.userId = userId;
    this.sessionToken = sessionToken;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    const ws = new WebSocket(`wss://ui-backend/ws/cvs-monitor`, {
      headers: { 'Authorization': `Bearer ${this.sessionToken}` }
    });

    ws.onopen = () => {
      console.log('Connected to CVS monitor');
      this.reconnectAttempts = 0;
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    ws.onclose = () => this.handleReconnection();
    ws.onerror = (error) => this.handleError(error);

    return ws;
  }

  subscribeToExecution(executionId) {
    this.ws.send(JSON.stringify({
      action: 'subscribe',
      execution_id: executionId
    }));
  }

  handleMessage(message) {
    switch (message.type) {
      case 'status':
        this.updateExecutionStatus(message);
        break;
      case 'log':
        this.appendLogMessage(message);
        break;
      case 'completion':
        this.showExecutionResults(message);
        break;
    }
  }
}
```

---

## Data Persistence Strategy

### Client Database Schema

```sql
-- User execution tracking
CREATE TABLE test_executions (
  id VARCHAR(50) PRIMARY KEY,                -- CVS execution ID
  user_id INT NOT NULL,                      -- Client user ID
  suite_name VARCHAR(100) NOT NULL,
  display_name VARCHAR(200),                 -- User-friendly name
  configuration JSON NOT NULL,               -- Test configuration
  status VARCHAR(20) NOT NULL,               -- current status
  started_at TIMESTAMP NOT NULL,
  completed_at TIMESTAMP,
  final_results JSON,                        -- Final test results
  error_message TEXT,
  cvs_api_response JSON,                     -- Raw CVS API response
  user_notes TEXT,                           -- User-added notes
  INDEX idx_user_executions (user_id, started_at DESC),
  INDEX idx_execution_status (status, started_at)
);

-- WebSocket session management
CREATE TABLE websocket_sessions (
  session_id VARCHAR(50) PRIMARY KEY,
  user_id INT NOT NULL,
  execution_id VARCHAR(50),
  connected_at TIMESTAMP NOT NULL,
  last_activity TIMESTAMP NOT NULL,
  disconnected_at TIMESTAMP,
  client_info JSON,                          -- Browser, IP, etc.
  INDEX idx_active_sessions (user_id, connected_at)
);

-- Execution events log (for debugging and replay)
CREATE TABLE execution_events (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  execution_id VARCHAR(50) NOT NULL,
  event_type VARCHAR(50) NOT NULL,          -- status, error, completion
  event_data JSON,
  timestamp TIMESTAMP NOT NULL,
  INDEX idx_execution_events (execution_id, timestamp)
);

-- User preferences and caching
CREATE TABLE user_preferences (
  user_id INT PRIMARY KEY,
  favorite_suites JSON,                     -- Frequently used test suites
  default_configurations JSON,              -- Saved configurations
  ui_settings JSON,                         -- UI preferences
  notification_settings JSON,               -- Notification preferences
  updated_at TIMESTAMP NOT NULL
);

-- Schema and configuration caching
CREATE TABLE cached_schemas (
  suite_name VARCHAR(100) PRIMARY KEY,
  schema_version VARCHAR(50),
  json_schema JSON NOT NULL,
  ui_schema JSON,                           -- Transformed for UI
  cached_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  INDEX idx_schema_expiry (expires_at)
);
```

### Data Storage Rules

#### What to Store
```javascript
const persistentData = {
  // Always store
  executions: {
    metadata: "execution ID, user, suite, timestamps",
    configuration: "test parameters and cluster config",
    finalResults: "completion status, metrics, artifacts",
    errors: "error messages and stack traces"
  },
  
  // Store for user experience
  userPreferences: {
    favorites: "frequently used test suites",
    defaults: "saved configurations per suite",
    notifications: "user notification preferences"
  },
  
  // Store for performance
  cachedSchemas: {
    jsonSchemas: "CVS schemas with TTL",
    uiSchemas: "transformed schemas for forms",
    metadata: "schema versions and update timestamps"
  }
};
```

#### What NOT to Store
```javascript
const streamOnlyData = {
  // Too much data - stream only
  realTimeLogs: "verbose test execution logs",
  heartbeats: "WebSocket keepalive messages",
  progressUpdates: "frequent progress percentage updates",
  
  // Temporary data
  authTokens: "JWT tokens (regenerate as needed)",
  wsConnections: "WebSocket connection state",
  messageQueue: "in-flight message buffers"
};
```

---

## Security Architecture

### Authentication Boundaries

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Internet      │    │   Datacenter     │    │   Datacenter    │
│                 │    │                  │    │                 │
│  User Session   │───▶│  UI Backend      │───▶│  CVS API        │
│  (Cookies/JWT)  │    │  (Session→JWT)   │    │  (JWT Auth)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
   User Auth Layer      Authentication Proxy    Service Auth Layer
```

### Security Flow Details

#### 1. User Authentication (Internet → UI Backend)
```javascript
// UI Backend - User session management
class UserAuthenticationService {
  async authenticateUser(req, res, next) {
    const sessionToken = req.cookies.session_token || req.headers.authorization;
    
    if (!sessionToken) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    
    try {
      const userSession = await this.validateSession(sessionToken);
      req.user = userSession;
      next();
    } catch (error) {
      res.status(401).json({ error: 'Invalid session' });
    }
  }
  
  async validateSession(token) {
    // Validate against client user database
    const session = await db.query('SELECT * FROM user_sessions WHERE token = ?', [token]);
    if (!session || session.expires_at < new Date()) {
      throw new Error('Invalid or expired session');
    }
    return session;
  }
}
```

#### 2. Service Authentication (UI Backend → CVS API)
```javascript
// UI Backend - CVS service authentication
class CVSServiceAuthentication {
  constructor(jwtSecret, serviceId) {
    this.jwtSecret = jwtSecret;
    this.serviceId = serviceId;
    this.tokenCache = new Map();
  }
  
  async getServiceToken() {
    const cacheKey = 'cvs-service-token';
    const cachedToken = this.tokenCache.get(cacheKey);
    
    // Return cached token if valid
    if (cachedToken && cachedToken.expires > Date.now()) {
      return cachedToken.token;
    }
    
    // Generate new JWT for CVS API
    const payload = {
      sub: this.serviceId,
      aud: 'cvs-api',
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + (60 * 60), // 1 hour
      permissions: ['execute', 'read', 'websocket']
    };
    
    const token = jwt.sign(payload, this.jwtSecret);
    
    // Cache token
    this.tokenCache.set(cacheKey, {
      token,
      expires: Date.now() + (55 * 60 * 1000) // 55 minutes (5 min buffer)
    });
    
    return token;
  }
}
```

### Network Security Model

#### Internet-Datacenter Isolation
- **UI Frontend**: Served from CDN, communicates only with UI Backend
- **UI Backend**: Single entry point to datacenter, handles all authentication
- **CVS API Server**: Internal only, no direct internet access
- **CVS CLI**: Completely isolated, no network access

#### Security Controls
```yaml
# Network segmentation
firewall_rules:
  internet_to_ui_backend:
    ports: [443]
    protocols: [HTTPS, WSS]
    authentication: required
  
  ui_backend_to_cvs_api:
    ports: [8443]
    protocols: [HTTPS, WSS] 
    authentication: JWT
  
  cvs_api_to_cvs_cli:
    type: local_process
    authentication: none
```

---

## Implementation Plan

### Development Phases

#### Phase 1: Basic Proxy (Week 1-2)
- **HTTP Proxy**: Basic request forwarding with JWT injection
- **Authentication**: User session validation and CVS JWT generation
- **Error Handling**: Basic error translation and logging
- **Testing**: Integration tests with CVS API

**Deliverables:**
- Working HTTP proxy for CVS API endpoints
- JWT service authentication
- Basic error handling

#### Phase 2: UI Integration (Week 3-4)  
- **Schema Transformation**: Convert CVS schemas to UI-friendly format
- **Form Generation**: Dynamic React form components
- **Caching Layer**: Schema and response caching
- **User Management**: Execution history and preferences

**Deliverables:**
- Dynamic form generation from CVS schemas
- User execution tracking
- Performance optimized with caching

#### Phase 3: WebSocket Proxy (Week 5-6)
- **WebSocket Multiplexing**: Single CVS connection, multiple client connections
- **Real-time Updates**: Bi-directional message routing
- **Connection Management**: Robust reconnection and error handling
- **Message Transformation**: UI-optimized message formatting

**Deliverables:**
- Full real-time monitoring capability
- Scalable WebSocket architecture
- Production-ready connection management

#### Phase 4: Production Hardening (Week 7-8)
- **Security Review**: Comprehensive security testing
- **Performance Optimization**: Caching, connection pooling, monitoring
- **Deployment**: Docker containers, orchestration, CI/CD
- **Documentation**: Operational runbooks and troubleshooting guides

**Deliverables:**
- Production-ready deployment
- Monitoring and alerting
- Complete documentation

### Technical Stack Recommendations

| **Component** | **Recommended Technology** | **Rationale** |
|---------------|---------------------------|---------------|
| **UI Backend** | Node.js + Express/Fastify | Fast development, excellent WebSocket support |
| **Database** | PostgreSQL | JSON support, ACID compliance, scalability |
| **Caching** | Redis | High performance, built-in expiration |
| **WebSocket** | Socket.io / native WebSocket | Reliable with fallbacks, good client support |
| **Authentication** | jsonwebtoken + Passport | Standard libraries, flexible strategies |
| **HTTP Client** | axios / node-fetch | Full-featured HTTP client with interceptors |

---

## Deployment Topology

### Container Architecture

```yaml
# docker-compose.yml - Production deployment
version: '3.8'
services:
  ui-frontend:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./frontend/dist:/usr/share/nginx/html
      - ./certs:/certs:ro
    networks:
      - frontend-network
  
  ui-backend:
    image: ui-backend:latest
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - CVS_API_URL=https://cvs-api:8443
      - DATABASE_URL=postgresql://user:pass@client-db:5432/client
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_FILE=/secrets/jwt_secret
    volumes:
      - ./secrets:/secrets:ro
    networks:
      - frontend-network
      - backend-network
    depends_on:
      - client-db
      - redis
      - cvs-api
  
  client-db:
    image: postgres:15
    environment:
      - POSTGRES_DB=client
      - POSTGRES_USER=client_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - client-db-data:/var/lib/postgresql/data
    networks:
      - backend-network
    secrets:
      - db_password
  
  redis:
    image: redis:7-alpine
    networks:
      - backend-network
  
  cvs-api:
    image: cvs-api:latest
    ports:
      - "8443:8443"
    environment:
      - CONFIG_FILE=/config/api-config.yaml
    volumes:
      - ./cvs-certs:/certs:ro
      - ./cvs-secrets:/secrets:ro
    networks:
      - backend-network

volumes:
  client-db-data:

networks:
  frontend-network:
  backend-network:

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### Network Topology

```
Internet
    │
    ▼
┌─────────────┐
│ Load        │
│ Balancer    │    
│ (nginx)     │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│    UI       │────▶│    UI       │
│  Frontend   │     │  Backend    │
│ (Static)    │     │ (Node.js)   │
└─────────────┘     └─────────────┘
                          │
                          ▼
               ┌─────────────────────────┐
               │     Backend Network     │
               │  ┌─────────┬─────────┐  │
               │  │Client   │   CVS   │  │
               │  │   DB    │   API   │  │  
               │  │ (PG)    │  (Go)   │  │
               │  └─────────┴─────────┘  │
               └─────────────────────────┘
```

### Scaling Considerations

#### Horizontal Scaling Points
- **UI Frontend**: CDN distribution, multiple edge locations
- **UI Backend**: Multiple instances behind load balancer
- **Client Database**: Read replicas for execution history queries
- **Redis Cache**: Clustering for high availability
- **CVS API**: Multiple instances with shared storage

#### Performance Monitoring
```javascript
// Key metrics to monitor
const monitoringMetrics = {
  // Performance metrics
  response_times: ['p50', 'p95', 'p99'],
  throughput: ['requests_per_second', 'websocket_connections'],
  
  // Business metrics  
  test_executions: ['active_count', 'completion_rate', 'error_rate'],
  user_activity: ['active_sessions', 'form_submissions', 'execution_requests'],
  
  // Infrastructure metrics
  database: ['connection_pool', 'query_times', 'cache_hit_rate'],
  websockets: ['active_connections', 'message_rate', 'proxy_latency']
};
```

---

*Document Version: 1.0*  
*Last Updated: April 16, 2026*  
*Authors: Client Integration Team*

> **Related Documentation**: For CVS API Server specifications, see [CVS REST API Server](api-server.md)