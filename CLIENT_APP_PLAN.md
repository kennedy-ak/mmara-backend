# MMara Backend - Client Application Implementation Plan

## Context
The MMara backend is a multi-agent RAG (Retrieval-Augmented Generation) system for Ghanaian legal assistance. It provides REST APIs for authentication, chat, document management, and admin operations. Currently, there is no client application to test these features. This plan will create a React web application to serve as a testing interface for all backend endpoints.

## Implementation Plan

### Phase 1: Project Setup
**Location:** `C:\Users\User2\Desktop\Mmara\mmara-client` (new directory)

1. **Initialize Next.js Project**
   - Use Next.js 14+ with App Router
   - TypeScript for type safety
   - Tailwind CSS for styling
   - shadcn/ui component library for modern UI components

2. **Configure Backend Connection**
   - Create environment variable `NEXT_PUBLIC_API_URL` pointing to `http://localhost:8000`
   - Set up Axios instance with interceptors for JWT token handling
   - Configure CORS (already allowed in backend: `http://localhost:3000`)

### Phase 2: Core Services & Utilities
**Files to create:**
- `lib/api-client.ts` - Axios instance with auth interceptors
- `lib/auth.ts` - Authentication service (login, register, logout, token refresh)
- `lib/chat.ts` - Chat/RAG service
- `lib/admin.ts` - Admin service
- `lib/types.ts` - TypeScript types matching backend Pydantic models
- `contexts/AuthContext.tsx` - Auth context for managing user state

### Phase 3: Authentication UI
**Routes:** `/login`, `/register`, `/forgot-password`, `/reset-password`

1. **Login Page** (`app/login/page.tsx`)
   - Email/password form
   - Remember me checkbox
   - Link to registration and password reset
   - JWT token storage (localStorage + httpOnly cookie)

2. **Registration Page** (`app/register/page.tsx`)
   - Full name, email, phone, password fields
   - Password validation (8+ chars, uppercase, lowercase, digit)
   - Success message with redirect to login

3. **Password Reset** (`app/forgot-password/page.tsx`, `app/reset-password/page.tsx`)
   - Email request form
   - Token-based password reset form

### Phase 4: Main Chat Interface
**Route:** `/dashboard` (protected)

1. **Chat Layout**
   - Sidebar: Chat history sessions with search
   - Main area: Chat messages display and input
   - Top bar: User profile, category selector, logout

2. **Chat Features**
   - Real-time chat with AI responses
   - Session management (create, switch, delete)
   - Category selection (criminal, road_traffic, general)
   - Display citations from legal documents
   - Confidence and urgency indicators
   - Message feedback (1-5 rating + comment)

3. **Streaming Support**
   - WebSocket connection for real-time streaming responses
   - Progressive message display as chunks arrive
   - Stop generation button
   - Auto-reconnect on disconnect
   - Fallback to REST API if WebSocket unavailable

### Phase 5: Admin Panel
**Route:** `/admin` (protected, admin role only)

1. **Dashboard Overview**
   - Document statistics
   - User statistics
   - System health status

2. **Document Management**
   - Upload documents (PDF, DOCX)
   - List documents with filters (category, doc_type, status)
   - Delete documents
   - Reindex button for rebuilding vector index

3. **User Management**
   - List all users
   - View user details
   - Delete users

4. **Testing Tools**
   - Test retrieval endpoint
   - View system metrics

### Phase 6: Admin Setup & User Management
**Route:** `/admin/setup` (one-time setup, then restricted)

1. **Initial Admin Setup**
   - Create first admin user page
   - Special registration endpoint or flag
   - After first admin exists, this route redirects to login

2. **User Role Management**
   - Promote regular users to admin
   - Demote admin to regular user
   - View all users with their roles

### Phase 7: WebSocket Streaming Chat
**Route:** `/dashboard` (enhanced with streaming)

1. **WebSocket Integration**
   - Establish WebSocket connection with JWT token auth
   - Handle connection lifecycle (connect, disconnect, reconnect)
   - Display streaming responses in real-time
   - Fallback to standard POST if WebSocket fails

2. **Streaming UX**
   - Typing indicator while streaming
   - Progressive message rendering
   - Stop generation button
   - Auto-scroll to latest message

### Phase 8: Styling & UX
- Use shadcn/ui components (Button, Input, Card, Dialog, etc.)
- Dark mode support
- Responsive design (mobile-friendly)
- Loading states and error handling
- Toast notifications for success/error messages

## Critical Files & References

### Backend API Endpoints to Integrate
| File | Endpoints |
|------|-----------|
| `app/api/v1/auth.py` | `/register`, `/login`, `/refresh`, `/me`, `/logout`, password-reset |
| `app/api/v1/chat.py` | `/message`, `/history`, `/feedback`, WebSocket `/chat/stream` |
| `app/api/v1/admin.py` | `/documents`, `/reindex`, `/retrieve`, `/stats` |
| `app/api/v1/users.py` | User profile and management endpoints |

### Backend Types to Mirror
| File | Types |
|------|-------|
| `app/models/user.py` | UserCreate, UserResponse, Token |
| `app/api/v1/chat.py` | ChatRequest, ChatResponse, ChatHistoryResponse |
| `app/api/v1/admin.py` | DocumentInfo, DocumentStats, RetrievalRequest |

## Implementation Order
1. Project setup (Next.js + shadcn/ui)
2. API client and auth service
3. Authentication pages (login, register)
4. Main dashboard layout with protected routes
5. Chat interface with session management
6. Admin panel with document management
7. Styling polish and error handling

## Verification
1. Start backend: `cd mmara-backend && uvicorn main:app --reload`
2. Start client: `cd mmara-client && npm run dev`
3. Test flow:
   - Register new user
   - Login with credentials
   - Send chat message in different categories
   - View chat history
   - Upload document as admin
   - Test password reset flow

## Dependencies to Install
```bash
npx create-next-app@latest mmara-client --typescript --tailwind --app
cd mmara-client
npx shadcn-ui@latest init
npx shadcn-ui@latest add button input card dialog select tabs textarea toast form label scroll-area avatar badge
npm install axios react-hook-form @hookform/resolvers zod date-fns
```

## Additional Features
- **Admin Setup Flow**: First-time admin user creation page with special endpoint
- **WebSocket Streaming**: Real-time chat with progressive response rendering
- **Role-Based Access**: Protected routes based on user role (user/admin)
