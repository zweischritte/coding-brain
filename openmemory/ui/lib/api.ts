import axios from 'axios';

/**
 * Configured axios instance with authentication.
 *
 * For development: Uses NEXT_PUBLIC_API_TOKEN from environment
 * For production: Replace with OAuth2/OIDC token from auth provider
 */
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8865',
});

// Add auth header to all requests
api.interceptors.request.use((config) => {
  const token = process.env.NEXT_PUBLIC_API_TOKEN;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
