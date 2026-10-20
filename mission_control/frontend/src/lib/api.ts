const BASE = '';

let token: string | null = localStorage.getItem('mc_token');

export function setToken(t: string | null) {
  token = t;
  if (t) localStorage.setItem('mc_token', t);
  else localStorage.removeItem('mc_token');
}

export function getToken() { return token; }

export async function api<T = unknown>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(opts.headers as Record<string, string> || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...opts, headers });
  if (res.status === 401) {
    setToken(null);
    window.location.href = '/login';
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  }
  // Handle 204 No Content (e.g. DELETE responses)
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return {} as T;
  }
  return res.json();
}

export const get = <T = unknown>(path: string) => api<T>(path);
export const post = <T = unknown>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined });
export const del = <T = unknown>(path: string) => api<T>(path, { method: 'DELETE' });

export function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const sep = path.includes('?') ? '&' : '?';
  return `${proto}//${window.location.host}${path}${sep}token=${token}`;
}
