import { useEffect, useRef, useCallback, useState } from 'react';
import { wsUrl } from '../lib/api';

export function useWebSocket<T = unknown>(path: string, onMessage?: (data: T) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const onMessageRef = useRef(onMessage);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep callback ref fresh to avoid stale closures
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let destroyed = false;

    function connect() {
      if (destroyed) return;
      const url = wsUrl(path);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!destroyed) setConnected(true);
      };

      ws.onclose = () => {
        if (!destroyed) {
          setConnected(false);
          reconnectTimer.current = setTimeout(() => {
            if (!destroyed) connect();
          }, 5000);
        }
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as T;
          setLastMessage(data);
          onMessageRef.current?.(data);
        } catch { /* ignore */ }
      };
    }

    connect();

    return () => {
      destroyed = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [path]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, lastMessage, send };
}
