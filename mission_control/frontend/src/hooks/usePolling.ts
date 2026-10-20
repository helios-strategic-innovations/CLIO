import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { get } from '../lib/api';

export function usePolling<T>(path: string, intervalMs: number = 30000, enabled: boolean = true) {
  const query = useQuery<T>({
    queryKey: ['poll', path],
    queryFn: () => get<T>(path),
    enabled,
    placeholderData: keepPreviousData,
    refetchInterval: enabled ? intervalMs : false,
    staleTime: intervalMs * 2,
  });

  return {
    data: query.data ?? null,
    error: query.error instanceof Error ? query.error.message : null,
    loading: enabled && query.isPending,
    isFetching: query.isFetching,
    isStale: query.isStale,
    dataUpdatedAt: query.dataUpdatedAt,
    refetch: () => { void query.refetch(); },
  };
}
