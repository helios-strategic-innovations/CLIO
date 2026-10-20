export default function StatusDot({ status }: { status: string }) {
  const cls = status === 'active' ? 'status-dot-green' :
              status === 'inactive' || status === 'dead' ? 'status-dot-red' :
              'status-dot-yellow';
  return <span className={`status-dot ${cls}`} />;
}
