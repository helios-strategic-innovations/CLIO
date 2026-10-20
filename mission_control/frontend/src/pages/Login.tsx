import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { LockKeyhole, LogIn } from 'lucide-react';
import { post, setToken } from '../lib/api';

export default function Login() {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await post<{ token: string }>('/api/auth/login', { password });
      setToken(res.token);
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="login-screen">
      <section className="login-panel" aria-labelledby="login-title">
        <img src="/AS1.jpg" alt="Atlas Strategies" className="login-lockup" />
        <div className="login-title-row">
          <LockKeyhole size={17} aria-hidden="true" />
          <h1 id="login-title">Mission Control</h1>
        </div>
        <form onSubmit={handleSubmit}>
          <label className="label login-label" htmlFor="atlas-password">Password</label>
          <input
            id="atlas-password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            placeholder="Enter password"
            autoFocus
            className="form-control"
          />
          {error && <p className="retry-note text-left" role="alert">{error}</p>}
          <button
            type="submit"
            disabled={loading || !password}
            className="primary-button login-submit"
          >
            <LogIn size={16} aria-hidden="true" />
            {loading ? 'Authenticating' : 'Sign In'}
          </button>
        </form>
      </section>
    </main>
  );
}
