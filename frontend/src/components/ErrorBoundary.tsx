import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * Catches render errors so one broken component does not blank the whole app.
 *
 * This matters most at demo time: a white screen reads as "it does not work",
 * while a visible error with a Reload button reads as "one panel broke". React
 * has no hook equivalent for this, so it has to be a class component.
 *
 * Note: it catches errors thrown during *render*. It does not catch errors in
 * event handlers or rejected promises - handle those where they happen (see
 * the try/catch in ChatPanel).
 */
interface Props {
  children: ReactNode;
  /** Optional custom fallback. Receives the error and a reset callback. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep the component stack - it is the part that actually tells you which
    // component threw. Swap in your error reporter here if you add one.
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  private reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback) return this.props.fallback(error, this.reset);

    return (
      <div className="mx-auto max-w-lg p-8">
        <div className="border-destructive/50 bg-card space-y-4 rounded-xl border p-6">
          <h2 className="font-semibold">Something broke in the UI</h2>
          <p className="text-muted-foreground text-sm">
            The rest of the app is fine. Check the browser console for the component stack.
          </p>
          <pre className="bg-muted max-h-40 overflow-auto rounded-md p-3 font-mono text-xs">
            {error.message}
          </pre>
          <div className="flex gap-2">
            <button
              onClick={this.reset}
              className="bg-primary text-primary-foreground h-9 rounded-md px-4 text-sm font-medium"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="h-9 rounded-md border px-4 text-sm font-medium"
            >
              Reload
            </button>
          </div>
        </div>
      </div>
    );
  }
}
