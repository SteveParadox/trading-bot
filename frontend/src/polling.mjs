// Share only requests that are still running. Never reuse a settled response.
export function runSingleFlight(requests, key, operation) {
  const current = requests.get(key);
  if (current) return current;
  const pending = Promise.resolve().then(operation).finally(() => requests.delete(key));
  requests.set(key, pending);
  return pending;
}
