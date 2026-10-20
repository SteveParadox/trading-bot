export function runSingleFlight<T>(requests: Map<string, Promise<T>>, key: string, operation: () => Promise<T>): Promise<T>;
