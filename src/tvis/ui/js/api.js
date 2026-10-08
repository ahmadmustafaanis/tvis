// POST /api {method, params} → result. Tensor/image views are cached (runs are immutable once complete).

const cache = new Map();
const CACHEABLE = new Set(["tensor", "image", "source", "batch"]);

export async function api(method, params = {}) {
  const key = CACHEABLE.has(method) ? `${method}:${JSON.stringify(params)}` : null;
  if (key && cache.has(key)) return cache.get(key);
  const promise = request(method, params);
  if (key) {
    cache.set(key, promise);
    promise.catch(() => cache.delete(key));
  }
  return promise;
}

async function request(method, params) {
  const response = await fetch("/api", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, params }),
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body.result;
}

export function clearCache() {
  cache.clear();
}
