import assert from 'node:assert/strict';
import test from 'node:test';
import { runSingleFlight } from '../src/polling.mjs';

test('slow polling shares one request and refreshes after completion', async () => {
  const requests = new Map();
  let finish;
  let calls = 0;
  const operation = () => { calls++; return new Promise(resolve => { finish = resolve; }); };
  const first = runSingleFlight(requests, 'core', operation);
  const second = runSingleFlight(requests, 'core', operation);
  await Promise.resolve();
  assert.equal(calls, 1);
  assert.equal(first, second);
  finish(42);
  assert.equal(await second, 42);
  assert.equal(requests.size, 0);
  assert.equal(await runSingleFlight(requests, 'core', async () => 43), 43);
});

test('failed requests clear the slot and preserve errors', async () => {
  const requests = new Map();
  await assert.rejects(runSingleFlight(requests, 'core', async () => { throw new Error('offline'); }), /offline/);
  assert.equal(requests.size, 0);
  assert.equal(await runSingleFlight(requests, 'core', async () => 'online'), 'online');
});
