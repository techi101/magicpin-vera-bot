const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('bot.py', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

async function verify(ok) {
  const nodes = Object.fromEntries(['categories', 'merchants', 'customers', 'triggers', 'live', 'status'].map(k => [k, { textContent: '' }]));
  vm.runInNewContext(source, {
    document: { getElementById: k => nodes[k] },
    fetch: async () => ({ ok, json: async () => ({ contexts_loaded: { category: 5, merchant: 50, customer: 200, trigger: 100 } }) }),
  });
  await new Promise(resolve => setImmediate(resolve));
  if (ok) {
    assert.equal(nodes.categories.textContent, 5);
    assert.equal(nodes.merchants.textContent, 50);
    assert.equal(nodes.customers.textContent, 200);
    assert.equal(nodes.triggers.textContent, 100);
    assert.equal(nodes.status.textContent, 'Healthy');
  } else {
    assert.equal(nodes.status.textContent, 'Offline');
  }
}
Promise.all([verify(true), verify(false)]).then(() => console.log('PASS dashboard counts and health-error rendering'));
