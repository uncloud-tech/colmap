// SPDX-License-Identifier: BSD-3-Clause
// Run with: node --test .github/tests/pr-agent-workflow.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const workflow = fs.readFileSync(
  path.join(__dirname, '../workflows/pr-agent.yml'), 'utf8');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

// Exercise the scripts embedded in the actual workflow, not copies of them.
function scriptFor(id) {
  const start = workflow.indexOf(`        id: ${id}\n`);
  assert.notEqual(start, -1, `Missing workflow step ${id}`);
  const step = workflow.slice(start).split(/\n      - name:/)[0];
  const script = step.split('          script: |\n')[1];
  assert.ok(script, `Missing script for ${id}`);
  return new AsyncFunction('github', 'context', 'core', 'process',
    script.split('\n').filter(line => line.startsWith('            '))
      .map(line => line.slice(12)).join('\n'));
}

function fixture(eventName = 'issue_comment', body = '/review') {
  const calls = [];
  const outputs = {};
  const state = { head: 'a'.repeat(40) };
  const github = { rest: {
    pulls: { get: async () => ({ data: { state: 'open', head: { sha: state.head } } }) },
    checks: {
      create: async args => { calls.push(['create', args]); return { data: { id: 123 } }; },
      update: async args => { calls.push(['update', args]); },
    },
  } };
  const context = {
    eventName, repo: { owner: 'example', repo: 'project' }, runId: 456,
    serverUrl: 'https://github.com',
    payload: { issue: { number: 39 }, comment: { body }, pull_request: { number: 39 } },
  };
  const core = {
    setOutput: (key, value) => { outputs[key] = String(value); },
    setFailed: message => { throw new Error(message); },
    info: () => {},
  };
  return { github, context, core, calls, outputs, state };
}

async function start(f) {
  await scriptFor('review-check')(f.github, f.context, f.core, { env: {} });
}
async function finish(f, outcome = 'success') {
  await scriptFor('review-result')(f.github, f.context, f.core, { env: {
    CHECK_ID: f.outputs['check-id'], REVIEW_HEAD_SHA: f.outputs['head-sha'],
    PR_NUMBER: f.outputs['pr-number'], AGENT_OUTCOME: outcome,
  } });
}

test('manual /review publishes the exact required check on the PR head', async () => {
  const f = fixture();
  await start(f);
  assert.equal(f.calls[0][1].name, 'review');
  assert.equal(f.calls[0][1].head_sha, f.state.head);
  assert.equal(f.calls[0][1].status, 'in_progress');
  await finish(f);
  assert.equal(f.calls[1][1].check_run_id, 123);
  assert.equal(f.calls[1][1].conclusion, 'success');
});

test('automatic review uses the same head-bound check', async () => {
  const f = fixture('pull_request');
  await start(f);
  assert.equal(f.calls[0][1].head_sha, f.state.head);
});

test('other commands cannot satisfy the review requirement', async () => {
  for (const body of ['/describe', '/improve', '/ask /review', '/reviewer']) {
    const f = fixture('issue_comment', body);
    await start(f);
    assert.equal(f.calls.length, 0, body);
  }
  const f = fixture('issue_comment', '/review incremental');
  await start(f);
  assert.equal(f.calls.length, 1);
});

test('agent failure, cancellation and skip never publish success', async () => {
  for (const outcome of ['failure', 'cancelled', 'skipped']) {
    const f = fixture();
    await start(f);
    await finish(f, outcome);
    assert.equal(f.calls[1][1].status, 'completed');
    assert.notEqual(f.calls[1][1].conclusion, 'success');
  }
});

test('API publication errors fail the step rather than disappearing', async () => {
  const f = fixture();
  f.github.rest.checks.create = async () => { throw new Error('permission denied'); };
  await assert.rejects(start(f), /permission denied/);
  assert.equal(f.outputs['check-id'], undefined);

  const g = fixture();
  await start(g);
  g.github.rest.checks.update = async () => { throw new Error('API unavailable'); };
  await assert.rejects(finish(g), /API unavailable/);
});

test('head movement cannot credit a different commit with a successful review', async () => {
  const f = fixture();
  await start(f);
  f.state.head = 'b'.repeat(40);
  await finish(f);
  assert.equal(f.calls[1][1].conclusion, 'cancelled');
});

test('workflow preserves trigger security and explicitly separates token limits', () => {
  assert.match(workflow, /checks: write/);
  assert.match(workflow, /github\.event\.sender\.type != 'Bot'/);
  assert.match(workflow, /github\.event\.pull_request\.head\.repo\.full_name == github\.repository/);
  assert.match(workflow, /comment\.author_association/);
  assert.doesNotMatch(workflow, /uses: actions\/checkout/);
  assert.match(workflow, /name: PR Agent execution/);
  assert.match(workflow, /if: always\(\) && steps\.review-check\.outputs\.check-id != ''/);
  assert.match(workflow, /AGENT_OUTCOME: \$\{\{ steps\.agent\.outcome \}\}/);
  assert.match(workflow, /FAIL_ON_TOOL_ERRORS: "true"/);
  const value = key => Number(workflow.match(new RegExp(`${key}: "(\\d+)"`))?.[1]);
  assert.equal(value('CONFIG__CUSTOM_MODEL_MAX_TOKENS'), 1048576);
  assert.equal(value('CONFIG__MAX_MODEL_TOKENS'), 1048576);
  assert.equal(value('CONFIG__MAX_OUTPUT_TOKENS'), 131072);
});
