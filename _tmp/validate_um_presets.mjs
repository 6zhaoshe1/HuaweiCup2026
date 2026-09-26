/**
 * 用 DSH 主机实际安装的插件 schema 校验 UltraMath 5 个预设的 agent.cordis.yml。
 *
 * 背景：dsh-agent-presets 在切换预设时逐行校验每个 row 的 config。
 * 本脚本复现同一校验：对每个 row 的 name（@deepseek-ai/*）加载该包导出的
 * schemastery Config 并调用之；cordis:group 递归其 config 数组。
 *
 * 用法：node _tmp/validate_um_presets.mjs
 */
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const CHECKOUT = 'C:/Users/16669/AppData/Local/npm-cache/_npx/1e7f6d9597241db0/node_modules';
const YAML_URL = pathToFileURL(join('C:/Users/16669/.dsh/profiles/web/node_modules/js-yaml/index.js')).href;
const yaml = (await import(YAML_URL)).default;

const ROOTS = {
  'bundle (插件包内)': 'C:/Users/16669/.dsh/profiles/web/node_modules/dsh-ultramath/presets',
  'synced (~/.dsh/.agent-presets)': 'C:/Users/16669/.dsh/.agent-presets',
};

const configCache = new Map();
/** 解析 @scope/name/subpath 形式的插件入口（含 package.json exports 子路径）。 */
function resolveEntry(pkg) {
  const parts = pkg.split('/');
  const scoped = pkg.startsWith('@');
  const dir = scoped ? join(CHECKOUT, parts[0], parts[1]) : join(CHECKOUT, parts[0]);
  const sub = scoped
    ? (parts.length > 2 ? './' + parts.slice(2).join('/') : '.')
    : (parts.length > 1 ? './' + parts.slice(1).join('/') : '.');
  const pkgJsonPath = join(dir, 'package.json');
  if (!existsSync(pkgJsonPath)) throw new Error(`package not installed: ${pkg}`);
  const pkgJson = JSON.parse(readFileSync(pkgJsonPath, 'utf8'));
  const exp = pkgJson.exports?.[sub];
  const rel = typeof exp === 'string' ? exp : exp?.default;
  return join(dir, rel ?? pkgJson.main ?? 'index.js');
}

async function loadConfig(pkg) {
  if (configCache.has(pkg)) return configCache.get(pkg);
  const mod = await import(pathToFileURL(resolveEntry(pkg)).href);
  const schema = mod.Config ?? null;
  configCache.set(pkg, schema);
  return schema;
}

async function validateRows(rows, prefix, problems) {
  for (const [i, row] of rows.entries()) {
    const where = `${prefix}[${i}] id=${row?.id ?? '?'} name=${row?.name ?? '?'}`;
    if (row?.name === 'cordis:group') {
      if (Array.isArray(row.config)) await validateRows(row.config, `${where}.config`, problems);
      continue;
    }
    if (typeof row?.name !== 'string' || !row.name.startsWith('@')) continue;
    let schema;
    try {
      schema = await loadConfig(row.name);
    } catch (error) {
      problems.push(`${where}: ${error.message}`);
      continue;
    }
    if (schema === null) continue;
    try {
      schema(row.config ?? {});
    } catch (error) {
      problems.push(`${where}: ${error.message}`);
    }
  }
}

let totalFailed = 0;
for (const [label, root] of Object.entries(ROOTS)) {
  console.log(`\n════ ${label} ════`);
  console.log(root);
  if (!existsSync(root)) {
    console.log('  (目录不存在)');
    continue;
  }
  for (const entry of readdirSync(root)) {
    const file = join(root, entry, 'agent.cordis.yml');
    if (!existsSync(file)) continue;
    const rows = yaml.load(readFileSync(file, 'utf8'));
    const problems = [];
    await validateRows(Array.isArray(rows) ? rows : [], 'root', problems);
    if (problems.length === 0) {
      console.log(`  ✅ ${entry}`);
    } else {
      totalFailed += 1;
      console.log(`  ❌ ${entry}`);
      for (const p of problems) console.log(`       - ${p}`);
    }
  }
}
console.log(`\n合计失败预设文件: ${totalFailed}`);
process.exitCode = totalFailed === 0 ? 0 : 1;
