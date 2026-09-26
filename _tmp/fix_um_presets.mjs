/**
 * 修复 UltraMath 预设与新版 dsh-persona schema 不兼容的问题：
 * persona row 的旧键 `text:` → 现行键 `prefix:`。
 *
 * 同时修补两处：
 *  1) 插件包内 presets/（真源；主机启动时由 dsh-ultramath 幂等同步覆盖目标侧）
 *  2) ~/.dsh/.agent-presets/（发现根；修补后当前运行中的主机无需重启即可用）
 *
 * 写入规则：UTF-8 无 BOM + LF 行尾（dsh-ultramath v0.6.2 起同步依赖 LF 规范化）。
 * 每个文件要求恰好 1 处匹配，否则跳过并报错，避免误改。
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';

const PRESETS = ['ultramath', 'ultramath-mathematician', 'ultramath-engineer', 'ultramath-writer', 'ultramath-reviewer'];
const ROOTS = {
  bundle: 'C:/Users/16669/.dsh/profiles/web/node_modules/dsh-ultramath/presets',
  synced: 'C:/Users/16669/.dsh/.agent-presets',
};
const BACKUP = 'E:/读研/26届数学建模比赛/_tmp/preset-backup';

const OLD = /^ {4}text: \|-$/;
const NEW = '    prefix: |-';

let changed = 0;
let problems = 0;

for (const [label, root] of Object.entries(ROOTS)) {
  for (const preset of PRESETS) {
    const file = join(root, preset, 'agent.cordis.yml');
    if (!existsSync(file)) {
      console.log(`  ? 不存在: ${file}`);
      problems += 1;
      continue;
    }
    const text = readFileSync(file, 'utf8').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const lines = text.split('\n');
    const hits = lines.map((l, i) => (OLD.test(l) ? i : -1)).filter((i) => i >= 0);

    // 备份原始字节
    const dest = join(BACKUP, label, preset, 'agent.cordis.yml');
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, readFileSync(file));

    if (hits.length !== 1) {
      console.log(`  ❌ ${label}/${preset}: 匹配 ${hits.length} 处（期望 1），已跳过`);
      problems += 1;
      continue;
    }
    const before = lines[hits[0]];
    lines[hits[0]] = NEW;
    writeFileSync(file, lines.join('\n'), 'utf8');
    console.log(`  ✅ ${label}/${preset}: ${before.trim()}  ->  ${NEW.trim()}`);
    changed += 1;
  }
}
console.log(`\n改写文件数: ${changed}，异常: ${problems}`);
console.log(`备份目录: ${BACKUP}`);
process.exitCode = problems === 0 ? 0 : 1;
