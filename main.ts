// A tiny stack virtual machine for make.fun's tiny stack ISA (version 1):
// runs every /input/<id>/program.bin on its input.bin exactly as the ISA
// specifies and writes /output/<id>/stdout.bin + exit.txt per program.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Immediate length per opcode; -1 marks a byte that is no opcode.
const IMM = new Int8Array(256).fill(-1);
for (const [op, n] of [[0x00, 0], [0x01, 1], [0x02, 4], [0x03, 0], [0x04, 0], [0x05, 0], [0x06, 0], [0x07, 0],
  [0x30, 2], [0x31, 2], [0x32, 2], [0x33, 2], [0x34, 0], [0x40, 0], [0x41, 0],
  [0x50, 0], [0x51, 0], [0x52, 0], [0x53, 0]] as const) IMM[op] = n;
for (let op = 0x10; op <= 0x22; op++) IMM[op] = 0;

// Values popped/pushed per opcode, for the stack checks before each step.
const POPS = new Int8Array(256);
const PUSHES = new Int8Array(256);
for (const [op, pops, pushes] of [[0x01, 0, 1], [0x02, 0, 1], [0x03, 1, 0], [0x04, 1, 2], [0x05, 2, 2],
  [0x06, 2, 3], [0x07, 3, 3], [0x17, 1, 1], [0x1b, 1, 1], [0x31, 1, 0], [0x32, 1, 0], [0x40, 1, 1],
  [0x41, 2, 0], [0x50, 1, 0], [0x51, 1, 0], [0x52, 0, 1], [0x53, 1, 0]] as const) {
  POPS[op] = pops; PUSHES[op] = pushes;
}
for (let op = 0x10; op <= 0x22; op++) if (op !== 0x17 && op !== 0x1b) { POPS[op] = 2; PUSHES[op] = 1; }

function run(prog: Uint8Array, input: Uint8Array): { out: Uint8Array; code: number } {
  const stack = new Int32Array(256);
  const calls = new Int32Array(64);
  const mem = new Int32Array(256);
  const bytes: number[] = [];
  let sp = 0, csp = 0, pc = 0, steps = 0, inAt = 0;
  const done = (code: number) => ({ out: Uint8Array.from(bytes), code });

  for (;;) {
    if (steps === 1_000_000) return done(10);            // step limit
    if (pc < 0 || pc >= prog.length) return done(9);     // pc outside the program
    const op = prog[pc];
    const imm = IMM[op];
    if (imm < 0) return done(5);                         // unknown opcode
    if (pc + imm >= prog.length && imm > 0) return done(9); // immediate cut off
    steps++;
    if (sp < POPS[op]) return done(2);                   // stack underflow
    if (sp - POPS[op] + PUSHES[op] > 256) return done(3); // stack overflow
    const next = pc + 1 + imm;
    pc = next;

    switch (op) {
      case 0x00: return done(0); // HALT
      case 0x01: stack[sp++] = (prog[next - 1] << 24) >> 24; break; // PUSH8, sign-extended
      case 0x02: // PUSH32, little-endian
        stack[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24);
        break;
      case 0x03: sp--; break; // POP
      case 0x04: stack[sp] = stack[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = stack[sp - 1]; stack[sp - 1] = stack[sp - 2]; stack[sp - 2] = t; break; } // SWAP
      case 0x06: stack[sp] = stack[sp - 2]; sp++; break; // OVER
      case 0x07: { // ROT: ( a b c -- b c a )
        const a = stack[sp - 3];
        stack[sp - 3] = stack[sp - 2]; stack[sp - 2] = stack[sp - 1]; stack[sp - 1] = a;
        break;
      }
      case 0x17: stack[sp - 1] = (-stack[sp - 1]) | 0; break; // NEG, wraps
      case 0x1b: stack[sp - 1] = ~stack[sp - 1]; break; // NOT
      case 0x30: case 0x31: case 0x32: { // JMP / JZ / JNZ, signed 16-bit offset from next
        const rel = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
        let take = op === 0x30;
        if (op !== 0x30) { const v = stack[--sp]; take = op === 0x31 ? v === 0 : v !== 0; }
        if (take) pc = next + rel;
        break;
      }
      case 0x33: // CALL: immediate is an absolute unsigned address
        if (csp === 64) return done(7);
        calls[csp++] = next;
        pc = prog[next - 2] | (prog[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return done(8);
        pc = calls[--csp];
        break;
      case 0x40: { // LOAD
        const addr = stack[sp - 1];
        if (addr < 0 || addr > 255) return done(6);
        stack[sp - 1] = mem[addr];
        break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const addr = stack[sp - 1], v = stack[sp - 2];
        if (addr < 0 || addr > 255) return done(6);
        sp -= 2; mem[addr] = v;
        break;
      }
      case 0x50: { // PRINT: decimal ASCII, no padding, no newline
        const s = String(stack[--sp]);
        if (bytes.length + s.length > 65_536) return done(11);
        for (let i = 0; i < s.length; i++) bytes.push(s.charCodeAt(i));
        break;
      }
      case 0x51: // PUTC
        if (bytes.length + 1 > 65_536) return done(11);
        bytes.push(stack[--sp] & 255);
        break;
      case 0x52: stack[sp++] = inAt < input.length ? input[inAt++] : -1; break; // IN
      case 0x53: return done(stack[--sp] & 255); // EXIT, low byte only
      default: { // the binary operators, 0x10..0x22 except NEG/NOT above
        const b = stack[sp - 1], a = stack[sp - 2];
        let r: number;
        switch (op) {
          case 0x10: r = (a + b) | 0; break;
          case 0x11: r = (a - b) | 0; break;
          case 0x12: r = Math.imul(a, b); break; // exact low 32 bits
          case 0x13: if (b === 0) return done(4); r = a === -2147483648 && b === -1 ? a : (a / b) | 0; break;
          case 0x14: if (b === 0) return done(4); r = b === -1 ? 0 : (a % b) | 0; break;
          case 0x15: if (b === 0) return done(4); r = Math.floor((a >>> 0) / (b >>> 0)) | 0; break; // DIVU
          case 0x16: if (b === 0) return done(4); r = ((a >>> 0) % (b >>> 0)) | 0; break; // MODU
          case 0x18: r = a & b; break;
          case 0x19: r = a | b; break;
          case 0x1a: r = a ^ b; break;
          case 0x1c: r = a << (b & 31); break;
          case 0x1d: r = a >> (b & 31); break; // arithmetic
          case 0x1e: r = (a >>> (b & 31)) | 0; break; // logical
          case 0x20: r = a === b ? 1 : 0; break;
          case 0x21: r = a < b ? 1 : 0; break;
          case 0x22: r = (a >>> 0) < (b >>> 0) ? 1 : 0; break; // LTU
          default: return done(5);
        }
        sp -= 2; stack[sp++] = r;
      }
    }
  }
}

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of readdirSync(inputDir).sort()) {
  const prog = new Uint8Array(readFileSync(join(inputDir, id, 'program.bin')));
  const input = new Uint8Array(readFileSync(join(inputDir, id, 'input.bin')));
  const r = run(prog, input);
  mkdirSync(join(outputDir, id), { recursive: true });
  writeFileSync(join(outputDir, id, 'stdout.bin'), r.out);
  writeFileSync(join(outputDir, id, 'exit.txt'), `${r.code}\n`);
}
