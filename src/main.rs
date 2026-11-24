mod engine;
mod gguf;
mod metal;
mod sampler;
mod server;
mod session;
mod tokenizer;
mod unicode_tables;

use engine::Engine;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use sampler::Sampler;
use session::{Renderer, Session};
use std::io::{BufRead, Write};
use tokenizer::Tokenizer;

const DEFAULT_MODEL: &str = "~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf";

const USAGE: &str = "inkling - a hand-fused Metal inference engine for Thinking Machines Inkling 2 Mini

usage: inkling <command> [options]

commands:
  chat              interactive multi-turn chat (KV cache reused across turns)
  run -p TEXT       one prompt, streamed
  bench             greedy prefill + decode benchmark
  check --ref P     compare logits against a llama.cpp reference dump (tools/ref_logits)
  serve             OpenAI-compatible HTTP server: --host H --port P (default 127.0.0.1:8090), --alias NAME (repeatable;
                    the first is reported), --default-effort E (requests without reasoning_effort; default: the
                    template's, high), --cache-mib N (conversation cache, default 8192), -c CTX (default 131072)

options:
  -m, --model PATH  GGUF file (default: LM Studio's Inkling-2-Mini Q4_K_M)
  -p, --prompt TEXT prompt text (run/bench)      -f FILE  read the prompt from a file
  -n N              max new tokens (default 2048, bench 128)
  -c, --ctx N       context length (default 16384)
  --effort E        reasoning effort: none|minimal|low|medium|high|xhigh|<0..0.99> (default high)
  --system TEXT     system prompt
  --temp T          temperature (default 1.0; 0 = greedy, fully GPU-pipelined)
  --top-p P, --top-k K, --seed S
  --greedy          same as --temp 0
  --hide-thinking   do not print the reasoning stream
  --kv-disk         keep only half of the (exact) K/V cache in memory while generating; the older half lives in a
                    temporary file and is read back for every token (slower generation)
  --kv8             8-bit K/V cache: half the cache memory, NOT exact (the default fp16 cache is)
  --raw             do not apply the chat template; print special tokens";

struct Opts {
    model: String, prompt: String, n: usize, ctx: usize, raw: bool, effort: String, reference: String, quiet: bool, stop: usize,
    system: String, temp: f32, top_p: f32, top_k: usize, seed: u64, hide_thinking: bool, n_set: bool,
    // serve
    host: String, port: u16, aliases: Vec<String>, default_effort: Option<String>, cache_mib: usize, ctx_set: bool,
}

fn parse() -> (String, Opts) {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    let home = std::env::var("HOME").unwrap_or_default();
    let mut o = Opts { model: DEFAULT_MODEL.replace('~', &home), prompt: "Write a haiku about the ocean.".into(), n: 2048, ctx: 16384,
                       raw: false, effort: "high".into(), reference: String::new(), quiet: false, stop: 1, system: String::new(),
                       temp: 1.0, top_p: 0.95, top_k: 64, seed: 0, hide_thinking: false, n_set: false,
                       host: "127.0.0.1".into(), port: 8090, aliases: Vec::new(), default_effort: None, cache_mib: 8192, ctx_set: false };
    while let Some(a) = args.next() {
        let mut v = || args.next().unwrap_or_else(|| { eprintln!("missing value for {a}"); std::process::exit(2) });
        match a.as_str() {
            "-m" | "--model" => o.model = v(),
            "-p" | "--prompt" => o.prompt = v(),
            "-f" | "--file" => o.prompt = std::fs::read_to_string(v()).expect("read prompt file"),
            "-n" => { o.n = v().parse().unwrap(); o.n_set = true; }
            "-c" | "--ctx" => { o.ctx = v().parse::<usize>().unwrap().div_ceil(64) * 64; o.ctx_set = true; }
            "--host" => o.host = v(),
            "--port" => o.port = v().parse().unwrap(),
            "--alias" => o.aliases.push(v()),
            "--default-effort" => o.default_effort = Some(v()),
            "--cache-mib" => o.cache_mib = v().parse().unwrap(),
            "--raw" => o.raw = true,
            "--kv8" => engine::KV8.store(true, std::sync::atomic::Ordering::Relaxed),
            "--kv-disk" => engine::KV_DISK.store(true, std::sync::atomic::Ordering::Relaxed),
            "--effort" => o.effort = v(),
            "--system" => o.system = v(),
            "--temp" => o.temp = v().parse().unwrap(),
            "--top-p" => o.top_p = v().parse().unwrap(),
            "--top-k" => o.top_k = v().parse().unwrap(),
            "--seed" => o.seed = v().parse().unwrap(),
            "--greedy" => o.temp = 0.0,
            "--hide-thinking" => o.hide_thinking = true,
            "--ref" => o.reference = v(),
            "-q" => o.quiet = true,
            "--stop" => o.stop = v().parse().unwrap(),
            "-h" | "--help" => { println!("{USAGE}"); std::process::exit(0) }
            _ => { eprintln!("unknown option {a}\n\n{USAGE}"); std::process::exit(2) }
        }
    }
    if o.seed == 0 { o.seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64; }
    (cmd, o)
}

/// reasoning effort as rendered by the GGUF chat template
fn effort_str(e: &str) -> String {
    let v = match e.trim() {
        "none" => 0.0, "minimal" => 0.1, "low" => 0.2, "medium" => 0.7, "high" => 0.9, "xhigh" | "max" => 0.99,
        x => x.parse::<f64>().unwrap_or(0.9).clamp(0.0, 0.99),
    };
    if v == 0.0 { "0".into() } else { format!("{v}") }
}

fn first_turn(o: &Opts, user: &str) -> String {
    let mut s = String::new();
    if !o.system.is_empty() { s += &format!("<|message_system|><|content_text|>{}<|end_message|>", o.system); }
    s += &format!("<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>", effort_str(&o.effort));
    s += &format!("<|message_user|><|content_text|>{user}<|end_message|>");
    s
}

fn softcap(x: f32, c: f32) -> f32 { if c > 0.0 { c * (x / c).tanh() } else { x } }

fn run(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    let tk = Tokenizer::load(&eng.g);
    if std::env::var("INK_KV_PREALLOC").is_ok() { eng.reserve(o.ctx, false); }     // dev: no cache growth (to compare against)
    let mut sess = Session::new(&eng, &tk);
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, eng.softcap, o.seed);
    let text = if o.raw { o.prompt.clone() } else { first_turn(o, &o.prompt) };
    let toks = tk.encode(&text, true);
    eng.prepare(toks.len(), toks.len());       // working memory for this prompt, mapped before the prefill starts
    // dev: INK_SPLIT_LAST=k prefills all but the last k tokens, then those k; with k=1 that last batch is a single
    // token, which the batched prefill gets wrong on a long context (garbage at 21K) - INK_FEED_ONE feeds it through
    // a decode step instead, as `serve` does
    let first = if let Some(k) = std::env::var("INK_SPLIT_LAST").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&k| k > 0 && k < toks.len()) {
        sess.prefill(&toks[..toks.len() - k]);
        if std::env::var("INK_FEED_ONE").is_ok() { sess.prefill_any(&toks[toks.len() - k..]) } else { sess.prefill(&toks[toks.len() - k..]) }
    } else { sess.prefill(&toks) };
    let mut out = std::io::stdout();
    let mut r = Renderer::new(!o.hide_thinking, o.raw);
    sess.generate(first, o.n, &mut sampler, |t| { if !o.quiet { r.push(&tk, t, &mut out); } true });
    r.end(&mut out);
    let s = sess.stats;
    eprintln!("\n[inkling] prompt {} tok @ {:.1} tok/s | generated {} tok @ {:.2} tok/s{} | {} at {} tokens",
              s.prompt_tokens, s.prompt_tokens as f64 / s.prompt_time.as_secs_f64(), s.gen_tokens, s.gen_tokens as f64 / s.gen_time.as_secs_f64(),
              if sampler.greedy() { " (greedy)" } else { "" }, kv_note(&eng, sess.n_past), sess.n_past);
}

fn chat(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    let tk = Tokenizer::load(&eng.g);
    let mut sess = Session::new(&eng, &tk);
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, eng.softcap, o.seed);
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    eprintln!("[inkling] chat ready (effort {}, {}). empty line or Ctrl-D to quit, /reset to clear the context.",
              o.effort, if sampler.greedy() { "greedy".to_string() } else { format!("temp {}", o.temp) });
    let mut turn = 0;
    loop {
        print!("\n\x1b[1m> \x1b[0m"); out.flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 { break; }
        let line = line.trim_end_matches(['\n', '\r']).to_string();
        if line.trim().is_empty() { break; }
        if line.trim() == "/reset" { sess = Session::new(&eng, &tk); turn = 0; eprintln!("[context cleared]"); continue; }
        let text = if turn == 0 { first_turn(o, &line) } else { format!("<|message_user|><|content_text|>{line}<|end_message|>") };
        turn += 1;
        let before = sess.stats;
        let toks = tk.encode(&text, true);
        if sess.ctx_left() < toks.len() + 64 { eprintln!("[context full: /reset]"); continue; }
        let first = sess.prefill(&toks);
        let mut r = Renderer::new(!o.hide_thinking, o.raw);
        let max = o.n.min(sess.ctx_left().saturating_sub(4));
        sess.generate(first, max, &mut sampler, |t| { r.push(&tk, t, &mut out); true });
        r.end(&mut out);
        let s = sess.stats;
        let (pt, gt) = (s.prompt_tokens - before.prompt_tokens, s.gen_tokens - before.gen_tokens);
        eprintln!("\n\x1b[2m[{pt} prompt tok @ {:.0} tok/s · {gt} tok @ {:.1} tok/s · context {}/{} · {}]\x1b[0m",
                  pt as f64 / (s.prompt_time - before.prompt_time).as_secs_f64(), gt as f64 / (s.gen_time - before.gen_time).as_secs_f64(),
                  sess.n_past, eng.max_ctx, kv_note(&eng, sess.n_past));
    }
}

fn bench(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    if let Ok(v) = std::env::var("INK_SKIP") { eng.skip.set(v.parse().unwrap()); }
    let tk = Tokenizer::load(&eng.g);
    let n = if o.n_set { o.n } else { 128 };
    let text = if o.raw { o.prompt.clone() } else { first_turn(o, &o.prompt) };
    let toks = tk.encode(&text, true);
    let mut sampler = Sampler::new(0.0, 1, 1.0, eng.softcap, 0);
    // warm-up pass (pipelines, page mappings)
    { let mut s = Session::new(&eng, &tk); let f = s.prefill(&toks[..toks.len().min(64)]); s.generate(f, 8, &mut sampler, |_| true); }
    let mut sess = Session::new(&eng, &tk);
    eng.prepare(toks.len(), toks.len());       // as `run` does: working memory allocated and mapped before the clock starts
    let first = sess.prefill(&toks);
    sess.generate(first, n, &mut sampler, |_| true);
    let s = sess.stats;
    println!("prompt {:5} tok: {:8.1} ms  {:7.1} tok/s | decode {:4} tok: {:8.1} ms  {:6.2} tok/s",
             s.prompt_tokens, s.prompt_time.as_secs_f64() * 1e3, s.prompt_tokens as f64 / s.prompt_time.as_secs_f64(),
             s.gen_tokens, s.gen_time.as_secs_f64() * 1e3, s.gen_tokens as f64 / s.gen_time.as_secs_f64());
    println!("memory besides the weights: {} at {} tokens | prefill scratch {:.0} MiB while prefilling, {:.1} MiB while generating",
             kv_note(&eng, sess.n_past), sess.n_past, s.peak_scratch as f64 / 1048576.0, eng.mem().1 as f64 / 1048576.0);
    if eng.tier_on() {
        let (rb, rt) = eng.tier_io();
        println!("K/V file: {:.0} MiB read per token, {:.1} GiB/s, {:.1} ms of each token's {:.1} ms", rb as f64 / 1048576.0 / s.gen_tokens as f64,
                 rb as f64 / 1073741824.0 / rt.as_secs_f64(), rt.as_secs_f64() * 1e3 / s.gen_tokens as f64, s.gen_time.as_secs_f64() * 1e3 / s.gen_tokens as f64);
    }
}

/// the K/V cache as it is held right now (with --kv-disk: the part in memory and the part that is only in the file)
fn kv_note(eng: &Engine, n: usize) -> String {
    let mib = |b: usize| b as f64 / 1048576.0;
    if eng.tier_on() { format!("K/V cache {:.0} MiB in memory + {:.0} MiB on disk (all in memory: {:.0} MiB)", mib(eng.mem().0), mib(eng.kv_disk_bytes(n)), mib(eng.kv_exact_bytes(n))) }
    else { format!("K/V cache {:.0} MiB", mib(eng.mem().0)) }
}

fn check(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    let toks: Vec<u32> = std::fs::read_to_string(format!("{}.tokens", o.reference)).unwrap().lines().map(|l| l.trim().parse().unwrap()).collect();
    let raw = std::fs::read(format!("{}.logits", o.reference)).unwrap();
    let nv = eng.n_vocab;
    let reflog: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    // the dump may hold only the last positions (trimmed references): align it to the end of the token list
    assert!(reflog.len() % nv == 0 && reflog.len() <= toks.len() * nv);
    let first_ref = toks.len() - reflog.len() / nv;
    eng.reset();
    eng.set_tokens(0, &toks);
    // --pf N: prefill the first N tokens as one batch, then decode the rest one by one
    let pf: usize = std::env::var("INK_PF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut agree = 0;
    let mut worst = 0f32;
    let kv_disk = engine::KV_DISK.load(std::sync::atomic::Ordering::Relaxed);
    let mut bits_hash = 0xcbf29ce484222325u64;       // over the raw bits of every logit of every checked step
    for pos in 0..toks.len() {
        if pf > 0 && pos + 1 < pf { continue; }
        let cb = eng.gpu.cmd();
        if pf > 0 && pos + 1 == pf {
            let mut p0 = 0;
            while p0 < pf {
                let t = (pf - p0).min(eng.chunk);
                let cbp = eng.gpu.cmd();
                eng.encode_prefill(&cbp, p0, t, p0 + t == pf, false);
                cbp.commit();
                cbp.waitUntilCompleted();
                p0 += t;
            }
        } else if kv_disk {
            eng.gen_reserve(pos);
            eng.step(pos, true, false).waitUntilCompleted();
        } else { eng.encode_step(&cb, pos, true, false); }
        cb.commit();
        cb.waitUntilCompleted();
        // dev: INK_KVQ=kbits,vbits rounds the cache as it is written, to measure what a coarser cache would cost
        if let Ok(v) = std::env::var("INK_KVQ") {
            let (kb, vb) = v.split_once(',').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap();
            if pf > 0 && pos + 1 == pf { eng.kv_quant_sim(0, pf, kb, vb); } else { eng.kv_quant_sim(pos, pos + 1, kb, vb); }
        }
        let lg: Vec<f32> = eng.logits().iter().map(|&x| softcap(x, eng.softcap)).collect();
        for x in eng.logits() { bits_hash = (bits_hash ^ x.to_bits() as u64).wrapping_mul(0x100000001b3); }
        if pos < first_ref { continue; }
        let r = &reflog[(pos - first_ref) * nv..(pos - first_ref + 1) * nv];
        let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        let (a1, a2) = (am(&lg), am(r));
        let maxdiff = lg.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        // KL(ref || ours)
        let lse = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() };
        let (l1, l2) = (lse(&lg), lse(r));
        let kl: f32 = r.iter().zip(&lg).map(|(b, a)| { let pb = (b - l2).exp(); pb * ((b - l2) - (a - l1)) }).sum();
        if a1 == a2 { agree += 1; }
        worst = worst.max(kl);
        println!("pos {pos:3} tok {:6} argmax ours {a1:6} ref {a2:6} {} maxdiff {maxdiff:.4} KL {kl:.6}", toks[pos], if a1 == a2 { "OK " } else { "XX " });
    }
    println!("argmax agreement {agree}/{} worst KL {worst:.6}", toks.len() - first_ref.max(pf.saturating_sub(1)));
    eprintln!("logit bits hash {bits_hash:016x}");
    if eng.tier_on() {
        let (n, (rb, rt)) = (toks.len(), eng.tier_io());
        eprintln!("disk tier: {:.1} MiB in memory, {:.1} MiB on disk only (all in memory: {:.1} MiB); read {:.0} MiB in {:.0} ms",
                  eng.mem().0 as f64 / 1048576.0, eng.kv_disk_bytes(n) as f64 / 1048576.0, eng.kv_exact_bytes(n) as f64 / 1048576.0, rb as f64 / 1048576.0, rt.as_secs_f64() * 1e3);
    }
}

fn stats(name: &str, v: &[f32]) {
    let n = v.len();
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    println!("{name:>12} [{:9.4} {:9.4} {:9.4} ... {:9.4} {:9.4} {:9.4}] sum = {s:.6}", v[0], v[1], v[2], v[n - 3], v[n - 2], v[n - 1]);
}

fn debug(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    let tk = Tokenizer::load(&eng.g);
    let toks = tk.encode(&o.prompt, true);
    eng.reset();
    eng.set_tokens(0, &toks);
    for pos in 0..toks.len() {
        eng.stop.set(if pos + 1 == toks.len() { o.stop } else { usize::MAX });
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, pos, o.stop == usize::MAX, false);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let ar = eng.ar;
    stats("x0", &eng.arena(0, engine::E));
    stats("inp0", &eng.arena(ar.inp[0], engine::E));
    stats("q", &eng.arena(ar.q, 3072));
    stats("k", &eng.arena(ar.k, 1024));
    stats("v", &eng.arena(ar.v, 1024));
    stats("rel", &eng.arena(ar.rel, 384));
    stats("sink", &eng.arena(ar.sink, 24));
    stats("attn_g", &eng.arena(ar.attn_g, 3072));
    stats("ffn_inp", &eng.arena(engine::Arena::vec(eng.last_mid.get()), engine::E));
    stats("fin", &eng.arena(ar.fin, engine::E));
    stats("rlog", &eng.arena(ar.rlog, 128));
    stats("sel", &eng.arena(ar.sel, 16));
    stats("act", &eng.arena(ar.act, 8 * 1152));
    stats("l_out", &eng.arena(engine::Arena::vec(eng.last_out.get()), engine::E));
}

fn prof(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    let vary = std::env::var("INK_VARY").is_ok();
    let toks: Vec<u32> = (0..o.ctx.min(4096)).map(|i| if vary { ((i as u64 * 2654435761u64) % 150000) as u32 + 1000 } else { 1000 }).collect();
    eng.set_tokens(0, &toks);
    let pos = o.n.min(o.ctx - 64);   // profile at this context position
    let time = |skip: u32, reps: usize| -> f64 {
        eng.skip.set(skip);
        let mut ts = Vec::new();
        let cbs: Vec<_> = (0..reps).map(|r| { let cb = eng.gpu.cmd(); eng.encode_step(&cb, if vary { pos + r } else { pos }, true, false); cb.commit(); cb }).collect();
        for cb in &cbs { cb.waitUntilCompleted(); ts.push(cb.GPUEndTime() - cb.GPUStartTime()); }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[reps / 2] * 1e3
    };
    time(0, 5);
    let base = time(0, 15);
    println!("full step at pos {pos}: {base:.3} ms GPU ({:.1} tok/s)", 1e3 / base);
    let mut tot = 0.0;
    for (c, name) in engine::CLS_NAMES.iter().enumerate() {
        let t = time(1 << c, 9);
        tot += base - t;
        println!("  {name:>11}: {:7.3} ms ({:5.1}%)", base - t, (base - t) / base * 100.0);
    }
    let rest = time(u32::MAX, 9);
    println!("  everything skipped: {rest:.3} ms | sum of classes {tot:.3} ms");
}

fn prof_pf(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    eng.set_tokens(0, &(0..8192).map(|i| 1000 + (i * 7) % 5000).collect::<Vec<u32>>());
    if o.prompt.len() > 500 {
        let tk = Tokenizer::load(&eng.g);
        let pt = tk.encode(&first_turn(o, &o.prompt), true);
        println!("using prompt ({} tokens)", pt.len());
        eng.set_tokens(0, &pt);
    }
    eng.tail.set(false);      // profile full layers (the MoE classes re-use the routing of the last layer)
    let t = eng.chunk;
    let p0 = o.n.min(o.ctx - t);
    // fill the cache up to p0 so attention sees a realistic context
    let mut q = 0;
    while q < p0 { let n = (p0 - q).min(t); let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, q, n, false, false); cb.commit(); cb.waitUntilCompleted(); q += n; }
    let once = |skip: u32| -> f64 {
        eng.skip.set(skip);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, p0, t, true, true);
        cb.commit(); cb.waitUntilCompleted();
        (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
    };
    once(0); once(0);
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let base = med((0..5).map(|_| once(0)).collect());
    println!("prefill chunk of {t} tokens at p0={p0}: {base:.1} ms ({:.0} tok/s)", t as f64 / base * 1e3);
    { let tl = eng.moe_tiles(); let s: u32 = tl.iter().flatten().sum();
      println!("  last-layer MoE tiles: {} of 8 simdgroups, {} of 4; assignments {s}, 16-token slots used {}", tl[0].len(), tl[1].len(),
               tl.iter().flatten().map(|&c| c.div_ceil(16)).sum::<u32>() * 16); }
    // isolation: run only one class (buffers keep realistic contents, incl. MoE routing, from the full pass)
    let mut tot = 0.0;
    for (c, name) in engine::CLS_NAMES.iter().enumerate() {
        once(0);
        let d = med((0..3).map(|_| once(!(1u32 << c) & !(1 << 21))).collect());
        let empty = med((0..3).map(|_| once(u32::MAX)).collect());
        let d = d - empty;
        tot += d;
        if d > 0.3 { println!("  {name:>11}: {d:8.2} ms ({:5.1}%)", d / base * 100.0); }
    }
    println!("  sum of isolated classes: {tot:.1} ms");
}

/// dev: greedy-generate and log tokens + per-layer expert routing of every step (for speculation analysis)
fn routes(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    let tk = Tokenizer::load(&eng.g);
    let text = if o.raw { o.prompt.clone() } else { first_turn(o, &o.prompt) };
    let prompt = tk.encode(&text, true);
    let mut sess = Session::new(&eng, &tk);
    let mut tok = sess.prefill(&prompt);
    let mut out = String::new();
    for t in &prompt { out += &format!("P {t}\n"); }
    for _ in 0..o.n {
        let pos = sess.n_past;
        eng.set_tokens(pos, &[tok]);
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, pos, true, true);
        cb.commit(); cb.waitUntilCompleted();
        sess.n_past += 1;
        let r = eng.routes();
        out += &format!("G {tok} {}\n", r.iter().map(|x| x.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")).collect::<Vec<_>>().join(";"));
        tok = eng.token_at(pos + 1);
        if tok == tk.eos { break; }
    }
    std::fs::write(&o.reference, out).unwrap();
}

/// dev: cost of a batched pass of T tokens vs one decode step
fn vcost(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    eng.set_tokens(0, &(0..4096).map(|i| 1000 + (i * 7) % 5000).collect::<Vec<u32>>());
    let pos = 300usize;
    let time = |f: &dyn Fn(&metal::CmdBuf)| -> f64 {
        let mut ts = Vec::new();
        for _ in 0..12 { let cb = eng.gpu.cmd(); f(&cb); cb.commit(); cb.waitUntilCompleted(); ts.push(cb.GPUEndTime() - cb.GPUStartTime()); }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap()); ts[6] * 1e3
    };
    let step = time(&|cb| eng.encode_step(cb, pos, true, true));
    println!("decode step: {step:.2} ms");
    for t in [1usize, 2, 4, 8, 16] {
        let pf = time(&|cb| eng.encode_prefill(cb, pos, t, true, true));
        println!("batched pass T={t:2}: {pf:6.2} ms = {:4.2} steps", pf / step);
    }
}

fn main() {
    let (cmd, o) = parse();
    match cmd.as_str() {
        "check" => check(&o),
        "debug" => debug(&o),
        "prof" => prof(&o),
        "routes" => routes(&o),
        "vcost" => vcost(&o),
        "prof_pf" => prof_pf(&o),
        "kmoe" => {
            let eng = Engine::load(&o.model, 4096);

            println!("k_moe_gu   (prod): {:6.1} GB/s", eng.kbench_moe("k_moe_gu", (engine::FF / 4, 8), 128));
            println!("k_moe_down (prod): {:6.1} GB/s", eng.kbench_moe("k_moe_down", (engine::E / 4, 1), 64));
        }
        "kgemm" => {
            let eng = Engine::load(&o.model, 4096);
            let t: usize = std::env::var("INK_T").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
            for (n, bm, bn) in [("kgs_64_128_8", 64, 128), ("kgs_64_64_4", 64, 64), ("kgs_64_32_4", 64, 32), ("kgs_64_96_6", 64, 96), ("kgs_64_112_7", 64, 112), ("kgs_64_192_12", 64, 192), ("kgs_64_256_16", 64, 256), ("kgb_64_128_8", 64, 128), ("kgb_64_64_4", 64, 64), ("kg_iso0_3", 64, 64), ("kg_iso2_3", 64, 64)] {
                println!("{n:>16}: {:5.1} TFLOPs", eng.kbench_gemm(n, bm, bn, t));
            }
        }
        "kmoepf" => {
            // MoE GEMM kernels alone, on the real routing of the prompt's last layer
            let eng = Engine::load(&o.model, o.ctx);
            eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
            eng.tail.set(false);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            let t = pt.len().min(eng.chunk);
            eng.reset();
            eng.set_tokens(0, &pt);
            let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, 0, t, false, false); cb.commit(); cb.waitUntilCompleted();
            let tl = eng.moe_tiles();
            let (gu, dn) = eng.kbench_moe_pf(t);
            println!("T={t}: tiles {} x8sg {} x4sg | gate/up {gu:.1} TFLOPs, down {dn:.1} TFLOPs (real flops)", tl[0].len(), tl[1].len());
        }
        "ksg" => {
            // per-simdgroup tile ops: cost of a tile vs threadgroup size and number of simdgroups that have tokens
            let eng = Engine::load(&o.model, 4096);
            for (name, nsg, nact) in [("kgsg_8", 8, 8), ("kgsgf0_8", 8, 8), ("kgsgf0_8", 8, 6), ("kgsgf0_8", 8, 4), ("kgsgf0_8", 8, 2), ("kgsg_4", 4, 4),
                                      ("kgsgf0_4", 4, 4), ("kgsgf0_4", 4, 2), ("kgsgf0_4", 4, 1), ("kgsg_16", 16, 16), ("kgsgf0_16", 16, 16)] {
                let tf = eng.kbench_sgn(name, nsg, nact, 3840);
                println!("{name:>10} {nsg:2} simdgroups, {nact:2} with tokens: {tf:5.1} TFLOPs -> tile cost {:.2}", (16 * nact) as f64 / tf);
            }
        }
        "qclip" => {
            // dev: are any key dimensions unused by all three query heads of their group?
            let eng = Engine::load(&o.model, 64);
            let pb = unsafe { std::slice::from_raw_parts(eng.pb.contents().as_ptr() as *const f32, eng.pb.length() / 4) };
            for (i, l) in eng.layers.iter().enumerate() {
                let w = &pb[l.q_clip as usize / 4..l.q_clip as usize / 4 + 24 * 128];
                let mut eff: Vec<f32> = (0..8 * 128).map(|c| { let (g, d) = (c / 128, c % 128); (0..3).map(|h| w[(g * 3 + h) * 128 + d].abs()).fold(0f32, f32::max) }).collect();
                eff.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let zeros = eff.iter().filter(|&&x| x == 0.0).count();
                println!("layer {i:2} swa {}: max|qclip| over the 3 heads, per key dim: min {:.4} p5 {:.4} p25 {:.4} median {:.4} max {:.4}  exact zeros {zeros}",
                    l.swa as u8, eff[0], eff[51], eff[256], eff[512], eff[1023]);
            }
        }
        "lazytest" => {
            // dev: does a no-copy Metal buffer over anonymous memory commit pages only when they are touched, and what does
            // the GPU's first touch of a page cost?
            let src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void k_w(device uint * B [[buffer(0)]], constant uint2 & a [[buffer(1)]], uint gid [[thread_position_in_grid]]) { B[a.x + gid * a.y] += gid + 1; }\n";
            let gpu = metal::Gpu::new(src, &["k_w"]);
            let foot = || unsafe { let mut ri: libc::rusage_info_v4 = std::mem::zeroed(); libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V4, &mut ri as *mut _ as *mut libc::rusage_info_t); ri.ri_phys_footprint as f64 / 1048576.0 };
            let len = 1usize << 30;
            let f0 = foot();
            let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) } as *mut u8;
            let buf = gpu.buffer_nocopy(ptr, len);
            println!("footprint: start {f0:.1} MiB, after wrapping 1 GiB of untouched memory {:.1} MiB", foot());
            let run = |first: u32, stride: u32, n: usize| {
                let cb = gpu.cmd(); let e = metal::Enc::new(&cb);
                e.pso(&gpu.pso("k_w")); e.buf(&buf, 0, 0); e.bytes(&[first, stride], 1);
                e.dispatch((n.div_ceil(64), 1, 1), 64); e.end();
                let t = std::time::Instant::now(); cb.commit(); cb.waitUntilCompleted(); t.elapsed().as_secs_f64() * 1e3
            };
            let mib = |m: usize| (m << 18) as u32;      // MiB -> uint index
            let rd = |m: usize| unsafe { *(ptr.add(m << 20) as *const u32) };
            let mut last = foot();
            // pages of the mapping that physically exist (the footprint ledger does not see pages only the GPU has touched)
            let resident = || { let mut v = vec![0i8; len / 16384]; unsafe { libc::mincore(ptr as *mut libc::c_void, len, v.as_mut_ptr()); } v.iter().filter(|&&x| x & 1 != 0).count() as f64 * 16384.0 / 1048576.0 };
            let mut lastr = resident();
            let mut step = |what: &str, ms: f64| { let (f, r) = (foot(), resident()); println!("{what}: {ms:.2} ms, footprint {:+.2} MiB (now {f:.1}), resident in the mapping {:+.2} MiB (now {r:.1})", f - last, r - lastr); last = f; lastr = r; };
            step("GPU writes one word at 0", run(0, 1, 1));
            step("GPU writes one word at 0 again", run(0, 1, 1));
            step("GPU writes one word at 1 MiB", run(mib(1), 1, 1));
            step("GPU writes one word at 100 MiB", run(mib(100), 1, 1));
            step("GPU writes one word at 101 MiB", run(mib(101), 1, 1));
            step("GPU writes one word at 132 MiB", run(mib(132), 1, 1));
            step("GPU writes one word at 164 MiB", run(mib(164), 1, 1));
            step("GPU writes 4 MiB dense at 200 MiB", run(mib(200), 1, 1 << 20));
            step("GPU writes 4 MiB dense at 200 MiB again", run(mib(200), 1, 1 << 20));
            step("GPU writes 32 MiB dense at 300 MiB", run(mib(300), 1, 8 << 20));
            step("GPU writes 32 MiB dense at 300 MiB again", run(mib(300), 1, 8 << 20));
            for i in 0..2048usize { unsafe { *ptr.add((400 << 20) + i * 16384) = 1; } }
            step("CPU touches 32 MiB at 400 MiB", 0.0);
            step("GPU writes 32 MiB dense at 400 MiB (CPU-touched)", run(mib(400), 1, 8 << 20));
            step("GPU writes 32 MiB dense at 400 MiB again", run(mib(400), 1, 8 << 20));
            println!("readback: {} {} {} {} (expect 2 1 1 2)", rd(0), rd(1), rd(100), rd(200));
            // one fresh page per dispatch
            let mut tf = 0.0; let mut tw = 0.0;
            for i in 0..200u32 { tf += run(mib(600) + i * 4096, 1, 64); }
            for i in 0..200u32 { tw += run(mib(600) + i * 4096, 1, 64); }
            step("200 dispatches, one fresh page each", tf);
            println!("per dispatch: fresh page {:.1} us, touched page {:.1} us", tf * 5.0, tw * 5.0);
        }
        "kvprune" => {
            // dev: what does evicting far keys of the full-attention layers cost? Prefill the prompt, decode -n tokens with the
            // whole cache (the reference), then decode the same tokens again with only a share of the far keys kept.
            let eng = Engine::load(&o.model, o.ctx);
            eng.tail.set(false);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            let (n, d) = (pt.len(), if o.n_set { o.n } else { 48 });
            eng.reset(); eng.reserve(n + d + 1, false); eng.ensure_batch(n.min(eng.chunk)); eng.qcap_alloc(n);
            eng.set_tokens(0, &pt);
            let mut p0 = 0;
            while p0 < n { let t = (n - p0).min(eng.chunk); let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, p0, t, p0 + t == n, false); cb.commit(); cb.waitUntilCompleted(); p0 += t; }
            let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
            let first = am(eng.logits()) as u32;
            let snap = eng.snapshot();
            let run = |toks: Option<&[u32]>| -> (Vec<u32>, Vec<Vec<f32>>, f64) {
                eng.restore(&snap);
                let (mut tok, mut ts, mut lgs) = (first, Vec::new(), Vec::new());
                let t0 = std::time::Instant::now();
                for i in 0..d {
                    if let Some(tt) = toks { tok = tt[i]; }
                    eng.set_tokens(n + i, &[tok]);
                    let cb = eng.gpu.cmd(); eng.encode_step(&cb, n + i, true, false); cb.commit(); cb.waitUntilCompleted();
                    let lg: Vec<f32> = eng.logits().iter().map(|&x| softcap(x, eng.softcap)).collect();
                    ts.push(tok); tok = am(&lg) as u32; lgs.push(lg);
                }
                (ts, lgs, t0.elapsed().as_secs_f64() * 1e3 / d as f64)
            };
            let (toks, refl, ms0) = run(None);
            let (_, again, _) = run(Some(&toks));
            let same = refl.iter().zip(&again).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
            println!("{n} prompt tokens, {d} decode steps, {ms0:.2} ms per step with the whole cache (rerun bit-identical: {same})");
            let full_mib = |keys: usize| (8 * keys * 4096) as f64 / 1048576.0;
            let lse = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() };
            let nq: usize = std::env::var("INK_NQ").ok().and_then(|v| v.parse().ok()).unwrap_or(192);
            let keeps = std::env::var("INK_KEEP").unwrap_or("1.0,0.5,0.25,0.125,0.0".into());
            for mode in std::env::var("INK_MODE").unwrap_or("sum".into()).split(',') { for sum in [false, true] { for keep in keeps.split(',').map(|k| k.parse::<f64>().unwrap()) {
                let t0 = std::time::Instant::now();
                let (far, kept) = eng.prune_build(n, keep, sum, nq, mode);
                let tb = t0.elapsed().as_secs_f64();
                let (_, lg, ms) = run(Some(&toks));
                let (mut agree, mut worst, mut mean) = (0, 0f32, 0f32);
                for (a, r) in lg.iter().zip(&refl) {
                    let (l1, l2) = (lse(a), lse(r));
                    let kl: f32 = r.iter().zip(a).map(|(b, a)| { let pb = (b - l2).exp(); pb * ((b - l2) - (a - l1)) }).sum();
                    if am(a) == am(r) { agree += 1; }
                    worst = worst.max(kl); mean += kl / d as f32;
                }
                println!("score {mode:3} keep {:5.1}% of {far} far keys{}: full-attention cache {:.0} -> {:.0} MiB | argmax {agree}/{d}  KL mean {mean:.5} worst {worst:.5} | {ms:.2} ms/step (selection {tb:.1} s)",
                         keep * 100.0, if sum { " + summary key" } else { "              " }, full_mib(n + d), full_mib(kept + sum as usize + 1024 + d));
            } } }
            *eng.prune.borrow_mut() = None;
        }
        "kvscreen" => {
            let eng = Engine::load(&o.model, o.ctx);
            eng.tail.set(false);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            let n = pt.len().min(eng.chunk);
            eng.reset(); eng.reserve(n + 1, false); eng.ensure_batch(n);
            eng.set_tokens(0, &pt);
            let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, 0, n, false, false); cb.commit(); cb.waitUntilCompleted();
            println!("last layer, {n} keys, last 48 queries x 24 heads: share of keys within a margin of the best score");
            for (m, mean, worst) in eng.kv_screen(n, 48, &[8.0, 12.0, 17.0, 25.0]) {
                println!("  within {m:4.0} logits (weight > e^-{m:.0}): mean {:6.2}% of keys, worst head {:6.2}%", mean * 100.0, worst * 100.0);
            }
        }
        "kvdump" => {
            // dev: prefill the prompt and write the K/V payload to <ref>.swa / <ref>.full (to study its redundancy)
            let eng = Engine::load(&o.model, o.ctx);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            let mut sess = Session::new(&eng, &tk);
            sess.prefill(&pt);
            let (swa, full) = eng.kv_dump(pt.len());
            std::fs::write(format!("{}.swa", o.reference), &swa).unwrap();
            std::fs::write(format!("{}.full", o.reference), &full).unwrap();
            println!("{} tokens: sliding-window payload {:.1} MiB, full-attention payload {:.1} MiB", pt.len(), swa.len() as f64 / 1048576.0, full.len() as f64 / 1048576.0);
        }
        "kpeak" => {
            let eng = Engine::load(&o.model, 4096);
            for (n, nsg) in [("kgpeak_8_0", 8), ("kgpeak_8_576", 8), ("kgpeak_8_128", 8), ("kgpeak_8_32", 8), ("kgpeak_4_0", 4), ("kgpeak_1_0", 1)] {
                println!("{n:>13}: {:5.1} TFLOPs", eng.kbench_peak(n, nsg, 1024));
            }
        }
        "kmode" => {
            let eng = Engine::load(&o.model, 4096);
            for n in ["kgmode_0_0", "kgmode_1_0", "kgmode_0_1", "kgmode_1_1"] {
                let (tf, rel, mx) = eng.kbench_mode(n, 1024);
                println!("{n}: {tf:5.1} TFLOPs   relative RMS error {rel:.2e}   max error / output RMS {mx:.2e}");
            }
        }
        "bw" => { let eng = Engine::load(&o.model, 4096); eng.bw_weights(); }
        "kbench" => {
            let eng = Engine::load(&o.model, 4096);
            if std::env::var("INK_KB_Q4").is_ok() { for (n, nr, nsg) in [("kb_dev_2_2", 2, 2), ("kb_dev_2_4", 2, 4), ("kb_dev_4_2", 4, 2),
                                 ("kb_dev_4_4", 4, 4), ("kb_dev_8_2", 8, 2), ("kb_dev_1_4", 1, 4), ("kb_blk_2_4", 2, 4), ("kb_blk_4_4", 4, 4),
                                 ("kb_blk_8_2", 8, 2), ("kb_blk_4_8", 4, 8)] {
                println!("{n:>14}: {:6.1} GB/s", eng.kbench(n, nr, nsg));
            } }
            for (n, nr, nsg) in [("k_moe_down", 2, 2), ("kd_dev_1_2", 1, 2), ("kd_dev_1_4", 1, 4), ("kd_dev_1_8", 1, 8), ("kd_dev_2_2", 2, 2),
                                 ("kd_dev_2_4", 2, 4), ("kd_dev_2_8", 2, 8), ("kd_dev_4_4", 4, 4), ("kd_tg_2_8", 2, 8)] {
                println!("{n:>14}: all {:6.1} GB/s | Q8_0 {:6.1} | Q5_0 {:6.1}", eng.kbench_down(n, nr * nsg, 32 * nsg, 0),
                         eng.kbench_down(n, nr * nsg, 32 * nsg, 8), eng.kbench_down(n, nr * nsg, 32 * nsg, 6));
            }
        }
        "serve" => server::serve(server::ServeOpts {
            model: o.model.clone(), ctx: if o.ctx_set { o.ctx } else { 131072 }, host: o.host.clone(), port: o.port,
            aliases: if o.aliases.is_empty() { vec!["Inkling-2-Mini".into()] } else { o.aliases.clone() },
            default_effort: o.default_effort.clone(), cache_mib: o.cache_mib, top_k: 128,
        }),
        // dev: print the prompt `serve` renders for a chat request (JSON body in -f FILE)
        "render" => {
            let req: serde_json::Value = serde_json::from_str(&o.prompt).expect("request JSON");
            match server::render(&req, o.default_effort.as_deref()) { Ok(p) => print!("{p}"), Err(e) => { eprintln!("{e}"); std::process::exit(1) } }
        }
        "run" => run(&o),
        "chat" => chat(&o),
        "bench" => bench(&o),
        _ => println!("{USAGE}"),
    }
}
