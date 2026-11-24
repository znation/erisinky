//! `inkling serve`: an OpenAI-compatible HTTP server (POST /v1/chat/completions, streamed or not; GET /v1/models,
//! /health) over the single-sequence engine.
//!
//! * The prompt is rendered exactly as the GGUF's chat template does (tool declarations, tool calls, tool results,
//!   reasoning effort), and the output is parsed back from the protocol tokens into `reasoning_content`, `content`
//!   and OpenAI `tool_calls`.
//! * One engine thread owns the GPU and serves requests in arrival order; every connection has its own thread.
//! * Conversation cache: after prefilling all but the last token of a prompt, the context state is copied out
//!   (`Engine::save_state`). A later prompt that starts with those tokens - the next turn of the same conversation,
//!   whose history differs from what was generated (clients strip the reasoning) exactly from there on - restores
//!   it and prefills only the new tokens. Several conversations stay cached at once, least recently used dropped
//!   first beyond `--cache-mib`.

use crate::engine::{Engine, StateSnap};
use crate::sampler::Sampler;
use crate::session::Session;
use crate::tokenizer::Tokenizer;
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct ServeOpts {
    pub model: String,
    pub ctx: usize,
    pub host: String,
    pub port: u16,
    pub aliases: Vec<String>,
    pub default_effort: Option<String>,
    pub cache_mib: usize,
    pub top_k: usize,
}

// ------------------------------------------------------------------------------------------------ chat template

const EFFORTS: &[(&str, &str)] = &[("none", "0"), ("minimal", "0.1"), ("low", "0.2"), ("medium", "0.7"), ("high", "0.9"),
                                    ("xhigh", "0.99"), ("max", "0.99")];

fn effort_text(v: Option<&Value>, default: Option<&str>) -> Result<String, String> {
    let v = match v { Some(Value::Null) | None => default.map(|d| Value::String(d.to_string())), Some(x) => Some(x.clone()) };
    match v {
        None => Ok("0.9".into()),
        Some(Value::String(s)) => {
            let k = s.trim();
            if let Some((_, t)) = EFFORTS.iter().find(|(n, _)| *n == k) { return Ok(t.to_string()); }
            match k.parse::<f64>() { Ok(x) => num_effort(x), Err(_) => Err(format!("Unknown reasoning_effort: {s}")) }
        }
        Some(Value::Number(n)) => num_effort(n.as_f64().unwrap_or(0.9)),
        Some(other) => Err(format!("Unknown reasoning_effort: {other}")),
    }
}

fn num_effort(x: f64) -> Result<String, String> {
    if !(0.0..=0.99).contains(&x) { return Err("reasoning_effort must be in [0.0, 0.99]".into()); }
    Ok(if x == 0.0 { "0".into() } else { format!("{x}") })
}

/// text parts of a message's content: a string, or an array of strings / {type: text, text} parts (images become a
/// note: this engine has no vision encoder)
fn content_parts(c: &Value) -> Vec<String> {
    match c {
        Value::String(s) => vec![s.clone()],
        Value::Array(parts) => parts.iter().map(|p| match p {
            Value::String(s) => s.clone(),
            Value::Object(o) => match o.get("type").and_then(|t| t.as_str()) {
                None | Some("text") | Some("input_text") => o.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string(),
                Some("image") | Some("input_image") | Some("image_url") => "[image omitted: this server has no vision encoder]".into(),
                Some(t) => format!("[unsupported content part: {t}]"),
            },
            _ => String::new(),
        }).collect(),
        _ => Vec::new(),
    }
}

/// a copy with every object's keys in byte order (serde_json keeps insertion order here, as the template does for
/// tool-call argument values; only the tool declarations are sorted throughout)
fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), sorted(&m[k]))).collect())
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        x => x.clone(),
    }
}

/// A tool call's arguments as the template wants them: an object (clients send OpenAI's JSON string).
fn call_args(f: &Value) -> Map<String, Value> {
    match f.get("arguments") {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) { Ok(Value::Object(m)) => m, _ => Map::new() },
        _ => Map::new(),
    }
}

/// The prompt for a chat request: the GGUF chat template of Inkling 2 Mini, transcribed.
pub fn render(req: &Value, default_effort: Option<&str>) -> Result<String, String> {
    let msgs = req.get("messages").and_then(|m| m.as_array()).ok_or("messages must be an array")?;
    let effort = effort_text(req.get("reasoning_effort").or_else(|| req.get("chat_template_kwargs").and_then(|k| k.get("reasoning_effort"))),
                             default_effort)?;
    let mut s = String::new();
    if let Some(tools) = req.get("tools").and_then(|t| t.as_array()).filter(|t| !t.is_empty()) {
        let specs: Vec<Value> = tools.iter().map(|tool| {
            let f = tool.get("function").unwrap_or(tool);
            let nonempty = |v: Option<&Value>| v.filter(|x| !x.is_null() && *x != &json!("") && *x != &json!({})).cloned();
            json!({
                "description": nonempty(f.get("description")).unwrap_or(json!("")),
                "name": f.get("name").cloned().unwrap_or(json!("")),
                "parameters": nonempty(f.get("parameters")).unwrap_or(json!({})),
                "type": nonempty(tool.get("type")).unwrap_or(json!("function")),
            })
        }).collect();
        // tojson(sort_keys=true, separators=(",", ":")): every object's keys sorted, compact
        s += "<|message_system|>tool_declare<|content_xml|>";
        s += &serde_json::to_string(&sorted(&Value::Array(specs))).unwrap();
        s += "<|end_message|>";
    }
    let effort_msg = format!("<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>");
    let mut effort_done = false;
    for m in msgs {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let rtok = match role {
            "user" => "<|message_user|>", "assistant" => "<|message_model|>", "system" | "developer" => "<|message_system|>",
            "tool" => "<|message_tool|>", _ => return Err(format!("Unknown message role: {role}")),
        };
        if !effort_done && rtok != "<|message_system|>" { s += &effort_msg; effort_done = true; }
        let mut hdr = rtok.to_string();
        let name = m.get("name").and_then(|n| n.as_str()).filter(|n| !n.is_empty());
        if role == "tool" {
            if let Some(n) = name { hdr += n; }
            else if let Some(id) = m.get("tool_call_id").and_then(|i| i.as_str()).filter(|i| !i.is_empty()) {
                for prev in msgs {
                    if prev.get("role").and_then(|r| r.as_str()) != Some("assistant") { continue; }
                    for tc in prev.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
                        if tc.get("id").and_then(|x| x.as_str()) == Some(id) {
                            if let Some(n) = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) { hdr += n; }
                        }
                    }
                }
            }
        } else if role == "assistant" { if let Some(n) = name { hdr += n; } }

        let calls: Vec<&Value> = if role == "assistant" { m.get("tool_calls").and_then(|t| t.as_array()).map(|a| a.iter().collect()).unwrap_or_default() } else { Vec::new() };
        let reasoning = if role == "assistant" { m.get("reasoning_content").and_then(|r| r.as_str()).filter(|r| !r.is_empty()) } else { None };
        if let Some(r) = reasoning { s += &format!("{hdr}<|content_thinking|>{r}<|end_message|>"); }
        match m.get("content") {
            Some(Value::String(c)) => {
                if !c.is_empty() || !(role == "assistant" && (!calls.is_empty() || reasoning.is_some())) {
                    s += &format!("{hdr}<|content_text|>{c}<|end_message|>");
                }
            }
            Some(c @ Value::Array(a)) if !a.is_empty() => {
                for part in content_parts(c) { s += &format!("{hdr}<|content_text|>{part}<|end_message|>"); }
            }
            _ => {}
        }
        for tc in &calls {
            let f = tc.get("function").unwrap_or(tc);
            let fname = f.get("name").and_then(|n| n.as_str()).ok_or("tool call function name must be a string")?;
            let args = call_args(f);
            s += &format!("<|message_model|>{fname}<|content_invoke_tool_json|>{{\"name\":{},\"args\":{{", serde_json::to_string(fname).unwrap());
            // top-level keys sorted (dictsort, case-sensitive: byte order); values compact, nested keys in the client's order
            let mut keys: Vec<&String> = args.keys().collect();
            keys.sort();
            let body: Vec<String> = keys.iter().map(|k| format!("{}:{}", serde_json::to_string(k).unwrap(), serde_json::to_string(&args[*k]).unwrap())).collect();
            s += &body.join(",");
            s += "}}<|end_message|>";
        }
        if role == "assistant" { s += "<|content_model_end_sampling|>"; }
    }
    if !effort_done { s += &effort_msg; }
    Ok(s)
}

// ------------------------------------------------------------------------------------------------ output parsing

pub enum Event {
    Reasoning(String),
    Content(String),
    ToolCall { name: String, arguments: String },
    Done { finish: &'static str, prompt_tokens: usize, cached_tokens: usize, completion_tokens: usize, reasoning_tokens: usize },
    Error(String),
}

#[derive(PartialEq, Clone, Copy)]
enum Mode { Between, Header, Thinking, Text, ToolJson }

/// protocol tokens -> events; bytes of a mode are held until they are valid UTF-8
struct Parser { mode: Mode, header: Vec<u8>, body: Vec<u8>, pend: Vec<u8>, pub n_tools: usize, pub n_thinking: usize }

impl Parser {
    fn new() -> Parser { Parser { mode: Mode::Between, header: Vec::new(), body: Vec::new(), pend: Vec::new(), n_tools: 0, n_thinking: 0 } }

    fn flush_text(&mut self, out: &mut Vec<Event>) {
        let valid = match std::str::from_utf8(&self.pend) { Ok(_) => self.pend.len(), Err(e) => e.valid_up_to() };
        if valid == 0 { return; }
        let s = String::from_utf8(self.pend[..valid].to_vec()).unwrap();
        self.pend.drain(..valid);
        match self.mode { Mode::Thinking => out.push(Event::Reasoning(s)), _ => out.push(Event::Content(s)) }
    }

    /// returns true at the end of the turn
    fn push(&mut self, tk: &Tokenizer, t: u32, out: &mut Vec<Event>) -> bool {
        if tk.is_special(t) {
            let name = String::from_utf8_lossy(&tk.decode_bytes(t)).to_string();
            match name.as_str() {
                "<|message_model|>" => { self.end_part(out); self.mode = Mode::Header; self.header.clear(); }
                "<|content_thinking|>" => { self.mode = Mode::Thinking; }
                "<|content_text|>" => { self.mode = Mode::Text; }
                "<|content_invoke_tool_json|>" => { self.mode = Mode::ToolJson; self.body.clear(); }
                "<|end_message|>" => self.end_part(out),
                "<|content_model_end_sampling|>" => { self.end_part(out); return true; }
                // anything else (a user/system/tool header) means the model left its turn
                _ => { self.end_part(out); return true; }
            }
            return false;
        }
        let b = tk.decode_bytes(t);
        match self.mode {
            Mode::Header => self.header.extend_from_slice(&b),
            Mode::ToolJson => self.body.extend_from_slice(&b),
            Mode::Thinking | Mode::Text => {
                if self.mode == Mode::Thinking { self.n_thinking += 1; }
                self.pend.extend_from_slice(&b);
                self.flush_text(out);
            }
            // text outside a message (the model skipped the header) is still the answer
            Mode::Between => { self.mode = Mode::Text; self.pend.extend_from_slice(&b); self.flush_text(out); }
        }
        false
    }

    /// the turn ended (EOS or the token budget): close whatever part was open
    fn finish(&mut self, out: &mut Vec<Event>) { self.end_part(out); }

    fn end_part(&mut self, out: &mut Vec<Event>) {
        match self.mode {
            Mode::Thinking | Mode::Text => { self.flush_text(out); self.pend.clear(); }
            Mode::ToolJson => {
                let header = String::from_utf8_lossy(&self.header).trim().to_string();
                let raw = String::from_utf8_lossy(&self.body).to_string();
                let (name, arguments) = match serde_json::from_str::<Value>(raw.trim()) {
                    Ok(v) => (v.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()).unwrap_or(header.clone()),
                              v.get("args").map(|a| serde_json::to_string(a).unwrap()).unwrap_or_else(|| "{}".into())),
                    Err(_) => (header.clone(), raw),
                };
                out.push(Event::ToolCall { name: if name.is_empty() { header } else { name }, arguments });
                self.n_tools += 1;
            }
            _ => {}
        }
        self.mode = Mode::Between;
    }
}

// ------------------------------------------------------------------------------------------------ engine thread

struct Job { prompt: String, max_tokens: Option<usize>, temp: f32, top_p: f32, top_k: usize, seed: u64, use_cache: bool, ev: Sender<Event> }

struct Cached { toks: Vec<u32>, snap: StateSnap, used: u64 }

fn common_prefix(a: &[u32], b: &[u32]) -> usize { a.iter().zip(b).take_while(|(x, y)| x == y).count() }

fn engine_thread(o: &ServeOpts, jobs: Receiver<Job>, ready: Sender<Result<usize, String>>) {
    let eng = Engine::load(&o.model, o.ctx);
    let tk = Tokenizer::load(&eng.g);
    let mut sess = Session::new(&eng, &tk);
    let _ = ready.send(Ok(eng.max_ctx));
    // tokens whose state is in the engine right now
    let mut live: Vec<u32> = Vec::new();
    let mut cache: Vec<Cached> = Vec::new();
    let budget = o.cache_mib << 20;
    let mut tick = 0u64;

    for job in jobs {
        tick += 1;
        let t0 = Instant::now();
        let toks = tk.encode(&job.prompt, true);
        let n = toks.len();
        if n == 0 { let _ = job.ev.send(Event::Error("empty prompt".into())); continue; }
        if n + 2 >= eng.max_ctx {
            let _ = job.ev.send(Event::Error(format!("the prompt is {n} tokens; this server's context is {} tokens", eng.max_ctx)));
            continue;
        }

        // where to start: the live context or a cached conversation, whichever covers more of the prompt (never all
        // of it: the last prompt token is always prefilled, its logits start the reply)
        // `"cache_prompt": false` in a request: prefill it from scratch (tests compare the two paths)
        let live_ok = job.use_cache && sess.pending().is_none() && live.len() < n && common_prefix(&live, &toks) == live.len();
        let mut base = if live_ok { live.len() } else { 0 };
        let mut pick: Option<usize> = None;
        for (i, c) in cache.iter().enumerate() {
            if !job.use_cache { break; }
            if c.toks.len() < n && c.toks.len() > base && common_prefix(&c.toks, &toks) == c.toks.len() { base = c.toks.len(); pick = Some(i); }
        }
        let t_restore = Instant::now();
        if let Some(i) = pick {
            sess.restore(&cache[i].snap);
            cache[i].used = tick;
        } else if !live_ok || base == 0 {
            sess.clear();
            base = 0;
        }
        let restore_ms = t_restore.elapsed().as_secs_f64() * 1e3;
        live.truncate(0);
        live.extend_from_slice(&toks[..base]);

        // prefill up to the last token, keep that state, then the last token
        let t_pf = Instant::now();
        eng.prepare(n, n - base);
        if n - 1 > base {
            sess.prefill_any(&toks[base..n - 1]);
            live.extend_from_slice(&toks[base..n - 1]);
        }
        let mut save_ms = 0.0;
        if n - 1 > base || pick.is_none() {
            let ts = Instant::now();
            // a cached prefix of this prompt is the same conversation one turn back: replace it
            cache.retain(|c| !(c.toks.len() <= n - 1 && common_prefix(&c.toks, &toks) == c.toks.len()));
            let snap = eng.save_state(n - 1);
            cache.push(Cached { toks: toks[..n - 1].to_vec(), snap, used: tick });
            let mut total: usize = cache.iter().map(|c| c.snap.bytes()).sum();
            while total > budget && cache.len() > 1 {
                let (i, _) = cache.iter().enumerate().min_by_key(|(_, c)| c.used).unwrap();
                total -= cache[i].snap.bytes();
                cache.remove(i);
            }
            save_ms = ts.elapsed().as_secs_f64() * 1e3;
        }
        let first = sess.prefill_any(&toks[n - 1..]);
        live.push(toks[n - 1]);
        let pf_s = t_pf.elapsed().as_secs_f64();

        // generate
        let mut sampler = Sampler::new(job.temp, job.top_k, job.top_p, eng.softcap, job.seed);
        let max = job.max_tokens.unwrap_or(usize::MAX).min(sess.ctx_left().saturating_sub(2)).max(1);
        let mut parser = Parser::new();
        let mut gen: Vec<u32> = Vec::new();
        let mut ended = false;
        let mut gone = false;
        let t_gen = Instant::now();
        sess.generate(first, max, &mut sampler, |t| {
            gen.push(t);
            let mut evs = Vec::new();
            ended = parser.push(&tk, t, &mut evs);
            for e in evs { if job.ev.send(e).is_err() { gone = true; } }
            !ended && !gone
        });
        let gen_s = t_gen.elapsed().as_secs_f64();
        // the engine stops at EOS without passing it on: fewer tokens than the budget means the turn ended
        if !ended && !gone && gen.len() < max { ended = true; }
        { let mut evs = Vec::new(); parser.finish(&mut evs); for e in evs { let _ = job.ev.send(e); } }
        // the context now holds the prompt and the generated tokens that went through the model
        let mut all = toks.clone();
        all.extend_from_slice(&gen);
        all.truncate(sess.n_past);
        live = all;
        sess.drop_pending();
        let finish = if parser.n_tools > 0 { "tool_calls" } else if ended { "stop" } else { "length" };
        let _ = job.ev.send(Event::Done { finish, prompt_tokens: n, cached_tokens: base, completion_tokens: gen.len(), reasoning_tokens: parser.n_thinking });
        let cached_gib = cache.iter().map(|c| c.snap.bytes()).sum::<usize>() as f64 / (1u64 << 30) as f64;
        eprintln!("[serve] prompt {n} tok (cached {base}{}) prefill {} tok {:.0} ms ({:.0} tok/s) | decode {} tok {:.1} tok/s | \
                   restore {restore_ms:.0} ms save {save_ms:.0} ms | {} conversations {cached_gib:.2} GiB | {finish}{} | {:.1} s",
                  if pick.is_some() { ", restored" } else if base > 0 { ", live" } else { "" },
                  n - base, pf_s * 1e3, (n - base) as f64 / pf_s, gen.len(), gen.len() as f64 / gen_s,
                  cache.len(), if gone { " (client gone)" } else { "" }, t0.elapsed().as_secs_f64());
    }
}

// ------------------------------------------------------------------------------------------------ HTTP

struct Req { method: String, path: String, body: Vec<u8> }

fn read_req(s: &mut BufReader<TcpStream>) -> Option<Req> {
    let mut line = String::new();
    if s.read_line(&mut line).ok()? == 0 { return None; }
    let mut it = line.split_whitespace();
    let method = it.next()?.to_string();
    let path = it.next()?.split('?').next()?.to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if s.read_line(&mut h).ok()? == 0 { return None; }
        let h = h.trim_end();
        if h.is_empty() { break; }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") { len = v.trim().parse().ok()?; }
        }
    }
    let mut body = vec![0u8; len];
    s.read_exact(&mut body).ok()?;
    Some(Req { method, path, body })
}

fn respond(s: &mut TcpStream, code: u16, body: &Value) {
    let text = body.to_string();
    let reason = match code { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", 503 => "Service Unavailable", _ => "Error" };
    let _ = write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
}

fn error_body(msg: &str, kind: &str) -> Value { json!({"error": {"message": msg, "type": kind}}) }

fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() }

fn handle(mut stream: TcpStream, jobs: Sender<Job>, aliases: &[String], ctx: usize, default_effort: Option<&str>, top_k: usize) {
    let mut r = BufReader::new(stream.try_clone().unwrap());
    let Some(req) = read_req(&mut r) else { return };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") | ("GET", "/v1/health") => respond(&mut stream, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") | ("GET", "/models") => {
            let data: Vec<Value> = aliases.iter().map(|a| json!({"id": a, "object": "model", "created": 0, "owned_by": "erisinky",
                                                                  "context_length": ctx, "max_model_len": ctx})).collect();
            respond(&mut stream, 200, &json!({"object": "list", "data": data}));
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let body: Value = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return respond(&mut stream, 400, &error_body(&format!("invalid JSON: {e}"), "invalid_request_error")),
            };
            let prompt = match render(&body, default_effort) {
                Ok(p) => p,
                Err(e) => return respond(&mut stream, 400, &error_body(&e, "invalid_request_error")),
            };
            let f = |k: &str| body.get(k).and_then(|v| v.as_f64());
            let max_tokens = body.get("max_completion_tokens").or_else(|| body.get("max_tokens")).and_then(|v| v.as_u64()).map(|v| v as usize);
            // the model card's sampling: temperature 1.0 without truncation (top_k is a cap on this CPU sampler)
            let temp = f("temperature").unwrap_or(1.0) as f32;
            let top_p = f("top_p").unwrap_or(1.0) as f32;
            let top_k = body.get("top_k").and_then(|v| v.as_i64()).filter(|&k| k > 0).map(|k| k as usize).unwrap_or(top_k);
            let seed = body.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64);
            let stream_out = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
            let usage_chunk = body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(|v| v.as_bool()).unwrap_or(false);
            let model = aliases.first().cloned().unwrap_or_default();
            let (tx, rx) = channel();
            let use_cache = body.get("cache_prompt").and_then(|v| v.as_bool()).unwrap_or(true);
            if jobs.send(Job { prompt, max_tokens, temp, top_p, top_k, seed, use_cache, ev: tx }).is_err() {
                return respond(&mut stream, 503, &error_body("engine stopped", "server_error"));
            }
            let id = format!("chatcmpl-{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos());
            let created = now();
            if stream_out { stream_reply(&mut stream, rx, &id, created, &model, usage_chunk); }
            else { full_reply(&mut stream, rx, &id, created, &model); }
        }
        _ => respond(&mut stream, 404, &error_body("not found", "invalid_request_error")),
    }
}

fn usage(prompt: usize, cached: usize, completion: usize, reasoning: usize) -> Value {
    json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
           "prompt_tokens_details": {"cached_tokens": cached}, "completion_tokens_details": {"reasoning_tokens": reasoning}})
}

fn stream_reply(s: &mut TcpStream, rx: Receiver<Event>, id: &str, created: u64, model: &str, usage_chunk: bool) {
    // the first event decides the status: an error before any output is a plain 400
    let first = match rx.recv() { Ok(e) => e, Err(_) => return respond(s, 503, &error_body("engine failed", "server_error")) };
    if let Event::Error(e) = &first { return respond(s, 400, &error_body(e, "invalid_request_error")); }
    if write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").is_err() { return; }
    let chunk = |delta: Value, finish: Value| json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                                                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    let send = |s: &mut TcpStream, v: &Value| -> bool { write!(s, "data: {v}\n\n").and_then(|_| s.flush()).is_ok() };
    if !send(s, &chunk(json!({"role": "assistant", "content": ""}), Value::Null)) { return; }
    let mut n_calls = 0usize;
    let mut ev = Some(first);
    loop {
        let e = match ev.take() { Some(e) => e, None => match rx.recv() { Ok(e) => e, Err(_) => break } };
        let ok = match e {
            Event::Reasoning(t) => send(s, &chunk(json!({"reasoning_content": t}), Value::Null)),
            Event::Content(t) => send(s, &chunk(json!({"content": t}), Value::Null)),
            Event::ToolCall { name, arguments } => {
                let call = json!({"index": n_calls, "id": format!("call_{}_{}", n_calls, &id[id.len().saturating_sub(8)..]), "type": "function",
                                  "function": {"name": name, "arguments": arguments}});
                n_calls += 1;
                send(s, &chunk(json!({"tool_calls": [call]}), Value::Null))
            }
            Event::Done { finish, prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens } => {
                let _ = send(s, &chunk(json!({}), json!(finish)));
                if usage_chunk {
                    let _ = send(s, &json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
                                            "usage": usage(prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens)}));
                }
                let _ = write!(s, "data: [DONE]\n\n");
                break;
            }
            Event::Error(e) => { let _ = send(s, &error_body(&e, "server_error")); break; }
        };
        if !ok { break; }   // the client left: dropping the receiver stops the generation
    }
}

fn full_reply(s: &mut TcpStream, rx: Receiver<Event>, id: &str, created: u64, model: &str) {
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    for e in rx {
        match e {
            Event::Reasoning(t) => reasoning += &t,
            Event::Content(t) => content += &t,
            Event::ToolCall { name, arguments } => {
                calls.push(json!({"id": format!("call_{}_{}", calls.len(), &id[id.len().saturating_sub(8)..]), "type": "function",
                                  "function": {"name": name, "arguments": arguments}}));
            }
            Event::Error(e) => return respond(s, 400, &error_body(&e, "invalid_request_error")),
            Event::Done { finish, prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens } => {
                let mut msg = json!({"role": "assistant", "content": if content.is_empty() && !calls.is_empty() { Value::Null } else { json!(content) }});
                if !reasoning.is_empty() { msg["reasoning_content"] = json!(reasoning); }
                if !calls.is_empty() { msg["tool_calls"] = json!(calls); }
                return respond(s, 200, &json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                                               "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                               "usage": usage(prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens)}));
            }
        }
    }
    respond(s, 503, &error_body("engine failed", "server_error"));
}

pub fn serve(o: ServeOpts) {
    let (jobs_tx, jobs_rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel();
    let eo = ServeOpts { model: o.model.clone(), ctx: o.ctx, host: o.host.clone(), port: o.port, aliases: o.aliases.clone(),
                         default_effort: o.default_effort.clone(), cache_mib: o.cache_mib, top_k: o.top_k };
    std::thread::spawn(move || engine_thread(&eo, jobs_rx, ready_tx));
    let ctx = match ready_rx.recv() { Ok(Ok(c)) => c, _ => { eprintln!("[serve] the engine failed to start"); std::process::exit(1) } };
    let addr = format!("{}:{}", o.host, o.port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| { eprintln!("[serve] cannot listen on {addr}: {e}"); std::process::exit(1) });
    eprintln!("[serve] ready: http://{addr}/v1 | models {:?} | context {ctx} | conversation cache {} MiB | default effort {}",
              o.aliases, o.cache_mib, o.default_effort.as_deref().unwrap_or("template (high)"));
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        let (jobs, aliases, eff, top_k) = (jobs_tx.clone(), o.aliases.clone(), o.default_effort.clone(), o.top_k);
        std::thread::spawn(move || handle(s, jobs, &aliases, ctx, eff.as_deref(), top_k));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_like_the_template() {
        let req = json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "bash", "arguments": "{\"z\":1,\"cmd\":\"ls\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
                {"role": "assistant", "content": "done"}
            ],
            "tools": [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}}}],
            "reasoning_effort": "low"
        });
        let s = render(&req, None).unwrap();
        assert_eq!(s, concat!(
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"run\",\"name\":\"bash\",\"parameters\":{\"type\":\"object\"},\"type\":\"function\"}]<|end_message|>",
            "<|message_system|><|content_text|>sys<|end_message|>",
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>",
            "<|message_user|><|content_text|>hi<|end_message|>",
            "<|message_model|>bash<|content_invoke_tool_json|>{\"name\":\"bash\",\"args\":{\"cmd\":\"ls\",\"z\":1}}<|end_message|><|content_model_end_sampling|>",
            "<|message_tool|>bash<|content_text|>a.txt<|end_message|>",
            "<|message_model|><|content_text|>done<|end_message|><|content_model_end_sampling|>"));
    }

    #[test]
    fn effort_default_and_none() {
        let req = json!({"messages": [{"role": "user", "content": "x"}]});
        assert!(render(&req, None).unwrap().contains("Thinking effort level: 0.9<"));
        assert!(render(&req, Some("none")).unwrap().contains("Thinking effort level: 0<"));
        let req = json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": 0.5});
        assert!(render(&req, None).unwrap().contains("Thinking effort level: 0.5<"));
    }
}
